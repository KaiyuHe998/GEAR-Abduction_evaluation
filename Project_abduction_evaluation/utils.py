dfrom typing import *
from types   import FunctionType
from tqdm import tqdm
import copy
import signal
from functools import wraps

import ast, textwrap, re
import sys

import io, tokenize

import json
from datasets import Dataset
import random
import itertools
import math
import traceback
from multiprocessing import Pool
import numpy as np
import inspect
import textwrap
import os 
import pandas as pd

import pickle, resource, signal
from multiprocessing import get_context, Pool
import multiprocessing as mp
from collections import Counter

# AST counter function transformer
import ast, textwrap, types, random
from typing import Callable

def get_intersection_problem_id(dataset_split_A, dataset_split_B):
    assert dataset_split_A.keys() == dataset_split_B.keys(), f'''dataset_split_A and dataset_split_B have different dataset names'''
    held_out_problem_id = {key:[] for key in list(dataset_split_A.keys())}
    for dataset_name, problem_ids in dataset_split_B.items():
        for problem_id in problem_ids:
            if problem_id in dataset_split_A[dataset_name]:
                held_out_problem_id[dataset_name].append(problem_id)
    print({i[0]:len(i[1]) for i in held_out_problem_id.items()})
    return held_out_problem_id

def load_jsonl_dataset(dataset_path):
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dataset.append(obj)
    return dataset

def load_arc_2025_dataset():
    train_i_path = '../data/arc_2025/arc-agi_training_challenges.json'
    train_o_path = '../data/arc_2025/arc-agi_training_solutions.json'
    eval_i_path =  '../data/arc_2025/arc-agi_evaluation_challenges.json'
    eval_o_path =  '../data/arc_2025/arc-agi_evaluation_solutions.json'
    question_dataset_1 = load_jsonl_dataset(train_i_path)[0]
    answer_dataset_1 = load_jsonl_dataset(train_o_path)[0]
    question_dataset_2 = load_jsonl_dataset(eval_i_path)[0]
    answer_dataset_2 = load_jsonl_dataset(eval_o_path)[0]
    final_dataset = {}
    for key in question_dataset_1:
        assert key in answer_dataset_1, f'''{key} in question_dataset_1 but not in answer_dataset_1'''
        assert len(question_dataset_1[key]['test']) == len(answer_dataset_1[key])
        cur_problem_data = {key: {
            'train':[(item['input'], item['output']) for item in question_dataset_1[key]['train']],
            'test':[(item[0]['input'], item[1]) for item in zip(question_dataset_1[key]['test'],answer_dataset_1[key])]
        }}
        assert key not in final_dataset, f'''{key} duplicated'''
        final_dataset.update(cur_problem_data)
        
    for key in question_dataset_2:
        assert key in answer_dataset_2, f'''{key} in question_dataset_2 but not in answer_dataset_2'''
        assert len(question_dataset_2[key]['test']) == len(answer_dataset_2[key])
        cur_problem_data = {key: {
            'train':[(item['input'], item['output']) for item in question_dataset_2[key]['train']],
            'test':[(item[0]['input'], item[1]) for item in zip(question_dataset_2[key]['test'],answer_dataset_2[key])]
        }}
        assert key not in final_dataset, f'''{key} duplicated'''
        final_dataset.update(cur_problem_data)
    return final_dataset
        
def load_other_dataset(dataset_name):
    final_dataset = {}
    if dataset_name == 'mini_arc':
        dataset_file_name = 'miniarc.jsonl'
    else:
        dataset_file_name = f'{dataset_name}.jsonl'
    file_path = os.path.join('../data',dataset_file_name)
    dataset = load_jsonl_dataset(file_path)
    
    for item in dataset:
        problem_id = item['idx']
        cur_problem_data = {str(problem_id):
                            {
                             'train': [(i['input'],i['output']) for i in item['train']],
                             'test': [(i['input'],i['output']) for i in item['test']],
                            }
                           }
        assert problem_id not in final_dataset, f'''{problem_id} duplicated'''
        final_dataset.update(cur_problem_data)
    return final_dataset

# calculate metric:
def calculate_hill_numbers(outputs: List[Hashable]) -> Dict[str, float]:
    """Calculates Hill numbers (q=0, 1, 2) for a single list of outputs."""
    if not outputs:
        return {'q0_richness': 0.0, 'q1_shannon': 0.0, 'q2_simpson': 0.0}
    freq_counts = Counter(outputs)
    n = len(outputs)
    proportions = [count / n for count in freq_counts.values()]
    q0 = float(len(freq_counts))
    shannon_entropy = -sum(p * math.log(p) for p in proportions if p > 0)
    q1 = math.exp(shannon_entropy)
    simpson_index = sum(p**2 for p in proportions)
    q2 = 1.0 / simpson_index if simpson_index > 0 else 0.0
    return {'q0_richness': q0, 'q1_shannon': q1, 'q2_simpson': q2}

def _calculate_beta_dist_pair(pair_of_output_lists: Tuple[List, List]) -> float:
    """[Helper] Calculates Bray-Curtis dissimilarity between a pair of hypothesis outputs."""
    outputs1, outputs2 = pair_of_output_lists
    counts1 = Counter(outputs1)
    counts2 = Counter(outputs2)
    all_keys = counts1.keys() | counts2.keys()
    numerator = sum(abs(counts1.get(key, 0) - counts2.get(key, 0)) for key in all_keys)
    denominator = sum(counts1.get(key, 0) + counts2.get(key, 0) for key in all_keys)
    return numerator / denominator if denominator > 0 else 0.0

def _calculate_beta_struct_pair_jaccard_distance(pair_of_mapping_sets: Tuple[Set, Set]) -> float:
    """[Helper] Calculates Jaccard distance between a pair of hypothesis mappings."""
    set1, set2 = pair_of_mapping_sets
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1) + len(set2) - intersection_size
    jaccard_similarity = intersection_size / union_size if union_size > 0 else 1.0
    return 1.0 - jaccard_similarity

def calculate_avg_alpha_diversity(
    hypotheses_mappings: List[List[Tuple[Hashable, Hashable]]], 
    processes: int = 4  # Reduced for typical notebook environments
) -> Dict[str, float]:
    """Calculates average alpha diversity based on Hill numbers."""
    if not hypotheses_mappings:
        return {'q0_richness': 0.0, 'q1_shannon': 0.0, 'q2_simpson': 0.0}
    hypotheses_outputs = [[pair[1] for pair in mapping] for mapping in hypotheses_mappings]
    with Pool(processes=processes) as pool:
        results = pool.map(calculate_hill_numbers, hypotheses_outputs)
    return pd.DataFrame(results).mean().to_dict()

def calculate_gamma_diversity_dist(
    hypotheses_mappings: List[List[Tuple[Hashable, Hashable]]]
) -> Dict[str, float]:
    """Calculates gamma diversity based on Hill numbers. (output as element)"""
    if not hypotheses_mappings:
        return {'q0_richness': 0.0, 'q1_shannon': 0.0, 'q2_simpson': 0.0}
    all_outputs = [pair[1] for mapping in hypotheses_mappings for pair in mapping]
    return calculate_hill_numbers(all_outputs)

def calculate_gamma_diversity_struct(
    hypotheses_mappings: List[List[Tuple[Hashable, Hashable]]]
) -> Dict[str, float]:
    """Calculates gamma diversity based on Hill numbers. (mapping as element)"""
    if not hypotheses_mappings:
        return {'q0_richness': 0.0, 'q1_shannon': 0.0, 'q2_simpson': 0.0}
    all_outputs = [pair for mapping in hypotheses_mappings for pair in mapping]
    return calculate_hill_numbers(all_outputs)

def calculate_avg_beta_dist(
    hypotheses_mappings: List[List[Tuple[Hashable, Hashable]]], 
    processes: int = 4
) -> float:
    """Calculates average beta diversity (output distribution) based on Bray-Curtis."""
    num_hypotheses = len(hypotheses_mappings)
    if num_hypotheses < 2: return 0.0
    hypotheses_outputs = [[pair[1] for pair in mapping] for mapping in hypotheses_mappings]
    pairs = itertools.combinations(hypotheses_outputs, 2)
    with Pool(processes=processes) as pool:
        results = pool.map(_calculate_beta_dist_pair, pairs)
    return sum(results) / len(results) if results else 0.0

def calculate_avg_beta_struct(
    hypotheses_mappings: List[List[Tuple[Hashable, Hashable]]], 
    processes: int = 16
) -> float:
    """Calculates average beta diversity (structural difference) based on Jaccard distance."""
    num_hypotheses = len(hypotheses_mappings)
    if num_hypotheses < 2: return 0.0
    hypotheses_mapping_sets = [set(mapping) for mapping in hypotheses_mappings]
    pairs = itertools.combinations(hypotheses_mapping_sets, 2)
    with Pool(processes=processes) as pool:
        results = pool.map(_calculate_beta_struct_pair_jaccard_distance, pairs)
    return sum(results) / len(results) if results else 0.0

def calculate_avg_beta_struct_simplified(  # If there are not many hypothesis using this one will be much faster
    hypotheses_mappings: List[List[Tuple[Hashable, Hashable]]]
) -> float:
    """
    Calculates average beta diversity (structural difference) based on Jaccard distance
    using a single-threaded approach.
    """
    num_hypotheses = len(hypotheses_mappings)
    if num_hypotheses < 2:
        return 0.0

    # transfer list of lists into list of sets
    hypotheses_mapping_sets = [set(mapping) for mapping in hypotheses_mappings]
    
    # generate all possible combinations
    pairs = itertools.combinations(hypotheses_mapping_sets, 2)
    
    distances = (_calculate_beta_struct_pair_jaccard_distance(p) for p in pairs)
    
    total_distance = 0.0
    num_pairs = 0
    for d in distances:
        total_distance += d
        num_pairs += 1
        
    return total_distance / num_pairs if num_pairs > 0 else 0.0

def preprocess_code(raw: str) -> str:
    """
    tidy gpt generated code string
    """
    # 1️⃣ change '\\n' &'\\t' 
    code = raw.encode('utf-8').decode('unicode_escape')  # :contentReference[oaicite:2]{index=2}

    # 2️⃣ drop Markdown fence
    code = re.sub(r'```(?:python)?\s*([\s\S]+?)```', r'\1', code)

    # 3️⃣ unified new row
    code = code.replace('\r\n', '\n').replace('\r', '\n')

    # 4️⃣ drop ' "
    lines = code.split('\n')
    cleaned = []
    for ln in lines:
        m = re.match(r'^\s*"(.*)"\s*$', ln)
        if m:
            ln = m.group(1)
        if ln.strip():
            if ln[0] in ['\'', '\"']:
                cleaned.append(ln[1:])
            else:
                cleaned.append(ln)
        
    return '\n'.join(cleaned)
    
def _extract_single_function(ns: Dict[str, Any]) -> FunctionType:
    funcs = [obj for obj in ns.values() if callable(obj)]
    if len(funcs) != 1:
        raise ValueError(f"Expected exactly 1 function, got {len(funcs)}")
    return funcs[0]


def parse_string_code_to_function(string_code: str):
    try:
        tree  = ast.parse(string_code, filename="<generated>", mode="exec")
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(funcs) != 1:
            raise ValueError(f"Expected exactly one function, found {len(funcs)}")
        namespace = {}
        exec(compile(tree, "<generated>", "exec"), {}, namespace)
        return namespace[funcs[0].name]
    except Exception as e:
        return repr(e)

# ==============================================================================
# 1. Constants
# ==============================================================================
# Single allocation memory limit (e.g., for one large list)
MAX_MEMORY_BYTES = 5 * 1024 * 1024  # 5 MB
# Cumulative memory limit for a single function call chain
MAX_CUMULATIVE_MEMORY_BYTES = 50 * 1024 * 1024 # 50 MB
# Max total operations (function calls + loop iterations)
MAX_CALLS = 20000
# Max recursion depth
MAX_DEPTH = 1000
MAX_INT_VALUE = 10**5

# ==============================================================================
# 2. Helper Code
# ==============================================================================

# This helper code is injected into the generated code AST.
# It contains the functions that perform the security checks at runtime.
GUARD_HELPER_CODE = """
import sys
import copy

# Constants will be injected by the transformer
MAX_CALLS = {max_calls}
MAX_DEPTH = {max_depth}
MAX_MEMORY_BYTES = {max_mem}
MAX_CUMULATIVE_MEMORY_BYTES = {max_cumulative_mem}
MAX_INT_VALUE = {max_int_value}
# ✅ TARGETED FIX: Add a safety limit for integer values. dataset we are using at most will have integer 99 if generate a very large integer should stop to avoid haluting
# This prevents a variable from growing into a "number bomb".


def __guard_check_mult(op1, op2):
    global __guard_cumulative_mem

    # Case 1: Sequence multiplication (memory bomb)
    is_op1_seq = isinstance(op1, (list, tuple, str))
    is_op2_int = isinstance(op2, int)
    if is_op1_seq and is_op2_int:
        seq, num = op1, op2
    elif isinstance(op1, int) and isinstance(op2, (list, tuple, str)):
        seq, num = op2, op1
    # ✅ TARGETED FIX: Add a new case for integer * integer multiplication.
    elif isinstance(op1, int) and isinstance(op2, int):
        # This is the "overfitting" check you requested. It stops `p` from growing too large.
        try:
            # Check operands and result against our safety limit.
            if op1 > MAX_INT_VALUE or op2 > MAX_INT_VALUE or op1 * op2 > MAX_INT_VALUE:
                raise ValueError(f"Integer multiplication result would exceed the safety limit of {{MAX_INT_VALUE}}.")
        except OverflowError:
            raise ValueError("Integer value is too large to handle.")
        return op1 * op2
    else:
        # Not a guarded type, let it pass
        return op1 * op2

    # Logic for sequence multiplication continues here...
    if not seq or num <= 0:
        return seq * num
    try:
        element_size = sys.getsizeof(seq[0]) if seq else 1
        estimated_size = element_size * len(seq) * num
    except IndexError:
        return seq * num

    if estimated_size > MAX_MEMORY_BYTES:
        raise MemoryError(f"SINGLE memory allocation from multiplication would exceed the limit.")
    
    __guard_cumulative_mem += estimated_size
    if __guard_cumulative_mem > MAX_CUMULATIVE_MEMORY_BYTES:
        raise MemoryError(f"CUMULATIVE memory allocation from multiplication would exceed the limit.")
    
    return seq * num

# The __guard_check_add and __guard_fix_return_value functions remain the same...
def __guard_check_add(op1, op2):
    global __guard_cumulative_mem
    if not (isinstance(op1, (str, list)) and type(op1) == type(op2)):
        return op1 + op2
    if isinstance(op1, str):
        estimated_increase = len(op2)
        final_size = len(op1) + estimated_increase
    else: # list
        estimated_increase = 8 * len(op2)
        final_size = 8 * (len(op1) + len(op2))
    if final_size > MAX_MEMORY_BYTES:
        raise MemoryError("SINGLE memory from concatenation would exceed limit.")
    __guard_cumulative_mem += estimated_increase
    if __guard_cumulative_mem > MAX_CUMULATIVE_MEMORY_BYTES:
        raise MemoryError("CUMULATIVE memory from concatenation exceeds limit.")
    return op1 + op2

def __guard_fix_return_value(value):
    try:
        return copy.deepcopy(value)
    except Exception:
        return value
""".format(
    max_mem=MAX_MEMORY_BYTES,
    max_cumulative_mem=MAX_CUMULATIVE_MEMORY_BYTES,
    max_calls=MAX_CALLS,
    max_depth=MAX_DEPTH,
    max_int_value = MAX_INT_VALUE
)

# ==============================================================================
# 3. AST Transformer
# ==============================================================================

class GuardTransformer(ast.NodeTransformer):
    """
    This transformer injects security guards into a Python function's AST.
    It combines all the required defenses to stop potential halting.
    """
    
    class VariableUsageVisitor(ast.NodeVisitor):
        """Helper to check if a variable is used within an expression."""
        def __init__(self, variable_name):
            self.variable_name = variable_name
            self.is_used = False
        def visit_Name(self, node: ast.Name):
            if node.id == self.variable_name and isinstance(node.ctx, ast.Load):
                self.is_used = True
            self.generic_visit(node)

    def __init__(self):
        super().__init__()
        self.loop_ctr_stack = []

    # First visit the generated code
    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Injects helper code and global counters at the top of the module."""
        helper_ast = ast.parse(GUARD_HELPER_CODE)
        # Initialize global counters for tracking calls, depth, and memory
        init_calls = ast.Assign(targets=[ast.Name("__guard_rec_calls", ast.Store())], value=ast.Constant(0))
        init_depth = ast.Assign(targets=[ast.Name("__guard_rec_depth", ast.Store())], value=ast.Constant(0))
        init_cum_mem = ast.Assign(targets=[ast.Name("__guard_cumulative_mem", ast.Store())], value=ast.Constant(0))
        
        node.body = helper_ast.body + [init_calls, init_depth, init_cum_mem] + node.body
        self.generic_visit(node)
        return node

    # COMBINED STRATEGY 1: Rewrite exploitable list comprehensions.
    # This logic comes from version0.
    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        # Only handle simple forms like: [expr for var in range(n)]
        if len(node.generators) == 1:
            generator = node.generators[0]
            is_simple_range = (isinstance(generator.iter, ast.Call) and 
                               isinstance(generator.iter.func, ast.Name) and 
                               generator.iter.func.id == 'range' and 
                               len(generator.iter.args) == 1 and not generator.ifs)
            
            if is_simple_range and isinstance(generator.target, ast.Name):
                loop_var_name = generator.target.id
                visitor = self.VariableUsageVisitor(loop_var_name)
                visitor.visit(node.elt)
                # If the loop variable is NOT used, it's a repetition pattern
                if not visitor.is_used:
                    # Rewrite `[expr for _ in range(n)]` to `__guard_check_mult([expr], n)`
                    element_node, count_node = node.elt, generator.iter.args[0]
                    new_node = ast.Call(
                        func=ast.Name(id='__guard_check_mult', ctx=ast.Load()),
                        args=[ast.List(elts=[self.visit(element_node)], ctx=ast.Load()), self.visit(count_node)],
                        keywords=[]
                    )
                    return ast.copy_location(new_node, node)
        return self.generic_visit(node)

    # COMBINED STRATEGY 2: Intercept both `*` and `+` binary operations.
    # This combines logic from both versions.
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if isinstance(node.op, ast.Mult):
            # Route `*` to the multiplication guard
            new_node = ast.Call(func=ast.Name(id='__guard_check_mult', ctx=ast.Load()), args=[self.visit(node.left), self.visit(node.right)], keywords=[])
            return ast.copy_location(new_node, node)
        elif isinstance(node.op, ast.Add):
            # Route `+` to the addition/concatenation guard
            new_node = ast.Call(func=ast.Name(id='__guard_check_add', ctx=ast.Load()), args=[self.visit(node.left), self.visit(node.right)], keywords=[])
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)

    # COMBINED STRATEGY 3: Intercept `+=` for concatenation bombs.
    # This logic comes from version1.
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """
        Intercepts in-place operations. This is the crucial fix.
        """
        load_target = copy.deepcopy(node.target)
        load_target.ctx = ast.Load()
        
        # Guard for `+=` (concatenation bomb)
        if isinstance(node.op, ast.Add):
            func_name = '__guard_check_add'
        # ✅ TARGETED FIX: Add a guard for `*=` (number bomb).
        elif isinstance(node.op, ast.Mult):
            func_name = '__guard_check_mult'
        else:
            # Let other operators like -=, /= pass
            return self.generic_visit(node)
            
        # Rewrite `x op= y` to `x = __guard_func(x, y)`
        call_node = ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[load_target, self.visit(node.value)],
            keywords=[]
        )
        new_node = ast.Assign(targets=[node.target], value=call_node)
        return ast.copy_location(new_node, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Injects entry/exit boilerplate into each function for recursion/call counting."""
        if node.name.startswith('__guard_'):
            return node # Don't guard our own helper functions
        
        self.loop_ctr_stack.append(f"__guard_loop_ctr_{random.randrange(1<<30)}")
        
        # Standard boilerplate to increment counters and check limits
        globals_decl = ast.Global(["__guard_rec_calls", "__guard_rec_depth", "__guard_cumulative_mem"])
        rec_depth_inc = ast.AugAssign(target=ast.Name("__guard_rec_depth", ast.Store()), op=ast.Add(), value=ast.Constant(1))
        rec_calls_inc = ast.AugAssign(target=ast.Name("__guard_rec_calls", ast.Store()), op=ast.Add(), value=ast.Constant(1))
        
        check_calls = ast.If(test=ast.Compare(left=ast.Name("__guard_rec_calls", ast.Load()), ops=[ast.Gt()], comparators=[ast.Name("MAX_CALLS", ast.Load())]), body=[ast.Raise(exc=ast.Call(ast.Name("TimeoutError", ast.Load()), [ast.Constant("Exceeded max total operations")], []), cause=None)], orelse=[])
        check_depth = ast.If(test=ast.Compare(left=ast.Name("__guard_rec_depth", ast.Load()), ops=[ast.Gt()], comparators=[ast.Name("MAX_DEPTH", ast.Load())]), body=[ast.Raise(exc=ast.Call(ast.Name("TimeoutError", ast.Load()), [ast.Constant("Exceeded max recursion depth")], []), cause=None)], orelse=[])
        
        # Reset counters only on the first entry into the sandboxed code
        reset_block = ast.If(
            test=ast.Compare(left=ast.Name("__guard_rec_depth", ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(1)]),
            body=[ast.Assign(targets=[ast.Name("__guard_rec_calls", ast.Store())], value=ast.Constant(0)),
                  ast.Assign(targets=[ast.Name("__guard_cumulative_mem", ast.Store())], value=ast.Constant(0))],
            orelse=[]
        )
        
        # Ensure recursion depth is decremented even if an error occurs
        rec_depth_dec = ast.AugAssign(target=ast.Name("__guard_rec_depth", ast.Store()), op=ast.Sub(), value=ast.Constant(1))
        
        # The original function body is placed inside a try...finally block
        original_body = self.generic_visit(node).body
        try_block = ast.Try(body=original_body, handlers=[], orelse=[], finalbody=[rec_depth_dec])
        
        node.body = [globals_decl, rec_depth_inc, rec_calls_inc, reset_block, check_depth, check_calls, try_block]
        
        self.loop_ctr_stack.pop()
        return ast.fix_missing_locations(node)
    
    # visit_For and visit_While guard against infinite loops
    def visit_For(self, node: ast.For) -> ast.AST: return self._guard_loop(node)
    def visit_While(self, node: ast.While) -> ast.AST: return self._guard_loop(node)

    def _guard_loop(self, node: ast.AST) -> ast.AST:
        if not self.loop_ctr_stack: return self.generic_visit(node)
        
        # Increment total operation counter and check limit
        incr_shared_ctr = ast.AugAssign(target=ast.Name("__guard_rec_calls", ast.Store()), op=ast.Add(), value=ast.Constant(1))
        check_loop = ast.If(test=ast.Compare(left=ast.Name("__guard_rec_calls", ast.Load()), ops=[ast.Gt()], comparators=[ast.Name("MAX_CALLS", ast.Load())]), body=[ast.Raise(exc=ast.Call(ast.Name("TimeoutError", ast.Load()), [ast.Constant("Exceeded max total operations")], []), cause=None)], orelse=[])
        
        # Insert checks at the beginning of the loop body
        node.body.insert(0, check_loop)
        node.body.insert(0, incr_shared_ctr)
        
        self.generic_visit(node)
        return node
    
    def visit_Return(self, node: ast.Return) -> ast.AST:
        """Wraps return values to prevent leaking mutable state."""
        if not node.value:
            return node
        
        new_value_node = ast.Call(
            func=ast.Name(id='__guard_fix_return_value', ctx=ast.Load()),
            args=[node.value],
            keywords=[]
        )
        node.value = new_value_node
        return ast.fix_missing_locations(node)


# ==============================================================================
# 4. Main Function
# ==============================================================================

def instrument_with_local_guard(code_str: str, timeout_seconds: int = 2):
    """
    Instruments untrusted code with the comprehensive guard and wraps it in a
    signal-based timeout as a final fallback.
    """
    class TimeoutException(Exception): pass

    def timeout_handler(signum, frame):
        raise TimeoutException(f"Function execution exceeded time limit of {timeout_seconds}s.")

    guarded_tree = None
    try:
        # The signal module is not available on Windows.
        if sys.platform != "win32":
            signal.signal(signal.SIGALRM, timeout_handler)

        tree = ast.parse(textwrap.dedent(code_str))
        transformer = GuardTransformer()
        guarded_tree = transformer.visit(tree)
        ast.fix_missing_locations(guarded_tree)
        
        namespace = {'__name__': 'guarded_mod'}
        exec(compile(guarded_tree, "<ast_guarded>", "exec"), namespace)
        
        # Find the user-defined function (ignoring our helpers)
        funcs = [obj for obj in namespace.values() if isinstance(obj, FunctionType) and not obj.__name__.startswith('__guard_')]
        if len(funcs) != 1: raise ValueError(f"Expected 1 function in code, but found {len(funcs)}")
        
        sandboxed_func = funcs[0]

        @wraps(sandboxed_func)
        def timeout_wrapper(*args, **kwargs):
            if sys.platform != "win32":
                signal.alarm(timeout_seconds)
            try:
                result = sandboxed_func(*args, **kwargs)
            finally:
                if sys.platform != "win32":
                    signal.alarm(0) # Disable the alarm
            return result

        return timeout_wrapper

    except Exception as e:
        error_info = f"An error occurred during instrumentation: {e}\n{traceback.format_exc()}"
        if guarded_tree:
            try:
                # If available, print the transformed code for debugging
                error_info += f"\n\n--- Transformed Code ---\n{ast.unparse(guarded_tree)}\n---"
            except Exception: pass
        # Return the error string instead of a function if instrumentation fails
        return error_info



def _runner(fn: Callable, arg: Any, q: mp.Queue):
    try:
        q.put(("OK", fn(arg)))
    except Exception as e:
        q.put(("ERR", repr(e)))

def safe_call(fn, arg, timeout_s=2, mem_mb=None): # not used this is slow, we only use the ast guard in our experiment
    ctx = mp.get_context("fork")      # 或 "spawn"
    q   = ctx.Queue(1)
    p   = ctx.Process(target=_runner, args=(fn, arg, q))
    p.start()

    # 仅做 CPU 时间限额
    try:
        resource.prlimit(p.pid, resource.RLIMIT_CPU,
                         (timeout_s + 1, timeout_s + 1))
        if mem_mb is not None:        # ← 只有给了 mem_mb 才限制地址空间
            resource.prlimit(
                p.pid, resource.RLIMIT_AS,
                (mem_mb * 1024 ** 2,) * 2
            )
    except Exception:
        pass

    p.join(timeout_s)
    if p.is_alive():
        p.kill(); p.join()
        raise TimeoutError(f"user fn timed-out>{timeout_s}s on {arg!r}")

    if q.empty():
        raise RuntimeError("child exited without result")

    status, payload = q.get()
    if status == "ERR":
        raise RuntimeError(payload)
    return payload


def generate_all_mappings(sample_space:List[Dict], 
                          hypothesis:callable,
                          num_proc = 16,
                          do_safe_call = False,
                          ):
    dataset = Dataset.from_list(sample_space)
    def gen_mapping_function(input_example):
        cur_input_static = input_example['input_example']
        cur_input_pass_in = copy.deepcopy(input_example['input_example'])
        assert callable(hypothesis), f"Current function is not callable: {hypothesis}"
        
        try:
            if do_safe_call:
                mapped_output = safe_call(hypothesis, cur_input_pass_in)
                mapped_output = copy.deepcopy(mapped_output)
            else:
                mapped_output = hypothesis(cur_input_pass_in)
                mapped_output = copy.deepcopy(mapped_output)
        except Exception as e:
            # if str(e) == 'exceeded max recursion calls':
            #     print(e)
            mapped_output = 'N/A: cur input is not defined on this function'
            # print('-------')
            # print(cur_input, type(cur_input))
            # print('-------')
        try:
            json_output = json.dumps(mapped_output)
        except (ValueError, TypeError) as e:
            mapped_output = 'N/A: cur input is not defined on this function'
        return {    "sample_mapping_pair": (json.dumps(cur_input_static), json.dumps(mapped_output)),
                }
    mapped_dataset = dataset.map(gen_mapping_function, num_proc = num_proc, desc=None)
    return mapped_dataset


# --- init observation generation functions -----------------------------------
def load_io_pairs_list_function(pair_count,seed = 2025, final_question_count = 100) -> Dict[str, Any]:
    # Since elements are only digits we can randomly select io pairs from one problem
    # every problem have 16 io pairs
    # in total 250 problems in this dataset
    assert pair_count > 0 and pair_count <= 16, f'''List function dataset only have 16 io pairs per example'''
    rng_pairs = random.Random(seed)      
    rng_questions = random.Random(seed)
    file_name = 'list_function.jsonl'
    root_data_dir = '../data'
    dataset_path = os.path.join(root_data_dir, file_name)
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  
            dataset.append(obj)
    io_pairs = {}
    for single_problem in dataset:
        problem_id = single_problem['idx']
        all_io_pairs = []
        for train_test in ['train', 'test']:
            for single_case in single_problem[train_test]:
                cur_io_pair = (single_case['input'], single_case['output'])
                all_io_pairs.append(cur_io_pair)
        init_io_pairs = rng_pairs.sample(all_io_pairs, pair_count)
        io_pairs.update({problem_id:init_io_pairs})
    sampled_keys = rng_questions.sample(list(io_pairs.keys()), final_question_count)
    random.seed(seed)
    final_io_pairs = {i:io_pairs[i] for i in sampled_keys}
    return final_io_pairs


def load_io_pairs_mini_arc(pair_count, seed = 2025, final_question_count = 100):
    # Since all the inputs are in the same shape (5*5) and all elements are digits, random sample
    # every problem have 4~6 io pairs 
    # intotal 130 problems in this dataset
    rng_pairs = random.Random(seed)      
    rng_questions = random.Random(seed)
    random.seed(seed)
    data_file = '../data/miniarc.jsonl'
    loaded_data = []
    all_possible_inputs = set()
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  
            loaded_data.append(obj)
    io_pairs = {}
    assert pair_count <= 4, f'''Some problems in this dataset only have 4 io cases'''
    for problem in loaded_data:
        problem_id = problem['idx']
        cur_io_pairs = []
        for train_test in ['train', 'test']:
            for single_case in problem[train_test]:
                cur_io_pair = (single_case['input'], single_case['output'])
                cur_io_pairs.append(cur_io_pair)
        selected_io_pairs = rng_pairs.sample(cur_io_pairs, pair_count)
        io_pairs.update({problem_id:selected_io_pairs})
    print(len(io_pairs))    
    random.seed(seed)
    sampled_keys = rng_questions.sample(list(io_pairs.keys()), final_question_count)
    final_io_pairs = {i:io_pairs[i] for i in sampled_keys}
    return final_io_pairs


def load_io_pairs_arc_2025(pair_count, seed = 2025, final_question_count = 100):
    # {2: 237, 5: 69, 4: 249, 3: 769, 6: 23, 8: 3, 7: 9, 10: 1} # every problem io pairs count
    # if question start with 1,2,3,4, when pair count = 3 (237 problems will be dropped) and pair count = 4 (1006 problems will be dropped)
    #{'arc-agi_test_challenges.json': 259, 'arc-agi_evaluation_challenges.json': 172, 'arc-agi_training_challenges.json': 1076}
    # above are the files containing cases that only have input (test case)
    rng_pairs = random.Random(seed)      
    rng_questions = random.Random(seed)
    root_dir = '../data/arc_2025'
    file_with_io_pairs = ['arc-agi_test_challenges.json', 'arc-agi_evaluation_challenges.json', 'arc-agi_training_challenges.json']
    io_pairs = {}
    no_output_count = {}
    io_count = {}
    for data_file in file_with_io_pairs:
        cur_no_output_count = 0
        with open(os.path.join(root_dir, data_file), 'r') as f:
            loaded_data = json.load(f)
        for problem_id, problem in loaded_data.items():
            cur_io_pairs = []
            for train_test in ['train', 'test']:
                for single_case in problem[train_test]:
                    if 'output' not in single_case:
                        cur_no_output_count += 1
                        continue
                    cur_io_pair = (single_case['input'], single_case['output'])
                    cur_io_pairs.append(cur_io_pair)
            if len(cur_io_pairs) < pair_count:
                continue
            else:
                selected_io_pairs = rng_pairs.sample(cur_io_pairs, pair_count)
                if problem_id in io_pairs:
                    continue
                io_pairs.update({problem_id:selected_io_pairs})
                
            if len(cur_io_pairs) not in io_count:
                io_count.update({len(cur_io_pairs):1})
            else:
                io_count[len(cur_io_pairs)] += 1
        no_output_count.update({data_file:cur_no_output_count})
    print(len(io_pairs))
    random.seed(seed)
    sampled_keys = rng_questions.sample(list(io_pairs.keys()), final_question_count)
    final_io_pairs = {i:io_pairs[i] for i in sampled_keys}
    return final_io_pairs

def load_io_pairs_acre(pair_count, seed = 2025, final_question_count = 100):
    # There are two output state, when sample, need to make sure that at leaset two output state are observed
    # every problem have minmium 6 io pairs with clear label, and every problem have at least one case for on and one case for off state
    # intotal 100 problems in this dataset
    file_name = 'acre.jsonl'
    root_data_dir = '../data'
    dataset_path = os.path.join(root_data_dir, file_name)
    rng_pairs = random.Random(seed)      
    rng_questions = random.Random(seed)
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dataset.append(obj)

    assert pair_count >= 2, 'For acre dataset should be at least two paris with different label'
    assert pair_count <= 6, 'For acre dataset some problem only have at most 6 available io pairs'
    io_pairs = {}
    for problem in dataset:
        problem_id = problem['idx']
        on_io_pairs = []
        off_io_pairs = []
        for train_test in ['train', 'test']:
            for case in problem[train_test]:
                if case['output'] in ['on', 'off']: # only care about the case have clear label
                    if case['output'] == 'on':
                        cur_io_pair = (case['input'], case['output'])
                        on_io_pairs.append(cur_io_pair)
                    else:
                        cur_io_pair = (case['input'], case['output'])
                        off_io_pairs.append(cur_io_pair)
        assert len(on_io_pairs) != 0, f'''{problem_id} don't have on cases'''
        assert len(off_io_pairs) != 0, f'''{problem_id} don't have off cases'''
        base_on_case = rng_pairs.sample(on_io_pairs, 1)[0]
        on_io_pairs.remove(base_on_case)
        base_off_case = rng_pairs.sample(off_io_pairs, 1)[0]
        off_io_pairs.remove(base_off_case)
        init_io_pairs = [base_on_case, base_off_case]
        remaining_cases = []
        remaining_cases.extend(on_io_pairs)
        remaining_cases.extend(off_io_pairs)
        remaining_len = pair_count - 2
        selected_remaining_cases = rng_pairs.sample(remaining_cases, remaining_len)
        init_io_pairs.extend(selected_remaining_cases)
        io_pairs.update({problem_id:init_io_pairs}) 
    print(len(io_pairs))
    random.seed(seed)
    sampled_keys = rng_questions.sample(list(io_pairs.keys()), final_question_count)
    final_io_pairs = {i:io_pairs[i] for i in sampled_keys}
    return final_io_pairs

# --- Sample space generation functions -----------------------------------
def generate_sample_space_list_function(seed = 2025, sample_space_size_per_layer = 1000):
    all_samples = [[]]
    random.seed(seed)
    elements = list(range(0,100))
    for i in range(16):
        cur_len_max_sample = len(elements)**(i+1)
        if cur_len_max_sample <= sample_space_size_per_layer:
            cur_layer_samples = []
            product_results = itertools.product(elements, repeat=i+1)
            for product_result in product_results:
                cur_layer_samples.append(list(product_result))
                
        else:
            cur_layer_set = set()
            while len(cur_layer_set) < sample_space_size_per_layer:
                seq = tuple(random.sample(elements, k=i + 1))
                cur_layer_set.add(seq)
            cur_layer_samples = [list(seq) for seq in cur_layer_set]
        all_samples.extend(cur_layer_samples)
            
    return [{'input_example': single_input} for single_input in all_samples]

def generate_sample_space_acre(seed = 2025, sample_space_size_per_layer = 1000):
    file_name = 'acre.jsonl'
    root_data_dir = '../data'
    random.seed(seed)
    dataset_path = os.path.join(root_data_dir, file_name)
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dataset.append(obj)
    possible_len = 0
    all_input = []
    for data_item in dataset:
        for sample_case in data_item['train']:
            for one_input in sample_case['input']:
                all_input.append(one_input)
            if len(sample_case['input']) >= possible_len:
                possible_len = len(sample_case['input'])
    
    all_colors = []
    all_materials = []
    all_shapes = []
    for single_input in all_input:
        color, material, shape = single_input.split(' ')
        all_colors.append(color)
        all_materials.append(material)
        all_shapes.append(shape)
    unique_colors = set(all_colors)
    unique_materials = set(all_materials)
    unique_shapes = set(all_shapes)
    unique_colors    = sorted(unique_colors)
    unique_materials = sorted(unique_materials)
    unique_shapes    = sorted(unique_shapes)
    elements = [' '.join([color, material, shape]) for color, material, shape in itertools.product(unique_colors, unique_materials, unique_shapes)]
    all_samples = []
    all_samples.append([])
    for i in range(8):
        cur_len_max_sample = len(elements)**(i+1)
        if cur_len_max_sample <= sample_space_size_per_layer:
            cur_layer_samples = []
            product_results = itertools.product(elements, repeat=i+1)
            for product_result in product_results:
                cur_layer_samples.append(list(product_result))
                
        else:
            cur_layer_set = set()
            while len(cur_layer_set) < sample_space_size_per_layer:
                seq = tuple(random.sample(elements, k=i + 1))
                cur_layer_set.add(seq)
            cur_layer_samples = [list(seq) for seq in cur_layer_set]
        all_samples.extend(cur_layer_samples)
            
    return [{'input_example': single_input} for single_input in all_samples]

def generate_sample_space_arc_2025(seed = 2025): # no random.sample needed currently for sample from Arc dataset
    root_dir = '../data/arc_2025'
    file_with_io_pairs = ['arc-agi_test_challenges.json', 'arc-agi_evaluation_challenges.json', 'arc-agi_training_challenges.json']
    all_possible_inputs = set()
    for data_file in file_with_io_pairs:
        with open(os.path.join(root_dir, data_file), 'r') as f:
            loaded_data = json.load(f)
        for key, value in loaded_data.items():
            for train_test in ['train', 'test']:
                for single_case in value[train_test]:
                    all_possible_inputs.add(json.dumps(single_case['input']))
    return [{'input_example': json.loads(i)} for i in all_possible_inputs]

def generate_sample_space_mini_arc(seed = 2025): # no random.sample needed currently for sample from Arc dataset
    data_file = '../data/miniarc.jsonl'
    loaded_data = []
    all_possible_inputs = set()
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  
            loaded_data.append(obj)
    for value in loaded_data:
        for train_test in ['train', 'test']:
            for single_case in value[train_test]:
                all_possible_inputs.add(json.dumps(single_case['input']))
    return [{'input_example': json.loads(i)} for i in all_possible_inputs]
    
# --- helpers --------------------------------------------------------------
def is_cheating(
    src_or_fn: Union[str, Callable],
    io_keys: Union[Dict[Any, Any], Iterable[Any]]
) -> bool:
    # 1. get source code
    if callable(src_or_fn):
        try:
            src = inspect.getsource(src_or_fn)
        except (OSError, TypeError):
            raise ValueError("can't get src code")
    elif isinstance(src_or_fn, str):
        src = src_or_fn
    else:
        raise TypeError("first argument should be the code")
    tree = ast.parse(textwrap.dedent(src))

    # 2. normalize keys in the given input out pairs
    if isinstance(io_keys, dict):
        raw_keys = list(io_keys.keys())
    else:
        raw_keys = list(io_keys)
    def normalize_key(k: Any) -> Any:
        return list(k) if isinstance(k, tuple) else k
    normalized = [normalize_key(k) for k in raw_keys]
    keys_json = {json.dumps(k, sort_keys=True) for k in normalized}

    # 3. get parameters of the function
    func_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_defs:
        assigns = [n for n in tree.body 
                   if isinstance(n, ast.Assign) 
                   and isinstance(n.value, ast.Lambda)]
        if not assigns:
            raise ValueError("can't get function from source code")
        lam = assigns[0].value
        func = ast.FunctionDef(
            name="__lambda__",
            args=lam.args,
            body=[ast.Return(value=lam.body)],
            decorator_list=[]
        )
    else:
        func = func_defs[0]
    params = [arg.arg for arg in func.args.args]
    param = params[0] if params else None

    # 4. get all dict defined values that appear in the io_keys
    dict_vars: Dict[str, set] = {}
    class DictCollector(ast.NodeVisitor):
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Dict):
                matched = set()
                for key_node in node.value.keys:
                    v = None
                    if isinstance(key_node, ast.Constant):
                        if isinstance(key_node.value, str):
                            try:
                                obj = ast.literal_eval(key_node.value)
                            except Exception:
                                return
                        else:
                            obj = key_node.value
                        v = obj
                    elif isinstance(key_node, ast.Tuple):
                        if all(isinstance(el, ast.Constant) for el in key_node.elts):
                            v = tuple(el.value for el in key_node.elts)
                    elif isinstance(key_node, ast.List):
                        if all(isinstance(el, ast.Constant) for el in key_node.elts):
                            v = [el.value for el in key_node.elts]
                    if v is None:
                        return
                    norm = normalize_key(v)
                    try:
                        j = json.dumps(norm, sort_keys=True)
                    except Exception:
                        return
                    if j in keys_json:
                        matched.add(j)
                # If mapping dict have all the io_keys, this function have potential to store all the io_pairs in the function
                if matched >= keys_json:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            dict_vars[tgt.id] = matched
            self.generic_visit(node)
    DictCollector().visit(tree)

    # 5. check if subscription on the stored dict
    class SubscriptionChecker(ast.NodeVisitor):
        def __init__(self): self.found = False
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id in dict_vars:
                self.found = True
            self.generic_visit(node)
    sub_checker = SubscriptionChecker()
    sub_checker.visit(tree)
    if sub_checker.found:
        return True

    # 6. check if io_keys is subscriped with if-else branches
    def key_json_from_ast(node) -> Union[str, None]:
        v = None
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                try:
                    obj = ast.literal_eval(node.value)
                except Exception:
                    obj = node.value
            else:
                obj = node.value
            v = obj
        elif isinstance(node, ast.Tuple):
            if all(isinstance(el, ast.Constant) for el in node.elts):
                v = tuple(el.value for el in node.elts)
        elif isinstance(node, ast.List):
            if all(isinstance(el, ast.Constant) for el in node.elts):
                v = [el.value for el in node.elts]
        if v is None:
            return None
        norm = normalize_key(v)
        try:
            return json.dumps(norm, sort_keys=True)
        except Exception:
            return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                container = node.comparators[0]
                if isinstance(container, ast.Name) and container.id in dict_vars:
                    return True
                if isinstance(container, ast.Dict):
                    return True
            left = node.left
            is_param_expr = False
            if isinstance(left, ast.Name) and left.id == param:
                is_param_expr = True
            elif isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id in ("str", "repr"):
                args = getattr(left, 'args', [])
                if len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == param:
                    is_param_expr = True
            elif isinstance(left, ast.JoinedStr):
                for part in left.values:
                    if isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name) and part.value.id == param:
                        is_param_expr = True
                        break
            if is_param_expr:
                for comp in node.comparators:
                    kj = key_json_from_ast(comp)
                    if kj and kj in keys_json:
                        return True
    return False

def detect_while_true_risk(src: str) -> Tuple[bool, List[int]]:
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        # 语法错误时也可视为风险
        return True, []
    
    risky_lines: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            test = node.test
            # Python3.8+: ast.Constant
            if isinstance(test, ast.Constant):
                if test.value is True or (isinstance(test.value, (int, float)) and test.value != 0):
                    risky_lines.append(node.lineno)
            # Python<3.8: ast.NameConstant
            elif getattr(ast, 'NameConstant', None) and isinstance(test, ast.NameConstant):
                if test.value is True:
                    risky_lines.append(node.lineno)
            # 也可扩展：检测 `while not False`、`while 1==1` 等模式
    
    return (len(risky_lines) > 0)



# Evaluate tools 


def parse_result_dict(result_file_path):
    with open(result_file_path, 'r',) as f:
        generated_results = json.load(f)
    parsed_results = {}
    for generated_result in generated_results:
        functions = []
        discriptions = []
        for single_respond in generated_result['generated_functions']:
            cur_discription = single_respond[0]
            cur_string_code = single_respond[1]
            if cur_string_code == 'tuple parse error':
                # instruction following error
                cur_function = 'tuple parse error'
            else:
                cur_function = preprocess_code(cur_string_code)
                cur_function = parse_string_code_to_function(cur_string_code)
            functions.append(cur_function)
            discriptions.append(cur_discription)
        cur_parsed_result = {'discriptions':discriptions,   # Parsed_result
                             'functions':functions,
                             'train_io': generated_result['ori_data']['train'], 
                             'test_io': generated_result['ori_data']['test'],
                             'selected_test_pairs': generated_result['selected_test_pairs']}
        parsed_results.update({generated_result['question_id']:cur_parsed_result})
    return parsed_results


def evaluate_hypothesis(parsed_results, 
                        sample_space_gen_function,  
                        print_evaluate = False, 
                        seed = 2025, 
                        sample_space_size_per_layer = 50000,
                        num_proc = 16,
                       ):
    tuple_parse_error_count = 0
    question_count = len(parsed_results)
    for cur_question_index, item in tqdm(enumerate(parsed_results.items()), total=len(parsed_results)):
        problem_index = item[0]
        parsed_result = item[1]
        consistancy_check_io_pairs = parsed_result['selected_test_pairs']
        cur_function_scores = []
        unique_mapping_datasets = {} # for each function there should be a dict for all parsed ad
        for cur_function_index, function in enumerate(parsed_result['functions']): 
            #print(f'''Current question count: {cur_question_index+1}, there are in total: {question_count} questions, Current function count: {cur_function_index+1}, there are in total: {len(parsed_result['functions'])} functions''', end = '\r')
            cur_function_instruction_following_score = 0 # tuple parse error
            cur_function_syntac_parse_score = 0          # function string is able to parsed to real function
            cur_function_runable_score = 0               # function is able to run
            cur_function_consistancy_score = 0           # function did explain the result 
            cur_function_score_dict = {
                'init_observation_size':len(consistancy_check_io_pairs),
                'instruction_following_score':cur_function_instruction_following_score,
                'syntac_parse_score': cur_function_syntac_parse_score,
                'runable_score': cur_function_runable_score,
                'consistancy_score': cur_function_consistancy_score, # metric 1 score: count of init_observation passed
                'sample_space_size': 'N/A',
                'undefined_sample_size': 'N/A',
            }
    
            if function == 'tuple parse error': # generated result can not be parsed into tuples
                cur_function_scores.append(cur_function_score_dict)
                tuple_parse_error_count += 1
                continue
            else:
                cur_function_score_dict['instruction_following_score'] += 1
                if not isinstance(function, str):
                    #print(f'current function can be parsed: {function}')
                    cur_function_score_dict['syntac_parse_score'] += 1
                else:
                    #print(f'generated function can not be parsed!!! {function}')
                    cur_function_scores.append(cur_function_score_dict)
                    #print(cur_function_score_dict)
                    continue
            # now the function successfully parsed to callable
            for test_case in consistancy_check_io_pairs:  # first check if current hypothesis is consistant with given observations
                test_input = ast.literal_eval(test_case[0])
                #print(f'test_input: {test_input}, type: {type(test_input)}')
                test_output = test_case[1]
                try:
                    model_output = function(test_input)
                except Exception as e:
                    print(f'Current init_observation case is not runnable {e}')
                    pass
                else:
                    cur_function_score_dict['runable_score'] += 1
                    if test_output == str(model_output):
                        cur_function_score_dict['consistancy_score'] += 1
                    else:
                        if print_evaluate:
                            print('---------------------------')
                            print(f'Model generated code result: {test_output}, {type(test_output)}')
                            print(f'Gold given output result: {model_output}, {type(model_output)}')
                            print(test_output == model_output)
                            print('---------------------------')
            # Now sample the sample space
            sample_space = sample_space_gen_function(seed, sample_space_size_per_layer = sample_space_size_per_layer)
            sample_space_size = len(sample_space)
            mapped_results = generate_all_mappings(sample_space, function, num_proc = num_proc)
            unique_mapping_pairs = [tuple(pair) for pair in mapped_results['sample_mapping_pair'] if json.loads(pair[1]) != 'N/A: cur input is not defined on this function']
            undefined_count = len(mapped_results) - len(unique_mapping_pairs)
            unique_mapping_datasets.update({f'function_{cur_function_index}_mapping_result':set(unique_mapping_pairs)})


            cur_function_score_dict.update({'sample_space_size': sample_space_size, 'undefined_sample_size': undefined_count})
            
            cur_function_scores.append(cur_function_score_dict)
        # All unique mappings of current hypotheses proposed
        if len(unique_mapping_datasets) == 0:
            unique_mapping_count = 0
            avg_jaccard_idx = 'N/A no mapping result available'
        elif len(unique_mapping_datasets) == 1:
            unique_mapping_count = len(list(unique_mapping_datasets.values())[0])
            avg_jaccard_idx = 'N/A only one mapping result available'
        else:
            all_unique_mappings = set()
            for cur_unique_mappings in unique_mapping_datasets.values():
                all_unique_mappings = all_unique_mappings.union(cur_unique_mappings)
            unique_mapping_count = len(all_unique_mappings)
            
            # now calculate the average jaccard index for different hypothesis
            avg_jaccard_idx = avg_jaccard_parallel(unique_mapping_datasets.values(), processes = num_proc)

        parsed_result.update({'function_scores':cur_function_scores,
                              'num_unique_mappings':unique_mapping_count,
                              'avg_jaccard_idx':avg_jaccard_idx,})
        #print(unique_mapping_count)
     
    return parsed_results, tuple_parse_error_count


def evaluate_hypothesis_unique_mapping_incremental(parsed_results, 
                        sample_space_gen_function,  
                        print_evaluate = False, 
                        seed = 2025, 
                        sample_space_size_per_layer = 50000,
                        num_proc = 16,
                       ):
    tuple_parse_error_count = 0
    question_count = len(parsed_results)
    for cur_question_index, item in tqdm(enumerate(parsed_results.items()), total=len(parsed_results)):
        problem_index = item[0]
        parsed_result = item[1]
        consistancy_check_io_pairs = parsed_result['selected_test_pairs']
        cur_function_scores = []
        unique_mapping_datasets = {} # for each function there should be a dict for all parsed ad
        all_current_unique_mappings = set()
        for cur_function_index, function in enumerate(parsed_result['functions']): 
            #print(f'''Current question count: {cur_question_index+1}, there are in total: {question_count} questions, Current function count: {cur_function_index+1}, there are in total: {len(parsed_result['functions'])} functions''', end = '\r')
            cur_function_instruction_following_score = 0 # tuple parse error
            cur_function_syntac_parse_score = 0          # function string is able to parsed to real function
            cur_function_runable_score = 0               # function is able to run
            cur_function_consistancy_score = 0           # function did explain the result 
            cur_function_score_dict = {
                'init_observation_size':len(consistancy_check_io_pairs),
                'instruction_following_score':cur_function_instruction_following_score,
                'syntac_parse_score': cur_function_syntac_parse_score,
                'runable_score': cur_function_runable_score,
                'consistancy_score': cur_function_consistancy_score, # metric 1 score: count of init_observation passed
                'sample_space_size': 'N/A',
                'undefined_sample_size': 'N/A',
            }
    
            if function == 'tuple parse error': # generated result can not be parsed into tuples
                cur_function_scores.append(cur_function_score_dict)
                tuple_parse_error_count += 1
                continue
            else:
                cur_function_score_dict['instruction_following_score'] += 1
                if not isinstance(function, str):
                    #print(f'current function can be parsed: {function}')
                    cur_function_score_dict['syntac_parse_score'] += 1
                else:
                    #print(f'generated function can not be parsed!!! {function}')
                    cur_function_scores.append(cur_function_score_dict)
                    #print(cur_function_score_dict)
                    continue
            # now the function successfully parsed to callable
            for test_case in consistancy_check_io_pairs:  # first check if current hypothesis is consistant with given observations
                test_input = ast.literal_eval(test_case[0])
                #print(f'test_input: {test_input}, type: {type(test_input)}')
                test_output = test_case[1]
                try:
                    model_output = function(test_input)
                except Exception as e:
                    print(f'Current init_observation case is not runnable {e}')
                    pass
                else:
                    cur_function_score_dict['runable_score'] += 1
                    if test_output == str(model_output):
                        cur_function_score_dict['consistancy_score'] += 1
                    else:
                        if print_evaluate:
                            print('---------------------------')
                            print(f'Model generated code result: {test_output}, {type(test_output)}')
                            print(f'Gold given output result: {model_output}, {type(model_output)}')
                            print(test_output == model_output)
                            print('---------------------------')
            # Now sample the sample space
            sample_space = sample_space_gen_function(seed, sample_space_size_per_layer = sample_space_size_per_layer)
            sample_space_size = len(sample_space)
            mapped_results = generate_all_mappings(sample_space, function, num_proc = num_proc)
            unique_mapping_pairs = [tuple(pair) for pair in mapped_results['sample_mapping_pair'] if json.loads(pair[1]) != 'N/A: cur input is not defined on this function']
            undefined_count = len(mapped_results) - len(unique_mapping_pairs)
            unique_mapping_datasets.update({f'function_{cur_function_index}_mapping_result':set(unique_mapping_pairs)})
            new_current_unique_mappings = all_current_unique_mappings.union(set(unique_mapping_pairs))
            increment_unique_mapping_cur_step = len(new_current_unique_mappings) - len(all_current_unique_mappings)
            all_current_unique_mappings = new_current_unique_mappings

            cur_function_scores.append(cur_function_score_dict)
            cur_area_score = 'N/A'
            # All unique mappings of current hypotheses proposed
            if len(unique_mapping_datasets) == 0:
                unique_mapping_count = 0
                avg_jaccard_idx = 'N/A no mapping result available'
            elif len(unique_mapping_datasets) == 1:
                unique_mapping_count = len(list(unique_mapping_datasets.values())[0])
                avg_jaccard_idx = 'N/A only one mapping result available'
            else:
                all_unique_mappings = set()
                for cur_unique_mappings in unique_mapping_datasets.values():
                    all_unique_mappings = all_unique_mappings.union(cur_unique_mappings)
                unique_mapping_count = len(all_unique_mappings)
                
                # now calculate the average jaccard index for different hypothesis
                avg_jaccard_idx = avg_jaccard_parallel(unique_mapping_datasets.values(), processes = num_proc)

            cur_function_score_dict.update({'cur_func_idx': f'function_{cur_function_index}',
                                            'sample_space_size': sample_space_size, 
                                            'undefined_sample_size': undefined_count,
                                            'increment_unique_mapping_cur_step': increment_unique_mapping_cur_step,
                                            'avg_jaccard_idx_cur_step': avg_jaccard_idx})
        parsed_result.update({'function_scores':cur_function_scores})
        del parsed_result['functions']

        #print(unique_mapping_count)
     
    return parsed_results, tuple_parse_error_count


def analysis_function_results(parsed_results:Dict[Dict[str,Any], Any], 
                              tuple_parse_error_count = None # check if all functions have been scored
                             ):
    total_functions = 0                       # total number of generated hypothesis
    all_instruction_following_score = 0       # total number of generated responses can be parsed into tuples
    all_syntac_parse_score = 0                # total number of function strings that can be parsed into callable functions
    
    total_init_observation_test_cases = 0     # total number of init test cases
    all_runable_score = 0                     # total number of runnable train cases
    all_consistancy_score = 0                 # total number of train cases that correctly mapped

    total_sample_space_size = 0               # for function that can be parsed to callable functions, total number of sample spaces(This should be the num_function * single_sample_size)
    undefined_sample_count = 0                # for function that can be parsed to callable functions, how many samples that can not be mapped to a value with given hypothesis

    all_unique_mapping_count = 0              # in total how many unique mappings are there for a set of hypothses
    all_avg_jaccard_idx = 0
    available_jaccard_idx_count = 0

    question_count = len(parsed_results)
    
    
    for tem_result in parsed_results.values():
        for one_function in tem_result['function_scores']:
            all_instruction_following_score += one_function['instruction_following_score']
            all_syntac_parse_score += one_function['syntac_parse_score']
            all_runable_score += one_function['runable_score']
            all_consistancy_score += one_function['consistancy_score']
            total_init_observation_test_cases += one_function['init_observation_size']
            if one_function['sample_space_size'] == 'N/A' or one_function['undefined_sample_size'] == 'N/A':
                # this function can not be parsed, no scores
                pass
                
            else:
                total_sample_space_size += one_function['sample_space_size']
                undefined_sample_count += one_function['undefined_sample_size']
            total_functions += 1
        all_unique_mapping_count += tem_result['num_unique_mappings']
        if isinstance(tem_result['avg_jaccard_idx'], str):
            pass
        else:
            all_avg_jaccard_idx += tem_result['avg_jaccard_idx']
            available_jaccard_idx_count += 1
    sample_size = total_sample_space_size/all_syntac_parse_score
    avg_hypothesis_generated = round(total_functions/question_count,4)
            
    if tuple_parse_error_count is not None:
        assert total_functions == all_instruction_following_score + tuple_parse_error_count, f'numbers are not match please check!'

    # check some conditions
    if all_syntac_parse_score > 0:
        assert (all_unique_mapping_count <= sample_size * all_syntac_parse_score), f'''The number of unique mapping count is not correct! all_unique_mapping_count: {all_unique_mapping_count}, sample_size: {sample_size}'''
        
    analysis_result = {
        'all_instruction_following_score':round(all_instruction_following_score/total_functions,4), 
        'all_syntac_parse_score': round(all_syntac_parse_score/all_instruction_following_score,4),                      
        'all_runable_score':round(all_runable_score/total_init_observation_test_cases,4),         # count all the failed cases including not passing parse           
        'all_consistancy_score':round(all_consistancy_score/total_init_observation_test_cases,4), # count all the failed cases including not passing parse
        'hypothesis_domain_score':1-round(undefined_sample_count/total_sample_space_size,4),      # only count for the functions that can be correctly parsed
        'uniue_mapping_score':round(all_unique_mapping_count/total_sample_space_size,4),
        'all_avg_jaccard_idx':round(all_avg_jaccard_idx/available_jaccard_idx_count,4),
        'init_observation_size':total_init_observation_test_cases/total_functions,
        'avg_hypothesis_generated':avg_hypothesis_generated,
        'sample_size':sample_size,
    }

    return analysis_result