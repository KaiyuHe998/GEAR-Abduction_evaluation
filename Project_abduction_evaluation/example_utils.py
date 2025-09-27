import re
import json
import os
import utils
import copy
import numpy as np
import ast
from collections import OrderedDict
from global_variables import *
import generate_utils
import pandas as pd

with open('training_data/held_out_ids_t3.json', 'r') as f:
    t3_held_out = json.load(f)

held_out_problem_id = t3_held_out
datasets = {}
for dataset_name in held_out_problem_id.keys():
    if dataset_name == 'arc_2025':
        cur_dataset = utils.load_arc_2025_dataset()
    else:
        cur_dataset = utils.load_other_dataset(dataset_name)
    # filter problems to held-out IDs
    cur_dataset = {item[0]: item[1] for item in cur_dataset.items() if item[0] in held_out_problem_id[dataset_name]}
    datasets.update({
        dataset_name: cur_dataset
    })


def _load_eval_data_for_named_groups(named_groups: OrderedDict[str, list[tuple]]):
    """For diversity: return (multi_model_data, group_map(model->group_label), group_order(list of labels))."""
    multi = {}
    gmap = {}
    order = list(named_groups.keys())
    for label, tup_list in named_groups.items():
        for tup in tup_list:
            idx, ckp, short_ctx, rnd = tup
            mname, edata = read_data(idx, ckp=ckp, short_context_str=short_ctx, round_number=rnd)
            multi[mname] = edata
            gmap[mname] = label
    return multi, gmap, order


def _load_results_for_named_groups(named_groups: OrderedDict[str, list[tuple]]):
    """For pass rate: return (final_results, group_map(model->group_label), group_order(list of labels))."""
    results = {}
    gmap = {}
    order = list(named_groups.keys())
    for label, tup_list in named_groups.items():
        for (idx, ckp, short_ctx, rnd) in tup_list:
            one = get_model_results(idx, short_context=short_ctx or '', ckp_count=ckp, round_number=rnd)
            model_key = next(iter(one.keys()))
            results.update(one)
            gmap[model_key] = label
    return results, gmap, order


def _calc_generalizability(undef_list, space_list):
    """
    Compute per-hypothesis generalizability as:
        g = 1 - (undefined_count / sample_space_size)

    Returns the average generalizability for a problem under a given scope (e.g., "all" or "valid").
    """
    vals = []
    if not isinstance(undef_list, (list, tuple)) or not isinstance(space_list, (list, tuple)):
        return np.nan
    for u, s in zip(undef_list, space_list):  # truncate to shortest length if sizes differ
        try:
            u = float(u); s = float(s)
        except Exception:
            continue
        if s <= 0:
            continue
        g = 1.0 - (u / s)
        # clamp to [0, 1] and filter NaN/Inf
        if np.isfinite(g):
            vals.append(min(1.0, max(0.0, g)))
    return float(np.mean(vals)) if vals else np.nan


def read_data(model_name_index: int, ckp=None, short_context_str=None, round_number=None):
    eval_result_root_dir = 'eval_results_t3'
    model_name = model_mapping_dict[model_name_index]
    if ckp is not None:
        model_name += f'_{ckp}'
    if short_context_str is not None:
        model_name += short_context_str
    if round_number is not None:
        model_name += f'_round{round_number}'
    print(f'''Loading data from {model_name}''')

    if len(model_name.split('/')) > 1:
        model_name = model_name.split('/')[1]
    eval_file_path = os.path.join(eval_result_root_dir, f'final_eval_result_save_t3_{model_name}.json')
    with open(eval_file_path, 'r') as f:
        eval_result = json.load(f)
    return model_name, eval_result


def generate_passrate_table_v3(
    multi_model_results: dict,
    dataset_name: str = 'all',
    first_pass_na: float = np.nan,
    index_base: int = 1,
    round_ndigits: int = 4
) -> pd.DataFrame:
    """
    Aggregate performance across models on a specific dataset (or all datasets):
      - Avg Train Pass Rate
      - Avg Test Pass Rate
      - Top-k Hypothesis Correctness (k = 1‒3)
      - Instruction Following stats: #Hypotheses, #Follow, #Non-Follow, Follow Rate
      - Inconsistency stats (only within Follow): #Inconsistent, #Consistent, Inconsistent Rate

    multi_model_results structure (returned by get_model_results()):
      {
        model_name: {
          dataset_name: {
            problem_id: [
              {
                'train_pass_rate': float,
                'test_pass_rate' : float,
                'instr_following': bool,
                'instr_fail_reason': Optional[str],
                'inconsistent': Optional[bool],   # For Follow=True: True/False; For Non-Follow: usually None
                'train_fail_count': Optional[int]
              }, ...
            ]
          }, ...
        }, ...
      }
    """
    print(f"--- Generating Pass-Rate Table for dataset='{dataset_name}' ---")

    rows = []
    for model_name, model_data in multi_model_results.items():
        # select dataset(s)
        datasets_to_use = list(model_data.keys()) if dataset_name == 'all' else [dataset_name]

        train_rates_all, test_rates_all = [], []
        topk_hits = {k: [] for k in range(1, 4)}  # per-problem hit for Top-1..Top-3

        total_hypo_count = 0
        follow_cnt = 0
        inconsistent_cnt = 0

        for d in datasets_to_use:
            if d not in model_data:
                continue

            for problem_hypos in model_data[d].values():
                if not problem_hypos:
                    continue

                # pass rates aggregation
                for res in problem_hypos:
                    train_rates_all.append(res.get("train_pass_rate", 0.0))
                    test_rates_all.append(res.get("test_pass_rate", 0.0))

                # IF / inconsistent statistics
                total_hypo_count += len(problem_hypos)
                for res in problem_hypos:
                    if bool(res.get('instr_following', False)):
                        follow_cnt += 1
                        if bool(res.get('inconsistent', False)):
                            inconsistent_cnt += 1

                # per-problem Top-k: whether any of the first k hypotheses achieves perfect test score
                for k in range(1, 4):
                    slice_k = problem_hypos[:k]
                    hit = any(r.get("test_pass_rate", 0.0) == 1.0 for r in slice_k)
                    topk_hits[k].append(hit)

        # construct row
        if not train_rates_all:
            row_vals = {
                'Avg Train Pass Rate': first_pass_na,
                'Avg Test Pass Rate': first_pass_na,
                **{f'Top{k} Accuracy': first_pass_na for k in range(1, 4)},
                '#Hypotheses': 0,
                '#Follow': 0,
                '#Non-Follow': 0,
                'Follow Rate': first_pass_na,
                '#Inconsistent': 0,
                '#Consistent': 0,
                'Inconsistent Rate': first_pass_na
            }
        else:
            nonfollow_cnt = int(max(total_hypo_count - follow_cnt, 0))
            follow_rate = (follow_cnt / total_hypo_count) if total_hypo_count > 0 else first_pass_na

            consistent_cnt = int(max(follow_cnt - inconsistent_cnt, 0))
            inconsistent_rate = (inconsistent_cnt / follow_cnt) if follow_cnt > 0 else first_pass_na

            row_vals = {
                'Avg Train Pass Rate': float(np.mean(train_rates_all)),
                'Avg Test Pass Rate': float(np.mean(test_rates_all)),
                **{f'Top{k} Accuracy': (float(np.mean(topk_hits[k])) if topk_hits[k] else first_pass_na)
                   for k in range(1, 4)},
                '#Hypotheses': int(total_hypo_count),
                '#Follow': int(follow_cnt),
                '#Non-Follow': int(nonfollow_cnt),
                'Follow Rate': follow_rate,
                '#Inconsistent': int(inconsistent_cnt),
                '#Consistent': int(consistent_cnt),
                'Inconsistent Rate': inconsistent_rate
            }

        row_vals['Model'] = model_name
        rows.append(row_vals)

    if not rows:
        print("Error: No data found; empty table returned.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index('Model')

    # round ratio columns
    ratio_cols = [
        'Avg Train Pass Rate', 'Avg Test Pass Rate',
        'Top1 Accuracy', 'Top2 Accuracy', 'Top3 Accuracy',
        'Follow Rate', 'Inconsistent Rate'
    ]
    for c in ratio_cols:
        if c in df.columns:
            df[c] = df[c].round(round_ndigits)

    # cast count fields to integer dtype
    int_cols = ['#Hypotheses', '#Follow', '#Non-Follow', '#Inconsistent', '#Consistent']
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype('Int64')

    return df


def get_model_results(model_index, short_context: str = '', ckp_count: int = None, round_number: int = None):
    """
    Extends baseline pass-rate stats with per-hypothesis fields:
      - instr_following: whether a runnable function was successfully generated
      - instr_fail_reason: reason for failure (for debugging)
      - inconsistent: True if any train case failed (only meaningful when instr_following=True)
      - train_fail_count: number of failed cases on train set (only when instr_following=True)
    """
    root_dir = 'generate_log/t3'
    assert short_context in ['', '_short_context', '_super_short_context'], \
        f'undefined short context type: {short_context}'
    root_hypothesis_t5_path = root_dir + short_context

    model_name = model_mapping_dict[model_index]
    model_results = {dataset_name: {} for dataset_name in datasets}
    if ckp_count is not None:
        model_name = model_name + str(ckp_count)  # Bug here: need to align with previous log naming
    if round_number is not None:
        model_name = model_name + f'_round{round_number}'

    for dataset_name in datasets:
        for problem_id in datasets[dataset_name]:
            hypothesis_file_path = os.path.join(
                root_hypothesis_t5_path, model_name, dataset_name, f'{problem_id}.json'
            )
            with open(hypothesis_file_path, 'r') as f:
                t5_hypotheses_log = json.load(f)
            hypotheses = [i['content'] for i in t5_hypotheses_log if i['role'] == 'assistant']

            cur_hypo_results = []
            for hypothesis in hypotheses:
                instr_following = True
                instr_fail_reason = None
                callable_function = None

                # 1) parse (description, code_str)
                try:
                    description, code_str = generate_utils.parse_string_tuple(hypothesis)
                except Exception as e:
                    code_str = 'tuple parse error'
                    instr_following = False
                    instr_fail_reason = f'tuple_parse_exception: {e}'

                if code_str == 'tuple parse error':
                    instr_following = False
                    instr_fail_reason = instr_fail_reason or 'tuple_parse_error'
                elif not isinstance(code_str, str):
                    instr_following = False
                    instr_fail_reason = 'code_not_str'

                # 2) preprocess + instrument
                if instr_following:
                    try:
                        code_str = utils.preprocess_code(code_str)
                    except Exception as e:
                        # preprocessing failure does not immediately mark as non-follow; still try instrumentation
                        instr_fail_reason = f'preprocess_exception: {e}'  # record but do not veto

                    try:
                        callable_function = utils.instrument_with_local_guard(code_str)
                    except Exception as e:
                        callable_function = None
                        instr_following = False
                        instr_fail_reason = f'instrument_exception: {e}'

                    if not callable(callable_function):
                        if instr_following:
                            instr_following = False
                            instr_fail_reason = (
                                'instrument_not_callable'
                                if not isinstance(callable_function, str)
                                else f'instrument_error_msg: {callable_function[:200]}'
                            )

                # 3) stats for non-runnable hypothesis
                if not instr_following:
                    # Not runnable: per legacy behavior, set pass rates to 0; consistency is None (excluded from stats)
                    cur_result = {
                        'train_pass_rate': 0.0,
                        'test_pass_rate': 0.0,
                        'instr_following': False,
                        'instr_fail_reason': instr_fail_reason,
                        'inconsistent': None,
                        'train_fail_count': None
                    }
                    cur_hypo_results.append(cur_result)
                    continue

                # 4) evaluate runnable hypothesis
                failed_train_cases = evaluate_one_hypothesis(
                    callable_function, dataset_name, datasets[dataset_name][problem_id]['train']
                )
                failed_test_cases = evaluate_one_hypothesis(
                    callable_function, dataset_name, datasets[dataset_name][problem_id]['test']
                )
                train_set_fail_rate = round(
                    len(failed_train_cases) / len(datasets[dataset_name][problem_id]['train']), 4
                )
                test_set_fail_rate = round(
                    len(failed_test_cases) / len(datasets[dataset_name][problem_id]['test']), 4
                )

                # consistency: any train failure implies inconsistent
                train_fail_count = len(failed_train_cases)
                inconsistent = train_fail_count > 0

                cur_result = {
                    'train_pass_rate': 1 - train_set_fail_rate,
                    'test_pass_rate': 1 - test_set_fail_rate,
                    'instr_following': True,
                    'instr_fail_reason': None,
                    'inconsistent': inconsistent,
                    'train_fail_count': train_fail_count
                }
                cur_hypo_results.append(cur_result)

            model_results[dataset_name].update({problem_id: cur_hypo_results})

    final_model_name = f'{model_name}{short_context}_round{round_number}'
    return {final_model_name: model_results}


def evaluate_one_hypothesis(callable_function, dataset_name, io_pairs):
    failed_cases = {}
    for init_pair_index, init_pair in enumerate(io_pairs):
        if isinstance(init_pair[0], str):
            assert dataset_name == 'list_function', f'''only list_function need to parse string input into actual list'''
            test_input = ast.literal_eval(init_pair[0])
        else:
            test_input = init_pair[0]
        if isinstance(init_pair[1], str):
            if dataset_name == 'acre':
                test_output = init_pair[1]
            else:
                assert dataset_name == 'list_function', f'''only list_function need to parse string input into actual list'''
                test_output = ast.literal_eval(init_pair[1])
        else:
            test_output = init_pair[1]
        try:
            cur_input_pass_in = copy.deepcopy(test_input)
            model_output = callable_function(cur_input_pass_in)
            model_output = copy.deepcopy(model_output)
        except Exception as e:
            failed_cases.update({json.dumps(init_pair): [json.dumps(test_output), 'N/A', str(e)]})  # io_pair: [gold_output, generated_output, err_msg]
        else:
            if test_output == model_output:
                pass
            else:
                try:
                    failed_cases.update({json.dumps(init_pair): [json.dumps(test_output), json.dumps(model_output), 'output not match']})
                except Exception as e:
                    failed_cases.update({json.dumps(init_pair): [test_output, 'Error: json can not dump the failed output result', 'output not match']})
    return failed_cases


def build_results_and_groups(config_list):
    """
    Supported elements in config_list:
      - int: index only (equivalent to (idx, None, '', None))
      - tuple: (model_idx, ckp, short_context_str, round_number)
      - list: defines a group; inside are multiple int/tuple items

    Returns:
      final_results: {model_name: model_results_dict}
      group_map:     {model_name: 'G1'/'G2'/...}  # assign group labels only for models inside groups
      group_order:   ['G1','G2', ...]             # preserve input group order for later appending
    """
    final_results = {}
    group_map = {}
    group_order = []
    group_counter = 0

    def _normalize_item(item):
        if isinstance(item, tuple):
            idx, ckp, short_ctx, rnd = item
        elif isinstance(item, int):
            idx, ckp, short_ctx, rnd = item, None, '', None
        else:
            raise ValueError(f"Unsupported config item: {item}")
        return idx, ckp, short_ctx, rnd

    def _load_one(item):
        idx, ckp, short_ctx, rnd = _normalize_item(item)
        one = get_model_results(idx, short_context=short_ctx, ckp_count=ckp, round_number=rnd)
        model_key = next(iter(one.keys()))
        final_results.update(one)
        return model_key

    for entry in config_list:
        if isinstance(entry, list):              # this is a group
            group_counter += 1
            gname = f"G{group_counter}"
            group_order.append(gname)
            for item in entry:
                mk = _load_one(item)
                group_map[mk] = gname
        else:                                    # single model
            _load_one(entry)

    return final_results, group_map, group_order


def append_group_summaries_passrate(df_in: pd.DataFrame,
                                    group_map: dict,
                                    group_order=None,
                                    summary_suffix="(Agg)",
                                    round_ndigits: int = 4) -> pd.DataFrame:
    """
    On top of the DataFrame returned by generate_passrate_table_v3, append a summary row per group.
    - For ratio/score columns: take the mean ('Avg ... Rate', 'Topk Accuracy', 'Follow Rate', 'Inconsistent Rate').
    - For count columns: any column name starting with '#' is summed (e.g., '#Hypotheses', '#Follow', '#Non-Follow', '#Inconsistent', '#Consistent').
    """
    if not group_map:
        return df_in

    df = df_in.copy()
    tmp = df.reset_index()  # restore 'Model' column
    tmp['Group'] = tmp['Model'].map(group_map)
    tmp = tmp.dropna(subset=['Group'])

    # identify columns: names starting with '#' are counts; others numeric columns are treated as ratios/scores
    numeric_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
    count_cols = [c for c in tmp.columns if c.startswith('#')]
    ratio_cols = [c for c in numeric_cols if c not in count_cols]

    grouped_mean = tmp.groupby('Group')[ratio_cols].mean(numeric_only=True)
    grouped_sum = tmp.groupby('Group')[count_cols].sum(numeric_only=True) if count_cols else pd.DataFrame()

    grouped = grouped_mean.join(grouped_sum, how='outer')
    # rounding
    for c in ratio_cols:
        if c in grouped.columns:
            grouped[c] = grouped[c].round(round_ndigits)

    # append in input order (or alphabetical if not provided)
    order = group_order or list(grouped.index)
    for g in order:
        if g in grouped.index:
            df.loc[f'[{g}] {summary_suffix}'] = grouped.loc[g]

    # keep count columns as integer-like display (nullable Int64)
    for c in count_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].round().astype('Int64')

    return df


def generate_performance_table_t3_v2(
    multi_model_data: dict,
    dataset_name: str,
    held_out_map: dict = None,
    subset: str = None,
    group_map: dict = None,
) -> pd.DataFrame:
    """
    Analyze multi-model performance on a given dataset (or all datasets) and produce a summary table.

    Args:
        multi_model_data (dict): {model_name: evaluation_data}
        dataset_name (str): dataset name, or 'all'
        held_out_map (dict, optional): mapping of held-out IDs
        subset (str, optional): 'train' | 'held_out' | None
    """
    if subset and not held_out_map:
        raise ValueError("If 'subset' is specified, 'held_out_map' must be provided.")

    title_dataset = "All Datasets (Averaged)" if dataset_name == 'all' else f"Dataset: '{dataset_name}'"
    subset_title = " (All Problems)" if subset is None else (" (Train Subset)" if subset == 'train' else " (Held-out Subset)")
    print(f"--- Starting Performance Analysis for {title_dataset}{subset_title} ---")

    all_records = []
    model_names = list(multi_model_data.keys())

    for model_name, eval_data in multi_model_data.items():
        datasets_to_process = list(eval_data.keys()) if dataset_name == 'all' else [dataset_name]

        for d_name in datasets_to_process:
            if d_name not in eval_data:
                if dataset_name != 'all':
                    print(f"Warning: Dataset '{d_name}' not found for model '{model_name}'. Skipping.")
                continue

            cur_held_out_ids = held_out_map.get(d_name, []) if held_out_map else []
            problem_results = eval_data[d_name]

            # iterate over problems
            for problem in problem_results:
                problem_id = problem['cur_problem_id']

                if subset:
                    is_held_out = problem_id in cur_held_out_ids
                    if (subset == 'train' and is_held_out) or (subset == 'held_out' and not is_held_out):
                        continue

                total_generated = int(problem.get('total_num_hypothesis_generated', 0) or 0)
                if total_generated <= 0:
                    continue

                num_valid_hypotheses = int(problem.get('total_num_valid_hypothesis_generated', 0) or 0)
                syntax_error_count = int(problem.get('all_syntac_parse_error', 0) or 0)
                syntax_error_count = max(0, min(syntax_error_count, total_generated))
                syntactically_correct_count = total_generated - syntax_error_count

                record = {
                    'model_name': model_name,
                    'instruction_following': 1.0 - (syntax_error_count / total_generated),
                    'consistency': (num_valid_hypotheses / syntactically_correct_count) if syntactically_correct_count > 0 else 0.0
                }
                gen_all = _calc_generalizability(
                    problem.get('undefined_count', []),
                    problem.get('sample_space_size', [])
                )
                # Only compute 'valid' scope generalizability when valid hypotheses exist; otherwise NaN.
                gen_valid = _calc_generalizability(
                    problem.get('undefined_count_valid', []),
                    problem.get('sample_space_size_valid', [])
                ) if num_valid_hypotheses > 0 else np.nan

                record['generalizability_all'] = gen_all
                record['generalizability_valid'] = gen_valid
                record['num_hypotheses_all'] = int(total_generated)
                record['num_hypotheses_valid'] = int(num_valid_hypotheses)

                # Compute average normalized input space size for both scopes (based on sample_space_size)
                raw_sample_space_list = problem.get('sample_space_size', [])
                valid_sample_spaces = [s for s in raw_sample_space_list if isinstance(s, (int, float))]
                avg_input_norm_space = float(np.mean(valid_sample_spaces)) if valid_sample_spaces else None

                # ---- Diversity based on VALID hypotheses (keep original columns) ----
                # Existing columns: normalized_gamma_diversity (valid) & beta_struct (valid)
                suffix = '_valid'
                gamma_q0_struct_valid = problem.get(f'gamma_diversity_q_0{suffix}_struct')
                beta_struct_valid = problem.get(f'avg_beta_diversity_struct{suffix}')
                if (gamma_q0_struct_valid is not None) and (beta_struct_valid is not None) and (avg_input_norm_space and avg_input_norm_space > 0):
                    record['normalized_gamma_diversity'] = gamma_q0_struct_valid / avg_input_norm_space
                    record['beta_struct'] = beta_struct_valid
                else:
                    # ensure columns exist (so aggregation can ignore NaN)
                    record['normalized_gamma_diversity'] = np.nan
                    record['beta_struct'] = np.nan

                # ---- Diversity based on ALL generated hypotheses (new) ----
                # New columns: normalized_gamma_diversity_all & beta_struct_all
                gamma_q0_struct_all = problem.get('gamma_diversity_q_0_struct')  # field for "all" scope
                beta_struct_all = problem.get('avg_beta_diversity_struct')       # field for "all" scope

                if (gamma_q0_struct_all is not None) and (avg_input_norm_space and avg_input_norm_space > 0):
                    record['normalized_gamma_diversity_all'] = gamma_q0_struct_all / avg_input_norm_space
                else:
                    record['normalized_gamma_diversity_all'] = np.nan

                record['beta_struct_all'] = beta_struct_all if (beta_struct_all is not None) else np.nan

                all_records.append(record)

    if not all_records:
        print(f"Error: No valid records found for the specified criteria. Cannot generate table.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # 2) aggregate with mean (ignore NaN)
    agg_funcs = {
        'instruction_following': 'mean',
        'consistency': 'mean',
        'normalized_gamma_diversity': 'mean',
        'beta_struct': 'mean',
        'generalizability_valid': 'mean',
        'num_hypotheses_valid': 'sum',
    }
    summary_df = df.groupby('model_name').agg(agg_funcs).reset_index()

    # 3) formatting & renaming
    summary_df.rename(columns={
        'model_name': 'Model',
        'instruction_following': 'Instruction Following',
        'consistency': 'Consistency',
        'beta_struct': 'Beta Struct Diversity (Valid)',
        'normalized_gamma_diversity': 'Normalized Gamma Diversity (Valid, q=0)',
        'generalizability_valid': 'Generalizability (Valid)',
        'num_hypotheses_valid': 'Hypotheses Generated (Valid)',
    }, inplace=True)

    summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=model_names, ordered=True)
    summary_df.sort_values('Model', inplace=True)
    summary_df.set_index('Model', inplace=True)

    print("--- Analysis Complete. Performance Table: ---")
    if group_map is not None:
        # restore index to map groups
        tmp = summary_df.reset_index()  # now has 'Model'
        # map model to group; drop unmatched or map to 'Ungrouped' if desired
        tmp['Group'] = tmp['Model'].map(group_map)

        # drop ungrouped rows (use .fillna('Ungrouped') above if you want to keep them)
        tmp = tmp.dropna(subset=['Group'])

        # average numeric columns only
        numeric_cols = tmp.select_dtypes(include=[np.number]).columns
        grouped_df = tmp.groupby('Group', sort=False)[numeric_cols].mean().round(4)

        # return "one row per group"
        return grouped_df

    return summary_df.round(4)


from string import Template

def make_split_agg_latex(
    final_agg_only: pd.DataFrame,
    grouped_cfgs,                          # OrderedDict: the 9 settings in desired order
    caption: str = 'Aggregated results across nine settings (group averages only).',
    label: str = 'tab:agg_9rows_split',
    as_percent: bool = False,              # True -> display as percentage with %
    decimals: int = 3,                     # decimals when not using percentage display
    cross_column: bool = False,            # True -> table* (two-column); False -> table (single column)
    strip_brackets_in_index: bool = False, # True -> turn "[name] (Agg)" into "name"
    div_block_title: str = 'Diversity block',
    acc_block_title: str = 'T3 accuracy block'
) -> str:
    """Split a 9-row aggregate table into two stacked tabular blocks within a single LaTeX table environment."""
    # -------- required columns --------
    div_cols_map = {
        'Instruction Following': 'Instruction Following Rate',
        'Consistency': 'Consistency',
        'Generalizability (Valid)': 'Generalizability',
        'Beta Struct Diversity (Valid)': 'Beta Diversity',
        'Normalized Gamma Diversity (Valid, q=0)': 'Gamma Diversity',
    }
    acc_cols_map = {
        'Avg Train Pass Rate': 'Avg Train Pass Rate',
        'Avg Test Pass Rate': 'Avg Test Pass Rate',
        'Top1 Accuracy': 'Top-1 Accuracy',
        'Top2 Accuracy': 'Top-2 Accuracy',
        'Top3 Accuracy': 'Top-3 Accuracy',
    }

    # add missing columns to avoid KeyError
    df = final_agg_only.copy()
    for col in list(div_cols_map.keys()) + list(acc_cols_map.keys()):
        if col not in df.columns:
            df[col] = np.nan

    # fixed 9-row order
    row_order = [f'[{label}] (Agg)' for label in grouped_cfgs.keys()]
    df_div = df.loc[row_order, list(div_cols_map.keys())].rename(columns=div_cols_map)
    df_acc = df.loc[row_order, list(acc_cols_map.keys())].rename(columns=acc_cols_map)

    # index name and optional shortening
    def _strip_idx(s: str) -> str:
        s = re.sub(r'^\[', '', s)
        s = re.sub(r'\]\s*\(Agg\)$', '', s)
        return s
    if strip_brackets_in_index:
        df_div.index = [_strip_idx(i) for i in df_div.index]
        df_acc.index = [_strip_idx(i) for i in df_acc.index]

    df_div.index.name = 'Setting (Aggregate)'
    df_acc.index.name = 'Setting (Aggregate)'

    # escape LaTeX special characters in row/column headers (values do not need escaping)
    def latex_escape_text(s: str) -> str:
        trans = {
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
            '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
        }
        return ''.join(trans.get(ch, ch) for ch in s)

    df_div.columns = [latex_escape_text(str(c)) for c in df_div.columns]
    df_acc.columns = [latex_escape_text(str(c)) for c in df_acc.columns]
    df_div.index   = [latex_escape_text(str(i)) for i in df_div.index]
    df_acc.index   = [latex_escape_text(str(i)) for i in df_acc.index]

    # value formatting
    def _fmt_val(v):
        if pd.isna(v): return '--'
        return f'{float(v)*100:.2f}\\%' if as_percent else f'{float(v):.{decimals}f}'

    fmt_div = {c: _fmt_val for c in df_div.columns}
    fmt_acc = {c: _fmt_val for c in df_acc.columns}

    colfmt_div = 'l' + 'c' * len(df_div.columns)
    colfmt_acc = 'l' + 'c' * len(df_acc.columns)

    latex_div = df_div.to_latex(
        escape=False, index=True, bold_rows=False,
        column_format=colfmt_div, na_rep='--', formatters=fmt_div
    ).strip()

    latex_acc = df_acc.to_latex(
        escape=False, index=True, bold_rows=False,
        column_format=colfmt_acc, na_rep='--', formatters=fmt_acc
    ).strip()

    # combine into a single table environment with two stacked tabular blocks
    env = 'table*' if cross_column else 'table'
    wrapper = Template(r"""
\begin{$env}[t]
    \centering
    \setlength{\tabcolsep}{3pt}
    \renewcommand{\arraystretch}{0.95}
    \caption{$caption}
    \label{$label}
    {\itshape $div_title}\par
    \vspace{0.25em}
    \resizebox{\linewidth}{!}{%
$div_body
    }
    \vspace{0.6em}
    {\itshape $acc_title}\par
    \vspace{0.25em}
    \resizebox{\linewidth}{!}{%
$acc_body
    }
\end{$env}
""".strip())

    latex_table = wrapper.substitute(
        env=env,
        caption=caption,
        label=label,
        div_title=div_block_title,
        acc_title=acc_block_title,
        div_body=latex_div,
        acc_body=latex_acc
    )
    return latex_table
