# evaluate_t3.py
# This file evaluates t3 hypothesis using precomputed mappings.
# It mirrors the IO evaluation pipeline's statistics and output format.

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List
from tqdm.auto import tqdm

import utils
import generate_utils
from global_variables import *
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

# ----------------- Config -----------------
seed = 2025
unique_mapping_increase_threshold = 0.2  # used only for building dynamic curves; no early stop here
gen_save_root_dir_base = 'your path'
mapping_root_base = 'specify your mapping save path keep the same in final_generate_hypothesis_auto_save_mappings_t3.py'
eval_root_save_dir = 'your path'
num_proc_default = 16  # not used for computation here; kept for signature consistency

VALID_SHORT = {'', '_short_context', '_super_short_context'}

# ----------------- Helpers -----------------
def qualify_model_tag(base_name: str, ckp_count: str | None, round_number: str | None) -> str:
    tag = base_name
    if ckp_count is not None:
        tag += f"_ckp{ckp_count}"
    if round_number is not None:
        tag += f"_round{round_number}"
    return tag

def roots_for_short(short_context: str):
    assert short_context in VALID_SHORT, f"Invalid short_context: {short_context}"
    # hypotheses root
    gen_root = os.path.join(gen_save_root_dir_base, f"t3{short_context}")
    # mappings root
    map_root = os.path.join(mapping_root_base, f"t3{short_context}")
    return gen_root, map_root

def is_valid(eval_result: Dict[str, Any]) -> bool:
    # Compatible with IO-style per-hypothesis evaluation schema
    # (your evaluate_one_generation for t3 writes the same keys)
    if isinstance(eval_result, str):
        return False
    return eval_result.get('is_consistant', False) and not eval_result.get('is_cheat', False)

# ----------------- Core -----------------
def evaluate_one_datafolder_t3(
    model_tag: str,
    dataset_name: str,
    gen_root: str,
    map_root: str,
    held_out_ids: List[str],
    print_process: bool = False,
    num_proc: int = num_proc_default,
) -> List[Dict[str, Any]]:
    """
    Read hypotheses and precomputed mapping results for a (model_tag, dataset).
    Returns a list of per-problem evaluation dicts (same shape as IO version).
    """
    eval_results: List[Dict[str, Any]] = []

    # Hypothesis directory for this dataset
    hypo_dir = os.path.join(gen_root, model_tag, dataset_name)
    if not os.path.exists(hypo_dir):
        # No hypotheses for this dataset: return empty
        return eval_results

    # We iterate over files present in the folder, then (optionally) filter by held-out list
    file_names = [f for f in os.listdir(hypo_dir) if f.endswith('.json') and not f.startswith('.')]
    # Keep only held-out problems if provided
    held_out_set = set(held_out_ids) if held_out_ids else None

    for file_name in tqdm(file_names, desc=f"{dataset_name}", leave=False):
        history_log_path = os.path.join(hypo_dir, file_name)
        with open(history_log_path, 'r') as f:
            history_messages = json.load(f)

        cur_problem_id = file_name.split('.')[0]
        if held_out_set is not None and cur_problem_id not in held_out_set:
            continue

        generated_hypothesis = [m for m in history_messages if m.get('role') == 'assistant']
        assert len(generated_hypothesis) > 0, \
            f"no generated hypothesis: model={model_tag}, dataset={dataset_name}, problem={cur_problem_id}"

        # Mapping directory
        mapping_dir = os.path.join(map_root, model_tag, dataset_name, cur_problem_id)
        os.makedirs(mapping_dir, exist_ok=True)

        evaluation_history: List[Dict[str, Any] | str] = []
        total_num_hypothesis_generated = 0
        dynamic_evaluation: Dict[str, Any] | None = None

        for hypothesis_idx, one_hypothesis in enumerate(generated_hypothesis):
            mapping_file = os.path.join(mapping_dir, f"{hypothesis_idx}.json")

            # Progress line (no nested f-strings issues)
            progress_key = os.path.join(model_tag, dataset_name, cur_problem_id, str(hypothesis_idx))
            print(f"{progress_key}   Cur time: {datetime.now()}", end='\r')

            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    evaluate_result = json.load(f)
                # normalize tuple lists if needed
                if not isinstance(evaluate_result, str) and 'unique_mapping_list' in evaluate_result:
                    evaluate_result['unique_mapping_list'] = [tuple(pair) for pair in evaluate_result['unique_mapping_list']]
            else:
                # Keep same behavior as IO version: require mapping files to exist already
                raise AssertionError(
                    f"Missing mapping file: {mapping_file}. "
                    "Please run the t3 mapping generation first."
                )

            evaluation_history.append(evaluate_result)
            # Build dynamic curves (no early-stop effect here; just reuse the util)
            cont, dyn = generate_utils.if_continue_generation(
                evaluation_history,
                unique_mapping_increase_threshold=unique_mapping_increase_threshold,
                print_process=print_process
            )
            dynamic_evaluation = dyn
            total_num_hypothesis_generated += 1

        assert dynamic_evaluation is not None, \
            f"dynamic_evaluation is None for dataset={dataset_name}, problem={cur_problem_id}"

        # Aggregate stats across all hypotheses
        all_mapping_lists: List[List[tuple]] = []
        mapping_lists_valid: List[List[tuple]] = []
        valid_hypothesis_index: List[int] = []
        total_num_valid_hypothesis_generated = 0

        for h_idx, er in enumerate(evaluation_history):
            if isinstance(er, str):
                all_mapping_lists.append([])
                continue
            all_mapping_lists.append(er.get('unique_mapping_list', []))
            if is_valid(er):
                total_num_valid_hypothesis_generated += 1
                mapping_lists_valid.append(er.get('unique_mapping_list', []))
                valid_hypothesis_index.append(h_idx)

        # Alpha/Beta/Gamma diversities (all)
        avg_alpha_diversity_q_0 = None
        avg_alpha_diversity_q_1 = None
        avg_alpha_diversity_q_2 = None
        avg_beta_diversity_dist = None
        avg_beta_diversity_struct = None
        gamma_diversity_q_0_dist = None
        gamma_diversity_q_1_dist = None
        gamma_diversity_q_2_dist = None
        gamma_diversity_q_0_struct = None
        gamma_diversity_q_1_struct = None
        gamma_diversity_q_2_struct = None

        if len(all_mapping_lists) == 0:
            raise AssertionError(
                f"Should have hypotheses for problem={cur_problem_id}, dataset={dataset_name}, model={model_tag}"
            )
        elif len(all_mapping_lists) == 1:
            avg_alpha = utils.calculate_avg_alpha_diversity(all_mapping_lists)
            avg_alpha_diversity_q_0 = avg_alpha['q0_richness']
            avg_alpha_diversity_q_1 = avg_alpha['q1_shannon']
            avg_alpha_diversity_q_2 = avg_alpha['q2_simpson']

            gamma_dist = utils.calculate_gamma_diversity_dist(all_mapping_lists)
            gamma_diversity_q_0_dist = gamma_dist['q0_richness']
            gamma_diversity_q_1_dist = gamma_dist['q1_shannon']
            gamma_diversity_q_2_dist = gamma_dist['q2_simpson']

            gamma_struct = utils.calculate_gamma_diversity_struct(all_mapping_lists)
            gamma_diversity_q_0_struct = gamma_struct['q0_richness']
            gamma_diversity_q_1_struct = gamma_struct['q1_shannon']
            gamma_diversity_q_2_struct = gamma_struct['q2_simpson']
        else:
            avg_alpha = utils.calculate_avg_alpha_diversity(all_mapping_lists)
            avg_alpha_diversity_q_0 = avg_alpha['q0_richness']
            avg_alpha_diversity_q_1 = avg_alpha['q1_shannon']
            avg_alpha_diversity_q_2 = avg_alpha['q2_simpson']

            avg_beta_diversity_dist = utils.calculate_avg_beta_dist(all_mapping_lists)
            avg_beta_diversity_struct = utils.calculate_avg_beta_struct(all_mapping_lists)

            gamma_dist = utils.calculate_gamma_diversity_dist(all_mapping_lists)
            gamma_diversity_q_0_dist = gamma_dist['q0_richness']
            gamma_diversity_q_1_dist = gamma_dist['q1_shannon']
            gamma_diversity_q_2_dist = gamma_dist['q2_simpson']

            gamma_struct = utils.calculate_gamma_diversity_struct(all_mapping_lists)
            gamma_diversity_q_0_struct = gamma_struct['q0_richness']
            gamma_diversity_q_1_struct = gamma_struct['q1_shannon']
            gamma_diversity_q_2_struct = gamma_struct['q2_simpson']

        # Valid-only stats
        avg_alpha_diversity_q_0_valid = None
        avg_alpha_diversity_q_1_valid = None
        avg_alpha_diversity_q_2_valid = None
        avg_beta_diversity_dist_valid = None
        avg_beta_diversity_struct_valid = None
        gamma_diversity_q_0_valid_dist = None
        gamma_diversity_q_1_valid_dist = None
        gamma_diversity_q_2_valid_dist = None
        gamma_diversity_q_0_valid_struct = None
        gamma_diversity_q_1_valid_struct = None
        gamma_diversity_q_2_valid_struct = None

        if len(mapping_lists_valid) == 0:
            pass
        elif len(mapping_lists_valid) == 1:
            avg_alpha_valid = utils.calculate_avg_alpha_diversity(mapping_lists_valid)
            avg_alpha_diversity_q_0_valid = avg_alpha_valid['q0_richness']
            avg_alpha_diversity_q_1_valid = avg_alpha_valid['q1_shannon']
            avg_alpha_diversity_q_2_valid = avg_alpha_valid['q2_simpson']

            gamma_dist_valid = utils.calculate_gamma_diversity_dist(mapping_lists_valid)
            gamma_diversity_q_0_valid_dist = gamma_dist_valid['q0_richness']
            gamma_diversity_q_1_valid_dist = gamma_dist_valid['q1_shannon']
            gamma_diversity_q_2_valid_dist = gamma_dist_valid['q2_simpson']

            gamma_struct_valid = utils.calculate_gamma_diversity_struct(mapping_lists_valid)
            gamma_diversity_q_0_valid_struct = gamma_struct_valid['q0_richness']
            gamma_diversity_q_1_valid_struct = gamma_struct_valid['q1_shannon']
            gamma_diversity_q_2_valid_struct = gamma_struct_valid['q2_simpson']
        else:
            avg_alpha_valid = utils.calculate_avg_alpha_diversity(mapping_lists_valid)
            avg_alpha_diversity_q_0_valid = avg_alpha_valid['q0_richness']
            avg_alpha_diversity_q_1_valid = avg_alpha_valid['q1_shannon']
            avg_alpha_diversity_q_2_valid = avg_alpha_valid['q2_simpson']

            avg_beta_diversity_dist_valid = utils.calculate_avg_beta_dist(mapping_lists_valid)
            avg_beta_diversity_struct_valid = utils.calculate_avg_beta_struct(mapping_lists_valid)

            gamma_dist_valid = utils.calculate_gamma_diversity_dist(mapping_lists_valid)
            gamma_diversity_q_0_valid_dist = gamma_dist_valid['q0_richness']
            gamma_diversity_q_1_valid_dist = gamma_dist_valid['q1_shannon']
            gamma_diversity_q_2_valid_dist = gamma_dist_valid['q2_simpson']

            gamma_struct_valid = utils.calculate_gamma_diversity_struct(mapping_lists_valid)
            gamma_diversity_q_0_valid_struct = gamma_struct_valid['q0_richness']
            gamma_diversity_q_1_valid_struct = gamma_struct_valid['q1_shannon']
            gamma_diversity_q_2_valid_struct = gamma_struct_valid['q2_simpson']

        # Derived sequences for valid-only incremental gains
        undefined_count_valid = []
        sample_space_size_valid = []
        tem_unique_mapping_set = set()
        unique_mapping_learned_valid_io = []

        for er in evaluation_history:
            if isinstance(er, str):
                continue
            if is_valid(er):
                cur_set = set(er.get('unique_mapping_list', []))
                union_set = tem_unique_mapping_set | cur_set
                increase = len(union_set) - len(tem_unique_mapping_set)
                tem_unique_mapping_set = union_set
                unique_mapping_learned_valid_io.append(increase)
                undefined_count_valid.append(er.get('undefined_count'))
                sample_space_size_valid.append(er.get('sample_space_len'))

        # Assemble per-problem result (align with IO version keys)
        evaluation_result = {
            'total_num_hypothesis_generated': total_num_hypothesis_generated,
            'total_num_valid_hypothesis_generated': total_num_valid_hypothesis_generated,
            'unique_mapping_learned': dynamic_evaluation['unique_mapping_learned'],
            'unique_mapping_learned_valid_io': unique_mapping_learned_valid_io,

            'avg_alpha_diversity_q_0': avg_alpha_diversity_q_0,
            'avg_alpha_diversity_q_1': avg_alpha_diversity_q_1,
            'avg_alpha_diversity_q_2': avg_alpha_diversity_q_2,
            'avg_beta_diversity_dist': avg_beta_diversity_dist,
            'avg_beta_diversity_struct': avg_beta_diversity_struct,

            'gamma_diversity_q_0_dist': gamma_diversity_q_0_dist,
            'gamma_diversity_q_1_dist': gamma_diversity_q_1_dist,
            'gamma_diversity_q_2_dist': gamma_diversity_q_2_dist,
            'gamma_diversity_q_0_struct': gamma_diversity_q_0_struct,
            'gamma_diversity_q_1_struct': gamma_diversity_q_1_struct,
            'gamma_diversity_q_2_struct': gamma_diversity_q_2_struct,

            'avg_alpha_diversity_q_0_valid': avg_alpha_diversity_q_0_valid,
            'avg_alpha_diversity_q_1_valid': avg_alpha_diversity_q_1_valid,
            'avg_alpha_diversity_q_2_valid': avg_alpha_diversity_q_2_valid,
            'avg_beta_diversity_dist_valid': avg_beta_diversity_dist_valid,
            'avg_beta_diversity_struct_valid': avg_beta_diversity_struct_valid,
            'gamma_diversity_q_0_valid_dist': gamma_diversity_q_0_valid_dist,
            'gamma_diversity_q_1_valid_dist': gamma_diversity_q_1_valid_dist,
            'gamma_diversity_q_2_valid_dist': gamma_diversity_q_2_valid_dist,
            'gamma_diversity_q_0_valid_struct': gamma_diversity_q_0_valid_struct,
            'gamma_diversity_q_1_valid_struct': gamma_diversity_q_1_valid_struct,
            'gamma_diversity_q_2_valid_struct': gamma_diversity_q_2_valid_struct,

            'undefined_count': [er['undefined_count'] if not isinstance(er, str) else er for er in evaluation_history],
            'undefined_count_valid': undefined_count_valid,
            'sample_space_size': [er['sample_space_len'] if not isinstance(er, str) else er for er in evaluation_history],
            'sample_space_size_valid': sample_space_size_valid,

            'input_io_pair_count': 't3',  # IO field retained; fixed label for t3
            'cur_problem_id': cur_problem_id,
            'all_syntac_parse_error': dynamic_evaluation.get('parse_error_count'),
            'all_consistancy_error': [
                (len(er.get('failed_cases', [])) if not isinstance(er, str) else er) for er in evaluation_history
            ],
            'can_still_generate': False,  # offline eval: no generation here
        }
        eval_results.append(evaluation_result)

    return eval_results

# ----------------- Main -----------------
if __name__ == "__main__":
    # load held-out list
    with open('your held out ids', 'r') as f:
        held_out = json.load(f)  # {dataset_name: [problem_ids]}

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        help=f"choose the model of experiment from {model_mapping_dict}")
    parser.add_argument("-s", "--short_context", action="store_true")
    parser.add_argument("-ss", "--super_short_context", action="store_true")
    parser.add_argument("-c", "--ckp_count", type=str, help="optional checkpoint suffix, e.g. 1000")
    parser.add_argument(
        "-r", "--round_count",            
        type=int,                       
        help=f"number of round want to evaluate",
    )
    args = parser.parse_args()

    assert not (args.short_context and args.super_short_context), \
        "short_context and super_short_context cannot both be true"

    # resolve model name
    model_name = model_mapping_dict[args.model_name]
    if model_name in openai_models:
        model_use_flag = 'use openai model'
    elif model_name in huggingface_models:
        model_use_flag = 'use huggingface model'
        # make a cleaner filename for HF models
        hf_tail = model_name.split('/')[-1]
        
    else:
        raise AssertionError(f"{model_name} not defined, please check the model you are using")

    # resolve short-context roots
    short_tag = '_short_context' if args.short_context else ('_super_short_context' if args.super_short_context else '')
    gen_root, map_root = roots_for_short(short_tag)

    # model tag (includes optional checkpoint/round)
    round_count = int(args.round_count)
    for cur_round_index in range(1,round_count+1):  # Debug here 
        model_tag = qualify_model_tag(model_name, args.ckp_count, cur_round_index)
        model_save_name = model_tag.split('/')[-1]
        eval_file_name = f'final_eval_result_save_t3_{model_save_name}.json'
    
        # where to save the consolidated eval results
        os.makedirs(eval_root_save_dir, exist_ok=True)
        eval_save_file_path = os.path.join(eval_root_save_dir, eval_file_name)


        if os.path.exists(eval_save_file_path):
            print(f"result file already exists, pass: {eval_save_file_path}")
        else:
            eval_results_all: Dict[str, Any] = {}
            # progress bar per dataset
            todo_datasets = [ds for ds in dataset_names if ds in held_out]  # only datasets with held-out ids
            pbar = tqdm(total=len(todo_datasets), desc="Total")
    
            for dataset_name in todo_datasets:
                # collect held-out ids for this dataset
                held_ids = held_out.get(dataset_name, [])
                # evaluate this (model_tag, dataset)
                tem_eval = evaluate_one_datafolder_t3(
                    model_tag=model_tag,
                    dataset_name=dataset_name,
                    gen_root=gen_root,
                    map_root=map_root,
                    held_out_ids=held_ids,
                    print_process=False,
                )
                # in IO version it nested by init_io_pair_count; here we keep a single level keyed by 't3'
                eval_results_all[dataset_name] = tem_eval
                pbar.update(1)
    
            pbar.close()
            with open(eval_save_file_path, 'w') as f:
                json.dump(eval_results_all, f)
            print(f"saved: {eval_save_file_path}")
