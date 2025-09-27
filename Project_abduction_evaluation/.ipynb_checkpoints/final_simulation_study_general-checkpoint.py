import random
rng = random.Random(2025)
from tqdm import tqdm
import json
import os
import utils
import copy
import numpy as np
import ast
from global_variables import *
import generate_utils
import argparse
from itertools import combinations
import pandas as pd
from datetime import datetime

DEBUG = True
def dbg(msg):
    if DEBUG:
        try:
            from tqdm import tqdm
            tqdm.write(str(msg))
        except Exception:
            print(str(msg), flush=True)

# -----------------------------
# Load data
# -----------------------------
preference_label_data_path = None
assert preference_label_data_path is not None, f'''Please use final_rl_data_gen.py to generate rl data first! And specify your path'''
with open(preference_label_data_path, 'r') as f:
    preference_data = json.load(f)

with open('your held_out_id file path', 'r') as f:
    t3_held_out = json.load(f)

# unified held_out_problem_id 
held_out_problem_id = t3_held_out
ori_problem_datasets = {}
for dataset_name in held_out_problem_id.keys():
    if dataset_name == 'arc_2025':
        cur_dataset = utils.load_arc_2025_dataset()
    else:
        cur_dataset = utils.load_other_dataset(dataset_name)
    ori_problem_datasets.update({dataset_name: cur_dataset})

# Read initial IO pairs
init_io_data = {}
for dataset_name in dataset_names:
    if dataset_name not in init_io_data:
        init_io_data[dataset_name] = {}
    for init_io_count in init_io_pair_counts:
        if dataset_name == 'acre' and init_io_count == 1:
            continue
        file_name = f'{dataset_name}_{init_io_count}.json'
        file_path = os.path.join('../data/io_pairs', dataset_name, file_name)
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        init_io_data[dataset_name][str(init_io_count)] = loaded_data

# -----------------------------
# Helpers
# -----------------------------
def evaluate_one_hypothesis(callable_function, dataset_name, io_pairs):
    failed_cases = {}
    for init_pair_index, init_pair in enumerate(io_pairs):
        if isinstance(init_pair[0], str):
            assert dataset_name == 'list_function', 'only list_function need to parse string input into actual list'
            test_input = ast.literal_eval(init_pair[0])
        else:
            test_input = init_pair[0]

        if isinstance(init_pair[1], str):
            if dataset_name == 'acre':
                test_output = init_pair[1]
            else:
                assert dataset_name == 'list_function', 'only list_function need to parse string input into actual list'
                test_output = ast.literal_eval(init_pair[1])
        else:
            test_output = init_pair[1]

        try:
            cur_input_pass_in = copy.deepcopy(test_input)
            model_output = callable_function(cur_input_pass_in)
            model_output = copy.deepcopy(model_output)
        except Exception as e:
            failed_cases.update({json.dumps(init_pair): [json.dumps(test_output), 'N/A', str(e)]})
        else:
            if test_output != model_output:
                try:
                    failed_cases.update({json.dumps(init_pair): [json.dumps(test_output), json.dumps(model_output), 'output not match']})
                except Exception:
                    failed_cases.update({json.dumps(init_pair): [test_output, 'Error: json can not dump the failed output result', 'output not match']})
    return failed_cases



def get_callbale(input_str):
    description, code_str = generate_utils.parse_string_tuple(input_str)
    if code_str == 'tuple parse error':
        assert False, 'here should be valid hypothesis'
    if not isinstance(code_str, str):
        assert False, 'here should be valid hypothesis'
    try:
        code_str = utils.preprocess_code(code_str)
    except Exception:
        assert False, 'here should be valid hypothesis'

    callable_function = utils.instrument_with_local_guard(code_str)
    if isinstance(callable_function, str):
        print(callable_function)
        assert False, 'here should be valid hypothesis'
    return callable_function


def _init_metrics():
    # Keep consistent with previous behavior: initialize total_count to 0.0001 to avoid division-by-zero in ratios
    return {
        'rejected_better_count': 0,
        'equal_count': 0,
        'equal_pass_count': 0,
        'equal_fail_count': 0,
        'chosen_better_count': 0,
        'total_count': 0.0001
    }


def _rate(x, den):
    return 0.0 if den == 0 else round(float(x) / float(den), 4)


def _metrics_to_json_dict(m):
    total = int(m['total_count'])
    equal = int(m['equal_count'])
    eq_pass = int(m['equal_pass_count'])
    eq_fail = int(m['equal_fail_count'])
    chosen = int(m['chosen_better_count'])
    rejected = int(m['rejected_better_count'])
    return {
        "total_comparisons": total,
        "equal": {
            "count": equal,
            "rate": _rate(equal, total),
            "pass_count": eq_pass,
            "pass_rate": _rate(eq_pass, total),
            "fail_count": eq_fail,
            "fail_rate": _rate(eq_fail, total)
        },
        "chosen_better": {
            "count": chosen,
            "rate": _rate(chosen, total)
        },
        "rejected_better": {
            "count": rejected,
            "rate": _rate(rejected, total)
        }
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # step 1 construct train/test based on init io gen data
    parser = argparse.ArgumentParser()
    parser.add_argument("-tc", "--test_case_count", type=int, required=True,
                        help="number of IO pairs per test set (r in combinations)")
    parser.add_argument("-tr", "--test_case_round", type=int, required=True,
                        help="how many distinct test sets to sample (without replacement)")
    args = parser.parse_args()
    test_sample_case_count = int(args.test_case_count)
    test_round_count = int(args.test_case_round)

    init_io_gen_datasets = {}
    for dataset_name in preference_data:
        if dataset_name not in init_io_gen_datasets:
            init_io_gen_datasets[dataset_name] = {}
        for init_io_count in preference_data[dataset_name]:
            if init_io_count not in init_io_gen_datasets[dataset_name]:
                init_io_gen_datasets[dataset_name][init_io_count] = {}
            for problem_id in preference_data[dataset_name][init_io_count]:
                if problem_id not in init_io_data[dataset_name][init_io_count]:
                    # no data in this split
                    continue
                # hypotheses are generated with these io pairs
                init_io_pairs = init_io_data[dataset_name][init_io_count][problem_id]
                train_data = init_io_pairs

                all_io_pairs = []
                all_io_pairs.extend(ori_problem_datasets[dataset_name][problem_id]['train'])
                all_io_pairs.extend(ori_problem_datasets[dataset_name][problem_id]['test'])
                all_io_pairs = [json.dumps(i) for i in all_io_pairs]
                train_data = [json.dumps(i) for i in train_data]
                assert (train_data[0]) in all_io_pairs, 'data should be in the ori_problem_data'
                unseen_io_pairs = set(all_io_pairs) - set(train_data)
                new_train = init_io_pairs
                new_test = [json.loads(i) for i in list(unseen_io_pairs)]

                # Combinations → sampling: each round uses r IO pairs, sample test_round_count rounds (without replacement)
                if test_sample_case_count <= 0 or test_sample_case_count > len(new_test):
                    continue
                test_cases = list(combinations(new_test, test_sample_case_count))
                if len(test_cases) == 0:
                    continue
                k = min(test_round_count, len(test_cases))
                selected_new_test_cases = rng.sample(test_cases, k)

                for round_idx, single_round_test_case in enumerate(selected_new_test_cases):
                    init_io_gen_problem_data = {
                        f'{problem_id}_{round_idx}': {
                            'train': new_train,
                            'test': list(single_round_test_case)  # tuple -> list
                        }
                    }
                    init_io_gen_datasets[dataset_name][init_io_count].update(init_io_gen_problem_data)

    # step 2 statistics (support dataset-level summary + breakdown by init_io_count)
    # Structure:
    #   dataset_statistic_data[ds]["summary"] = metrics
    #   dataset_statistic_data[ds]["by_init_io_count"][init_io_count] = metrics
    dataset_statistic_data = {
        ds: {
            "summary": _init_metrics(),
            "by_init_io_count": {},
            "by_context_len": {}   # Added
        } for ds in preference_data
    }

    total_iters = sum(
        len(preference_data[ds][io][pid][cl])
        for ds in preference_data
        for io in preference_data[ds]
        for pid in preference_data[ds][io]
        for cl in preference_data[ds][io][pid]
    )

    pbar = tqdm(total=total_iters, desc="Evaluating hypotheses")
    for dataset_name in preference_data:
        for init_io_count in preference_data[dataset_name]:
            for problem_id in preference_data[dataset_name][init_io_count]:
                for context_len in preference_data[dataset_name][init_io_count][problem_id]:
                    for context, chosen, rejected, reason in preference_data[dataset_name][init_io_count][problem_id][context_len]:
                        
                        if reason == 'fair comparsion':
                            dbg(f"[START] ds={dataset_name} | io={init_io_count} | pid={problem_id} | ctx_len={context_len} | reason={reason}")
                            # Find all actual rounds that exist for this problem
                            round_bucket = init_io_gen_datasets.get(dataset_name, {}).get(init_io_count, {})
                            available_round_ids = [rid for rid in round_bucket.keys() if rid.startswith(f"{problem_id}_")]
                            test_case_count = [len(round_bucket[rid]['test']) for rid in available_round_ids]
                            dbg(f"[available_round_ids] count={len(available_round_ids)} | sample={available_round_ids[:5]}| test_case_count = {test_case_count}")
                            if not available_round_ids:
                                dbg("[SKIP] no available rounds -> skip this (counts won't change)")
                                # This problem generated no rounds; skip it
                                pbar.update(1)
                                continue

                            ds_summary = dataset_statistic_data[dataset_name]["summary"]
                            ds_by_io = dataset_statistic_data[dataset_name]["by_init_io_count"]
                            if init_io_count not in ds_by_io:
                                ds_by_io[init_io_count] = _init_metrics()
                            io_metrics = ds_by_io[init_io_count]


                            ds_by_ctx = dataset_statistic_data[dataset_name]["by_context_len"]
                            if context_len not in ds_by_ctx:
                                ds_by_ctx[context_len] = _init_metrics()
                            ctx_metrics = ds_by_ctx[context_len]

                            # Pre-parse chosen/rejected
                            callable_chosen = get_callbale(chosen)
                            callable_rejected = get_callbale(rejected)

                            # Sanity check on the first round's train set (train is the same across rounds)
                            rep_train_set = round_bucket[available_round_ids[0]]['train']
                            assert len(evaluate_one_hypothesis(callable_chosen,   dataset_name, rep_train_set)) == 0, \
                                'here should be valid hypothesis'
                            assert len(evaluate_one_hypothesis(callable_rejected, dataset_name, rep_train_set)) == 0, \
                                'here should be valid hypothesis'

                            # Parse hypotheses from context in advance
                            context_hypothesis_strs = [i['content'] for i in context if i['role'] == 'assistant']
                            context_funcs = []
                            for s in context_hypothesis_strs:
                                try:
                                    context_funcs.append(get_callbale(s))
                                except AssertionError:
                                    continue

                            # Evaluate round by round
                            for rid in available_round_ids:
                                pack = round_bucket[rid]
                                train_set = pack['train']
                                test_set  = pack['test']

                                # Check whether any hypothesis in the context already passes this round's test; if yes, skip this round
                                any_context_pass = False
                                for cf in context_funcs:
                                    assert len(evaluate_one_hypothesis(cf, dataset_name, train_set)) == 0, \
                                        'here should be valid hypothesis'
                                    if len(evaluate_one_hypothesis(cf, dataset_name, test_set)) == 0:
                                        any_context_pass = True
                                        break
                                if any_context_pass:
                                    continue  # This round doesn't require comparison

                                # Enter comparison (this round counts as one)
                                ds_summary['total_count'] += 1
                                io_metrics['total_count'] += 1
                                ctx_metrics['total_count'] += 1 

                                # Compare on this round's test set
                                c_cnt = len(evaluate_one_hypothesis(callable_chosen,   dataset_name, test_set))
                                r_cnt = len(evaluate_one_hypothesis(callable_rejected, dataset_name, test_set))

                                if r_cnt == c_cnt:
                                    ds_summary['equal_count'] += 1
                                    io_metrics['equal_count'] += 1
                                    ctx_metrics['equal_count'] += 1         
                                    if c_cnt == 0:
                                        ds_summary['equal_pass_count'] += 1
                                        io_metrics['equal_pass_count'] += 1
                                        ctx_metrics['equal_pass_count'] += 1  
                                    else:
                                        ds_summary['equal_fail_count'] += 1
                                        io_metrics['equal_fail_count'] += 1
                                        ctx_metrics['equal_fail_count'] += 1  
                                elif r_cnt > c_cnt:
                                    ds_summary['chosen_better_count'] += 1
                                    io_metrics['chosen_better_count'] += 1
                                    ctx_metrics['chosen_better_count'] += 1   
                                else:
                                    ds_summary['rejected_better_count'] += 1
                                    io_metrics['rejected_better_count'] += 1
                                    ctx_metrics['rejected_better_count'] += 1 

                        # Progress bar printing (based on summary)
                        summary_parts = []
                        for ds, pack in dataset_statistic_data.items():
                            stats = pack["summary"]
                            total = stats['total_count']
                            equal = stats['equal_count']
                            eq_pass = stats['equal_pass_count']
                            eq_fail = stats['equal_fail_count']
                            chosen = stats['chosen_better_count']
                            rejected = stats['rejected_better_count']
                            denom = max(1, int(total))
                            summary_parts.append(
                                f"{ds}: total={int(total)}, "
                                f"eq={equal}({round(equal/denom,4)}), "
                                f"eq_pass={eq_pass}({round(eq_pass/denom,4)}), "
                                f"eq_fail={eq_fail}({round(eq_fail/denom,4)}), "
                                f"ch={chosen}({round(chosen/denom,4)}), "
                                f"rej={rejected}({round(rejected/denom,4)})"
                            )
                        print(" | ".join(summary_parts), end='\r')
                        pbar.update(1)

    pbar.close()

    # Print results (dataset-level summary + breakdown by init_io_count)
    print("\n=== Per-dataset statistics ===")
    for ds, pack in dataset_statistic_data.items():
        stats = pack["summary"]
        total = max(1, stats['total_count'])
        equal = stats['equal_count']
        eq_pass = stats['equal_pass_count']
        eq_fail = stats['equal_fail_count']
        chosen = stats['chosen_better_count']
        rejected = stats['rejected_better_count']

        print(f"\nDataset: {ds}")
        print(f"  Total comparisons: {int(stats['total_count'])}")
        print(f"  Equal count: {equal} ({equal/total:.4f})")
        print(f"    Equal PASS: {eq_pass} ({eq_pass/total:.4f})")
        print(f"    Equal FAIL: {eq_fail} ({eq_fail/total:.4f})")
        print(f"  Chosen better count: {chosen} ({chosen/total:.4f})")
        print(f"  Rejected better count: {rejected} ({rejected/total:.4f})")

        # by init_io_count
        if pack["by_init_io_count"]:
            print("  -- by init_io_count --")
            for io_k, io_m in sorted(pack["by_init_io_count"].items(), key=lambda x: str(x[0])):
                io_total = max(1, io_m['total_count'])
                print(f"    * {io_k}: total={int(io_m['total_count'])}, "
                      f"eq={io_m['equal_count']}({io_m['equal_count']/io_total:.4f}), "
                      f"eq_pass={io_m['equal_pass_count']}({io_m['equal_pass_count']/io_total:.4f}), "
                      f"eq_fail={io_m['equal_fail_count']}({io_m['equal_fail_count']/io_total:.4f}), "
                      f"ch={io_m['chosen_better_count']}({io_m['chosen_better_count']/io_total:.4f}), "
                      f"rej={io_m['rejected_better_count']}({io_m['rejected_better_count']/io_total:.4f})")

        if pack["by_context_len"]:
            print("  -- by_context_len --")
            for ctx_k, ctx_m in sorted(pack["by_context_len"].items(), key=lambda x: str(x[0])):
                ctx_total = max(1, ctx_m['total_count'])
                print(f"    * {ctx_k}: total={int(ctx_m['total_count'])}, "
                      f"eq={ctx_m['equal_count']}({ctx_m['equal_count']/ctx_total:.4f}), "
                      f"eq_pass={ctx_m['equal_pass_count']}({ctx_m['equal_pass_count']/ctx_total:.4f}), "
                      f"eq_fail={ctx_m['equal_fail_count']}({ctx_m['equal_fail_count']/ctx_total:.4f}), "
                      f"ch={ctx_m['chosen_better_count']}({ctx_m['chosen_better_count']/ctx_total:.4f}), "
                      f"rej={ctx_m['rejected_better_count']}({ctx_m['rejected_better_count']/ctx_total:.4f})")

    # Overall summary (based on each dataset's summary)
    overall_total = sum(pack["summary"]['total_count'] for pack in dataset_statistic_data.values())
    overall_equal = sum(pack["summary"]['equal_count'] for pack in dataset_statistic_data.values())
    overall_eq_pass = sum(pack["summary"]['equal_pass_count'] for pack in dataset_statistic_data.values())
    overall_eq_fail = sum(pack["summary"]['equal_fail_count'] for pack in dataset_statistic_data.values())
    overall_chosen = sum(pack["summary"]['chosen_better_count'] for pack in dataset_statistic_data.values())
    overall_rejected = sum(pack["summary"]['rejected_better_count'] for pack in dataset_statistic_data.values())

    den = max(1, overall_total)
    print("\n=== Overall statistics ===")
    print(f"Total comparisons: {int(overall_total)}")
    print(f"Equal count: {int(overall_equal)} ({overall_equal/den:.4f})")
    print(f"  Equal PASS: {int(overall_eq_pass)} ({overall_eq_pass/den:.4f})")
    print(f"  Equal FAIL: {int(overall_eq_fail)} ({overall_eq_fail/den:.4f})")
    print(f"Chosen better count: {int(overall_chosen)} ({overall_chosen/den:.4f})")
    print(f"Rejected better count: {int(overall_rejected)} ({overall_rejected/den:.4f})")

    # -----------------------------
    # JSON summary (includes by_init_io_count and by_context_len)
    # -----------------------------
    per_dataset_json = {}
    for ds, pack in dataset_statistic_data.items():
        # dataset summary
        per_dataset_json[ds] = _metrics_to_json_dict(pack["summary"])

        # add by_init_io_count
        per_dataset_json[ds]["by_init_io_count"] = {}
        for io_k, io_m in pack["by_init_io_count"].items():
            per_dataset_json[ds]["by_init_io_count"][str(io_k)] = _metrics_to_json_dict(io_m)

        # add by_context_len
        per_dataset_json[ds]["by_context_len"] = {}
        for ctx_k, ctx_m in pack["by_context_len"].items():
            per_dataset_json[ds]["by_context_len"][str(ctx_k)] = _metrics_to_json_dict(ctx_m)

    overall_json = {
        "total_comparisons": int(overall_total),
        "equal": {
            "count": int(overall_equal),
            "rate": _rate(overall_equal, overall_total),
            "pass_count": int(overall_eq_pass),
            "pass_rate": _rate(overall_eq_pass, overall_total),
            "fail_count": int(overall_eq_fail),
            "fail_rate": _rate(overall_eq_fail, overall_total)
        },
        "chosen_better": {
            "count": int(overall_chosen),
            "rate": _rate(overall_chosen, overall_total)
        },
        "rejected_better": {
            "count": int(overall_rejected),
            "rate": _rate(overall_rejected, overall_total)
        }
    }

    results_to_save = {
        "meta": {
            "analysis_name": "final_meta_analysis_1",
            "seed": 2025,
            "timestamp": datetime.now().isoformat()
        },
        "per_dataset": per_dataset_json,
        "overall": overall_json
    }

    os.makedirs("meta_analysis_outputs", exist_ok=True)
    out_path = os.path.join("meta_analysis_outputs", f"final_meta_analysis_1_summary_{args.test_case_count}_{args.test_case_round}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)

    print(f"\n✅ JSON summary saved to: {out_path}")
