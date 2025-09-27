# final_meta_analysis_alignment.py
# For different contexts, compute the violation probability of a "single hypothesis" on hidden cases (sampled test rounds)
# + Based on cached mapping files, compute the diversity of the "passing set per round" (β/γ)
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
from datetime import datetime


def find_hypo_mapping_file(hypothesis_str, init_io, dataset_name, problem_id):
    root_hypothesis_save_path = 'generate_log'
    all_model_names = list(model_mapping_dict.values())[:8]
    hypothesis_str = generate_utils.parse_string_tuple(hypothesis_str)[0]  # align generation logs via description
    for model_name in all_model_names:
        cur_save_path = os.path.join(
            root_hypothesis_save_path, model_name, dataset_name,
            f'init_io_pair_count_{init_io}', f'{problem_id}.json'
        )
        if not os.path.exists(cur_save_path):
            continue
        with open(cur_save_path, 'r') as f:
            cur_generate_log = json.load(f)
        cur_all_hypothesis = [i['content'] for i in cur_generate_log if i.get('role') == 'assistant']
        cur_all_hypothesis_tuples = [generate_utils.parse_string_tuple(i)[0] for i in cur_all_hypothesis]
        for index, cur_hypothesis_str in enumerate(cur_all_hypothesis_tuples):
            if hypothesis_str == cur_hypothesis_str:
                path = os.path.join(model_name, dataset_name, f'init_io_{init_io}', str(problem_id), f'{index}.json')
                mapping_file = get_mapping(path)
                assert not isinstance(mapping_file, str), 'here should be a valid hypothesis'
                return mapping_file
    # If not found, raise (the caller will try/except and skip this hypothesis)
    raise FileNotFoundError(f'hypothesis not found {(hypothesis_str, init_io, dataset_name, problem_id)}')

def get_mapping(path):
    mapping_save_root_path = './hypothesis_mappings'  # the path where you save mappings
    file_path = os.path.join(mapping_save_root_path, path)
    with open(file_path, 'r') as f:
        mapping_file = json.load(f)
    return mapping_file

# -----------------------------
# Misc
# -----------------------------
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




held_out_problem_id = t3_held_out
ori_problem_datasets = {}
for dataset_name in held_out_problem_id.keys():
    if dataset_name == 'arc_2025':
        cur_dataset = utils.load_arc_2025_dataset()
    else:
        cur_dataset = utils.load_other_dataset(dataset_name)
    ori_problem_datasets.update({dataset_name: cur_dataset})


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
# Helpers - alignment evaluation
# -----------------------------
def evaluate_one_hypothesis(callable_function, dataset_name, io_pairs):
    """Return a dict of failures: {io_pair_json: [gold, pred, msg]}; empty dict means the entire round passed."""
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

def get_callable(input_str):
    """Parse a (desc, code) string into an executable function; if invalid / not passing the train set, the caller will assert or skip."""
    description, code_str = generate_utils.parse_string_tuple(input_str)
    if code_str == 'tuple parse error':
        raise AssertionError('here should be valid hypothesis (tuple parse error)')
    if not isinstance(code_str, str):
        raise AssertionError('here should be valid hypothesis (code not str)')
    try:
        code_str = utils.preprocess_code(code_str)
    except Exception:
        raise AssertionError('here should be valid hypothesis (preprocess failed)')
    callable_function = utils.instrument_with_local_guard(code_str)
    if isinstance(callable_function, str):
        dbg(callable_function)
        raise AssertionError('here should be valid hypothesis (instrument failed)')
    return callable_function

def _init_align_metrics():
    return {
        'total_rounds': 0,
        'pass_rounds': 0,
        'fail_rounds': 0,
        'sum_failed_cases': 0,
        'sum_total_cases': 0
    }

def _update_align_metrics(m, failed_cnt, total_cnt):
    m['total_rounds'] += 1
    if failed_cnt == 0:
        m['pass_rounds'] += 1
    else:
        m['fail_rounds'] += 1
    m['sum_failed_cases'] += int(failed_cnt)
    m['sum_total_cases']  += int(total_cnt)

def _rate(num, den):
    return 0.0 if den == 0 else round(float(num) / float(den), 4)

def _metrics_to_json_align(m):
    total = int(m['total_rounds'])
    pass_r = int(m['pass_rounds'])
    fail_r = int(m['fail_rounds'])
    sum_fail = int(m['sum_failed_cases'])
    sum_total = int(m['sum_total_cases'])
    return {
        "rounds": {
            "total": total,
            "pass": pass_r,
            "fail": fail_r,
            "pass_rate": _rate(pass_r, total),
            "fail_rate": _rate(fail_r, total)
        },
        "avg_failed_cases_per_round": _rate(sum_fail, total),
        "failed_fraction_overall": _rate(sum_fail, sum_total)
    }

# -----------------------------
# Helpers - diversity statistics (based on cached mapping)
# -----------------------------
def _init_diversity_metrics():
    # Only compute γ, with two views: any-pass and multi-pass; remove all β fields
    return {
        'rounds_any': 0, 'rounds_multi': 0,

        # Any-pass rounds (≥1)
        'sum_gd_q0_any': 0.0, 'sum_gd_q1_any': 0.0, 'sum_gd_q2_any': 0.0, 'cnt_gd_any': 0,
        'sum_gs_q0_any': 0.0, 'sum_gs_q1_any': 0.0, 'sum_gs_q2_any': 0.0, 'cnt_gs_any': 0,

        # Multi-pass rounds (≥2)
        'sum_gd_q0_multi': 0.0, 'sum_gd_q1_multi': 0.0, 'sum_gd_q2_multi': 0.0, 'cnt_gd_multi': 0,
        'sum_gs_q0_multi': 0.0, 'sum_gs_q1_multi': 0.0, 'sum_gs_q2_multi': 0.0, 'cnt_gs_multi': 0,
    }

def _avg(sum_v, cnt_v):
    return None if cnt_v == 0 else round(sum_v / cnt_v, 6)

def _finalize_diversity(agg: dict):
    # Return only the average γ (q0/q1/q2); β is not included
    return {
        "rounds_any": agg['rounds_any'],
        "rounds_multi": agg['rounds_multi'],
        "avg_gamma_dist_any": {
            "q0": _avg(agg['sum_gd_q0_any'], agg['cnt_gd_any']),
            "q1": _avg(agg['sum_gd_q1_any'], agg['cnt_gd_any']),
            "q2": _avg(agg['sum_gd_q2_any'], agg['cnt_gd_any']),
        },
        "avg_gamma_struct_any": {
            "q0": _avg(agg['sum_gs_q0_any'], agg['cnt_gs_any']),
            "q1": _avg(agg['sum_gs_q1_any'], agg['cnt_gs_any']),
            "q2": _avg(agg['sum_gs_q2_any'], agg['cnt_gs_any']),
        },
        "avg_gamma_dist_multi": {
            "q0": _avg(agg['sum_gd_q0_multi'], agg['cnt_gd_multi']),
            "q1": _avg(agg['sum_gd_q1_multi'], agg['cnt_gd_multi']),
            "q2": _avg(agg['sum_gd_q2_multi'], agg['cnt_gd_multi']),
        },
        "avg_gamma_struct_multi": {
            "q0": _avg(agg['sum_gs_q0_multi'], agg['cnt_gs_multi']),
            "q1": _avg(agg['sum_gs_q1_multi'], agg['cnt_gs_multi']),
            "q2": _avg(agg['sum_gs_q2_multi'], agg['cnt_gs_multi']),
        },
    }

def _merge_diversity(dst: dict, src: dict):
    for k in dst.keys():
        if k.startswith('rounds_') or k.startswith('cnt_'):
            dst[k] += src[k]
        elif k.startswith('sum_'):
            dst[k] += src[k]

def _accumulate_diversity(agg: dict, mapping_results: list, is_multi: bool):
    """
    Only compute γ, and for each problem/round, normalize γ(q0/q1/q2) by sample_space_len before aggregating.
    mapping_results: List[dict] (each element is a deserialized mapping file for one hypothesis)
                     Required fields: unique_mapping_list, sample_space_len
    """
    # 1) Build the list of ALL mappings
    all_maps = []
    for r in mapping_results:
        if isinstance(r, dict):
            all_maps.append([tuple(p) for p in r.get("unique_mapping_list", [])])
        else:
            all_maps.append([])

    # 2) Compute γ (dist / struct)
    gd = utils.calculate_gamma_diversity_dist(all_maps)   # {'q0_richness','q1_shannon','q2_simpson'}
    gs = utils.calculate_gamma_diversity_struct(all_maps)

    # 3) Read sample_space_len directly from mapping (take the first valid one)
    sp = None
    for r in mapping_results:
        if isinstance(r, dict) and isinstance(r.get("sample_space_len"), int) and r["sample_space_len"] > 0:
            sp = r["sample_space_len"]
            break
    if sp is None:
        dbg("[warn] sample_space_len not found; skip normalization")

    # 4) Normalize (if we have sp)
    if sp:
        gd_q0 = gd['q0_richness'] / sp
        gd_q1 = gd['q1_shannon'] / sp
        gd_q2 = gd['q2_simpson'] / sp
        gs_q0 = gs['q0_richness'] / sp
        gs_q1 = gs['q1_shannon'] / sp
        gs_q2 = gs['q2_simpson'] / sp
    else:
        gd_q0, gd_q1, gd_q2 = gd['q0_richness'], gd['q1_shannon'], gd['q2_simpson']
        gs_q0, gs_q1, gs_q2 = gs['q0_richness'], gs['q1_shannon'], gs['q2_simpson']

    # 5) Accumulate (any-pass rounds)
    agg['rounds_any'] += 1
    agg['sum_gd_q0_any'] += gd_q0; agg['sum_gd_q1_any'] += gd_q1; agg['sum_gd_q2_any'] += gd_q2; agg['cnt_gd_any'] += 1
    agg['sum_gs_q0_any'] += gs_q0; agg['sum_gs_q1_any'] += gs_q1; agg['sum_gs_q2_any'] += gs_q2; agg['cnt_gs_any'] += 1

    # 6) Accumulate (multi-pass rounds, ≥2)
    if is_multi:
        agg['rounds_multi'] += 1
        agg['sum_gd_q0_multi'] += gd_q0; agg['sum_gd_q1_multi'] += gd_q1; agg['sum_gd_q2_multi'] += gd_q2; agg['cnt_gd_multi'] += 1
        agg['sum_gs_q0_multi'] += gs_q0; agg['sum_gs_q1_multi'] += gs_q1; agg['sum_gs_q2_multi'] += gs_q2; agg['cnt_gs_multi'] += 1

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # step 1: From original data, build multiple rounds with "fixed train, sampled test"
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
                    continue
                init_io_pairs = init_io_data[dataset_name][init_io_count][problem_id]

                # Aggregate all available IO for the problem (train+test), remove train to get unseen candidate pool
                all_io_pairs = []
                all_io_pairs.extend(ori_problem_datasets[dataset_name][problem_id]['train'])
                all_io_pairs.extend(ori_problem_datasets[dataset_name][problem_id]['test'])
                all_io_pairs = [json.dumps(i) for i in all_io_pairs]
                train_data = [json.dumps(i) for i in init_io_pairs]
                assert (train_data[0]) in all_io_pairs, 'data should be in the ori_problem_data'
                unseen_io_pairs = set(all_io_pairs) - set(train_data)
                new_train = init_io_pairs
                new_test = [json.loads(i) for i in list(unseen_io_pairs)]

                # Combinations → sampling
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

    # step 2: Statistic containers (alignment + diversity)
    dataset_statistic_data = {
        ds: {
            "summary": _init_align_metrics(),
            "by_init_io_count": {},
            "by_context_len": {},
            "diversity": {
                "summary": _init_diversity_metrics(),
                "by_init_io_count": {},
                "by_context_len": {},
            }
        } for ds in preference_data
    }

    # Dedup evaluation + cache passing rounds
    evaluated_hypo_keys = set()  # (ds, io, pid, hypo_str)
    pass_rounds_cache = {}       # (ds, io, pid, hypo_str) -> set(rid)

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
                # Find all rounds that actually exist for this problem
                round_bucket = init_io_gen_datasets.get(dataset_name, {}).get(init_io_count, {})
                available_round_ids = [rid for rid in round_bucket.keys() if rid.startswith(f"{problem_id}_")]
                if not available_round_ids:
                    num_items = sum(len(preference_data[dataset_name][init_io_count][problem_id][cl])
                                    for cl in preference_data[dataset_name][init_io_count][problem_id])
                    pbar.update(num_items)
                    continue

                rep_train_set = round_bucket[available_round_ids[0]]['train']  # same train set across rounds

                for context_len in preference_data[dataset_name][init_io_count][problem_id]:
                    # Record the passing set per round
                    round_pass_map = {rid: [] for rid in available_round_ids}

                    for context, chosen, rejected, reason in preference_data[dataset_name][init_io_count][problem_id][context_len]:
                        if reason != 'fair comparsion':
                            pbar.update(1)
                            continue

                        dbg(f"[START] ds={dataset_name} | io={init_io_count} | pid={problem_id} | ctx_len={context_len} | reason={reason}")
                        dbg(f"[available_round_ids] count={len(available_round_ids)} | sample={available_round_ids[:5]} | test_case_count={[len(round_bucket[r]['test']) for r in available_round_ids]}")

                        # Get containers at each level
                        ds_pack = dataset_statistic_data[dataset_name]
                        ds_summary = ds_pack["summary"]

                        ds_by_io = ds_pack["by_init_io_count"]
                        if init_io_count not in ds_by_io:
                            ds_by_io[init_io_count] = _init_align_metrics()
                        io_metrics = ds_by_io[init_io_count]

                        ds_by_ctx = ds_pack["by_context_len"]
                        if context_len not in ds_by_ctx:
                            ds_by_ctx[context_len] = _init_align_metrics()
                        ctx_metrics = ds_by_ctx[context_len]

                        # Gather candidate hypotheses: assistant outputs in context + chosen + rejected (deduplicated, order-preserving)
                        context_hypo_strs = [i['content'] for i in context if i.get('role') == 'assistant']
                        candidates = list(dict.fromkeys(context_hypo_strs + [chosen, rejected]))

                        for hypo_str in candidates:
                            dedup_key = (dataset_name, str(init_io_count), str(problem_id), hypo_str)
                            cache_key = dedup_key

                            if dedup_key in evaluated_hypo_keys:
                                # Restore previously passing rounds
                                for rid in available_round_ids:
                                    if rid in pass_rounds_cache.get(cache_key, set()):
                                        round_pass_map[rid].append(hypo_str)
                                continue

                            # First-time evaluation of this hypothesis (under this problem + io)
                            try:
                                fn = get_callable(hypo_str)
                            except AssertionError:
                                continue
                            if len(evaluate_one_hypothesis(fn, dataset_name, rep_train_set)) != 0:
                                continue

                            passed_rids = pass_rounds_cache.setdefault(cache_key, set())

                            for rid in available_round_ids:
                                pack_round = round_bucket[rid]
                                test_set = pack_round['test']
                                failed = evaluate_one_hypothesis(fn, dataset_name, test_set)
                                failed_cnt = len(failed)
                                round_size = len(test_set)

                                _update_align_metrics(ds_summary, failed_cnt, round_size)
                                _update_align_metrics(io_metrics, failed_cnt, round_size)
                                _update_align_metrics(ctx_metrics, failed_cnt, round_size)

                                if failed_cnt == 0:
                                    round_pass_map[rid].append(hypo_str)
                                    passed_rids.add(rid)

                            evaluated_hypo_keys.add(dedup_key)

                        # Progress bar (advance once per sample)
                        mini = []
                        for ds, pack_d in dataset_statistic_data.items():
                            s = pack_d["summary"]
                            total = max(1, s['total_rounds'])
                            mini.append(f"{ds}: rounds={s['total_rounds']} pass={s['pass_rounds']}({_rate(s['pass_rounds'], total)}) fail={s['fail_rounds']}({_rate(s['fail_rounds'], total)})")
                        print(" | ".join(mini), end='\r')
                        pbar.update(1)

                    # ============ Diversity accumulation (based on cached mapping) ============
                    div_pack = dataset_statistic_data[dataset_name]["diversity"]
                    div_summary = div_pack["summary"]

                    div_by_io = div_pack["by_init_io_count"]
                    if init_io_count not in div_by_io:
                        div_by_io[init_io_count] = _init_diversity_metrics()
                    div_io_metrics = div_by_io[init_io_count]

                    div_by_ctx = div_pack["by_context_len"]
                    if context_len not in div_by_ctx:
                        div_by_ctx[context_len] = _init_diversity_metrics()
                    div_ctx_metrics = div_by_ctx[context_len]

                    for rid, pass_list in round_pass_map.items():
                        # Dedup to avoid counting the same hypothesis multiple times
                        pass_list = list(dict.fromkeys(pass_list))
                        if len(pass_list) < 1:
                            continue

                        # Read each hypothesis mapping (skip if missing)
                        mapping_results = []
                        for h in pass_list:
                            try:
                                mp = find_hypo_mapping_file(h, init_io_count, dataset_name, problem_id)
                                mapping_results.append(mp)
                            except Exception as e:
                                dbg(f"[mapping miss] ds={dataset_name} io={init_io_count} pid={problem_id} hypo=... :: {e}")
                                continue

                        if len(mapping_results) < 1:
                            continue

                        is_multi = (len(mapping_results) >= 2)
                        _accumulate_diversity(div_summary, mapping_results, is_multi=is_multi)
                        _accumulate_diversity(div_io_metrics, mapping_results, is_multi=is_multi)
                        _accumulate_diversity(div_ctx_metrics, mapping_results, is_multi=is_multi)

    pbar.close()

    # step 3: Print/save results
    print("\n\n=== Per-dataset statistics (alignment + diversity) ===")
    per_dataset_json = {}
    overall_metrics = _init_align_metrics()

    for ds, pack in dataset_statistic_data.items():
        s = pack["summary"]
        total = max(1, s['total_rounds'])
        print(f"\nDataset: {ds}")
        print(f"  Rounds total: {s['total_rounds']}")
        print(f"  Pass rounds:  {s['pass_rounds']} ({_rate(s['pass_rounds'], total):.4f})")
        print(f"  Fail rounds:  {s['fail_rounds']} ({_rate(s['fail_rounds'], total):.4f})")
        print(f"  Avg failed cases/round: {_rate(s['sum_failed_cases'], total):.4f}")
        print(f"  Failed fraction overall: {_rate(s['sum_failed_cases'], s['sum_total_cases']):.4f}")

        # Aggregate overall alignment metrics
        for k in overall_metrics.keys():
            overall_metrics[k] += s[k]

        # by_init_io_count
        if pack["by_init_io_count"]:
            print("  -- by_init_io_count --")
            for io_k, io_m in sorted(pack["by_init_io_count"].items(), key=lambda x: str(x[0])):
                t = max(1, io_m['total_rounds'])
                print(f"    * {io_k}: rounds={io_m['total_rounds']}, "
                      f"pass={io_m['pass_rounds']}({_rate(io_m['pass_rounds'], t):.4f}), "
                      f"fail={io_m['fail_rounds']}({_rate(io_m['fail_rounds'], t):.4f}), "
                      f"avg_fail_cases/round={_rate(io_m['sum_failed_cases'], t):.4f}, "
                      f"failed_fraction={_rate(io_m['sum_failed_cases'], io_m['sum_total_cases']):.4f}")

        # by_context_len
        if pack["by_context_len"]:
            print("  -- by_context_len --")
            for ctx_k, ctx_m in sorted(pack["by_context_len"].items(), key=lambda x: str(x[0])):
                t = max(1, ctx_m['total_rounds'])
                print(f"    * {ctx_k}: rounds={ctx_m['total_rounds']}, "
                      f"pass={ctx_m['pass_rounds']}({_rate(ctx_m['pass_rounds'], t):.4f}), "
                      f"fail={ctx_m['fail_rounds']}({_rate(ctx_m['fail_rounds'], t):.4f}), "
                      f"avg_fail_cases/round={_rate(ctx_m['sum_failed_cases'], t):.4f}, "
                      f"failed_fraction={_rate(ctx_m['sum_failed_cases'], ctx_m['sum_total_cases']):.4f}")

        # Assemble per-dataset JSON (alignment)
        per_dataset_json[ds] = _metrics_to_json_align(pack["summary"])
        per_dataset_json[ds]["by_init_io_count"] = {str(io_k): _metrics_to_json_align(io_m)
                                                    for io_k, io_m in pack["by_init_io_count"].items()}
        per_dataset_json[ds]["by_context_len"] = {str(ctx_k): _metrics_to_json_align(ctx_m)
                                                  for ctx_k, ctx_m in pack["by_context_len"].items()}

        # Print & append diversity
        div_summary_final = _finalize_diversity(pack["diversity"]["summary"])
        print("  Diversity (passing hyps per round):")
        print(f"    rounds_any={div_summary_final['rounds_any']}, rounds_multi={div_summary_final['rounds_multi']}")
        
        print(f"    avg_gamma_dist_any(q0/q1/q2): "
              f"{div_summary_final['avg_gamma_dist_any']['q0']}/"
              f"{div_summary_final['avg_gamma_dist_any']['q1']}/"
              f"{div_summary_final['avg_gamma_dist_any']['q2']}")
        print(f"    avg_gamma_struct_any(q0/q1/q2): "
              f"{div_summary_final['avg_gamma_struct_any']['q0']}/"
              f"{div_summary_final['avg_gamma_struct_any']['q1']}/"
              f"{div_summary_final['avg_gamma_struct_any']['q2']}")

        per_dataset_json[ds]["diversity"] = div_summary_final
        per_dataset_json[ds]["diversity_by_init_io_count"] = {
            str(io_k): _finalize_diversity(io_m)
            for io_k, io_m in pack["diversity"]["by_init_io_count"].items()
        }
        per_dataset_json[ds]["diversity_by_context_len"] = {
            str(ctx_k): _finalize_diversity(ctx_m)
            for ctx_k, ctx_m in pack["diversity"]["by_context_len"].items()
        }

    # Overall
    overall_json = _metrics_to_json_align(overall_metrics)
    overall_div = _init_diversity_metrics()
    for _ds, _pack in dataset_statistic_data.items():
        _merge_diversity(overall_div, _pack["diversity"]["summary"])
    overall_div_json = _finalize_diversity(overall_div)

    results_to_save = {
        "meta": {
            "analysis_name": "final_meta_analysis_alignment",
            "seed": 2025,
            "timestamp": datetime.now().isoformat(),
            "test_case_count": test_sample_case_count,
            "test_case_round": test_round_count
        },
        "per_dataset": per_dataset_json,
        "overall": overall_json,
        "diversity_overall": overall_div_json
    }

    os.makedirs("meta_analysis_outputs", exist_ok=True)
    out_path = os.path.join(
        "meta_analysis_outputs",
        f"final_meta_analysis_defeasible_summary_{args.test_case_count}_{args.test_case_round}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)

    print(f"\n✅ JSON summary saved to: {out_path}")
