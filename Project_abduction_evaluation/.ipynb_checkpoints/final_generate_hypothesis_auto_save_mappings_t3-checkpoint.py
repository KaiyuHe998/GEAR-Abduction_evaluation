# This file can only be used in evaluating fine tuned LLMs since evaluation matric could change 
# This file will only generate hypothesis in held out problems
# This file is used to evaluate t3 hypothesis and mappings

import os
import argparse
from datetime import datetime, timedelta
from global_variables import *
os.environ["HF_HOME"] = "your huggingface root path"                
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_key

save_root_dir = 'generate_log'
# ------------Parameters------------



seed = 2025 # used for sample space collection in evaluation
num_proc = 1 # used for evaluation speed up
prompt_dict = code_prompt_dict_0_3_togather# select current prompt


def print_log_info(finished_number_problems, generated_hypothesis_count, evaluated_hypothesis_count, total_time_used_generation, total_time_used_evaluation):
    avg_time_evaluation_per_hypothesis = total_time_used_evaluation/evaluated_hypothesis_count if evaluated_hypothesis_count > 0 else '---NA---'
    avg_time_generation_per_hypothesis = total_time_used_generation/generated_hypothesis_count if generated_hypothesis_count > 0 else '---NA---'
    avg_hypo_generated_per_problem = round(total_hypothesis/(finished_number_problems),4) if finished_number_problems > 0 else '---NA---'
    total_time_used = datetime.now() - experiment_start_time
    avg_time_per_task = total_time_used/finished_number_problems if finished_number_problems > 0 else '---NA---'
    remaining_time_estimated = avg_time_per_task * (total_tasks-finished_number_problems) if avg_time_per_task!= '---NA---' else '---NA---'
    
    # TODO add current problem hypothesis count
    print(f'''generation_start_time: {experiment_start_time}, cur_time: {datetime.now()}, total_time_used:{total_time_used}, cur_task_progress: {finished_number_problems}/{total_tasks}, avg_time_per_task: {avg_time_per_task}, remaining_time: {remaining_time_estimated}, model_name: {model_name}, dataset_name: {dataset_name}, problem_idx(enumerate): {problem_idx}, avg_time_evaluation_per_hypothesis:{avg_time_evaluation_per_hypothesis}, avg_time_generation_per_hypothesis:{avg_time_generation_per_hypothesis}, avg_hypo_generated_per_problem: {avg_hypo_generated_per_problem}, total_num_hypothesis_evaluated: {evaluated_hypothesis_count}, total_num_hypothesis_generated: {generated_hypothesis_count}, hypothesis_cur_problem: {hypothesis_generated_count_this_problem}, cur_problem_id: {cur_problem_id}, cur_round_index: {cur_round_idx}''')    

from global_variables import *
import json

prompt_dict = code_prompt_dict_0_3_togather
with open('your held out ids', 'r') as f: # subsample arc_2025 dataset into 40 problems from training_data/held_out_ids.json We evaluate on origional dataset
    t3_held_out = json.load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model_name",            
        type=str,                      
        required=True,                 
        help=f"choose the model of experiment that you want to run choose from {model_mapping_dict}",
    )

    parser.add_argument(
        "-s", "--short_context",
        action="store_true",
        help="Enable short context mode",
    )

    parser.add_argument(
        "-ss", "--super_short_context",
        action="store_true",
        help="Enable super short context mode",
    )
    
    parser.add_argument(
        "-g", "--gpu",           
        type=str,                    
        required=True,               
        help=f"choose the model of experiment that you want to run choose from {model_mapping_dict}",
    )
    
    parser.add_argument(
        "-c", "--ckp_count",           
        type=str,                      
        help=f"choose the model checkpoint count of experiment that you want to run choose from {model_mapping_dict}",
    )

    parser.add_argument(
        "-r", "--round_count",         
        type=int,                     
        help=f"number of round want to evaluate",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from datasets import load_dataset
    import transformers
    from typing import *
    from openai import OpenAI
    import ast
    from tqdm import tqdm 
    import json
    import argparse
    import utils
    import generate_utils
    from datasets.utils.logging import disable_progress_bar
    import itertools
    import signal, sys

    

    print(f'using short context? {args.short_context}')
    print(f'using super short context? {args.super_short_context}')
    assert not (args.short_context and args.super_short_context), f'''args.short_context and args.super_short_context can not be true at the same time'''
    # only use the prompt style trained on that model

    round_count = int(args.round_count)
    
    disable_progress_bar()
    # init generate client
    
        
    # load_held out dataset
    ori_problem_datasets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset_name in t3_held_out.keys():
        if dataset_name == 'arc_2025':
            cur_dataset = utils.load_arc_2025_dataset()
        else:
            cur_dataset = utils.load_other_dataset(dataset_name)
        # 过滤：仅保留 held-out 的 problem_id
        cur_dataset = {pid: ex for pid, ex in cur_dataset.items()
                       if pid in t3_held_out[dataset_name]}
        ori_problem_datasets[dataset_name] = cur_dataset
    
    model_name = model_mapping_dict[args.model_name]
    if model_name in openai_models:
        client = OpenAI(api_key = open_ai_key)
        model_use_flag = 'use openai model'
    elif model_name in huggingface_models:
        client = generate_utils.HuggingFaceClient(model_name, ckp_count = args.ckp_count)
        model_use_flag = 'use huggingface model'
    else:
        assert False, f'''{model_name} not defined, please check the model you are using'''
    
    

    # safe usage parameters
    limit_hypothesis_generation = 3                    # at most generate x hypothesis
    limit_context_len = (limit_hypothesis_generation-1)*2+2 
    if args.super_short_context:
        limit_context_len += 1
    

    finished_number_problems = 0
    generated_hypothesis_count = 0
    evaluated_hypothesis_count = 0
    total_hypothesis = 0 # update after each problem is finished
    total_time_used_generation = timedelta(0)
    total_time_used_evaluation = timedelta(0)

    total_tasks = 0
    cur_task = 0

    for dataset_name in ori_problem_datasets:
        total_tasks += len(ori_problem_datasets[dataset_name])
    total_tasks = total_tasks * round_count
    unique_mapping_increase_threshold = 0 # not effective in t3 problem
    
    
    experiment_start_time = datetime.now()
    
    

    print(f'using short context? {args.short_context}')
    print(f'using super short context? {args.super_short_context}')
    assert not (args.short_context and args.super_short_context), f'''args.short_context and args.super_short_context can not be true at the same time'''
    if args.short_context:
        hypothesis_save_root_dir = 'your path'# not used in our paper
    elif args.super_short_context:
        hypothesis_save_root_dir = 'your path'# not used in our paper
    else:
        hypothesis_save_root_dir = 'generate_log/t3'

    if args.short_context:
        mapping_save_root_dir = 'your path' # not used
    elif args.super_short_context:
        mapping_save_root_dir = 'your path' # not used
    else:
        mapping_save_root_dir = './hypothesis_mappings/t3' # Specify your path, this file could be large

    with tqdm(total=total_tasks, desc='TOTAL', unit='problem') as pbar:
        for cur_round_idx in range(1, round_count+1):       # Debug here test generate
            for dataset_name in dataset_names:
                cur_problems = ori_problem_datasets[dataset_name]
                for problem_idx, cur_problem_id in enumerate(cur_problems):
                    hypothesis_generated_count_this_problem = 0
                    continue_generation = True # by default continue generation
                    evaluation_history = []
                    problem = cur_problems[cur_problem_id]
                    cur_hypothesis_save_folder_name = model_name
                    if args.ckp_count is not None:
                        cur_hypothesis_save_folder_name += f'_ckp{str(args.ckp_count)}'
                    if cur_round_idx is not None:
                        cur_hypothesis_save_folder_name += f'_round{str(cur_round_idx)}'
    
                    cur_hypothesis_save_folder_name = os.path.join(hypothesis_save_root_dir, cur_hypothesis_save_folder_name, dataset_name)
                    save_hypothesis_file_path = os.path.join(cur_hypothesis_save_folder_name, f'{cur_problem_id}.json')
                    if not os.path.exists(cur_hypothesis_save_folder_name):
                        os.makedirs(cur_hypothesis_save_folder_name)
                    cur_history_messages = None
                    # Check and load exist history file
    
                    
                    cur_mapping_model_name = model_name
                    if args.ckp_count is not None:
                        cur_mapping_model_name += f'_ckp{str(args.ckp_count)}'
                    if cur_round_idx is not None:
                        cur_mapping_model_name += f'_round{str(cur_round_idx)}'
                    cur_mapping_dir = os.path.join(mapping_save_root_dir, cur_mapping_model_name, dataset_name, str(cur_problem_id))
                    if not os.path.exists(cur_mapping_dir):
                        os.makedirs(cur_mapping_dir)
                    cur_problem_io_pairs = ori_problem_datasets[dataset_name][cur_problem_id]['train']
    
                    if os.path.exists(save_hypothesis_file_path):
                        with open(save_hypothesis_file_path, 'r') as f:
                            cur_history_messages = json.load(f)
                        cur_assistant_responses = [i for i in cur_history_messages if i['role'] == 'assistant']
                        assert len(cur_assistant_responses) > 0, f'''Saved file should have at least one assistant generated function'''
                        for hypothesis_idx, assistant_response in enumerate(cur_assistant_responses):
                            hypothesis_generated_count_this_problem += 1
    
                            
                                
                            mapping_save_file_name = f'{hypothesis_idx}.json'
                            mapping_file = os.path.join(cur_mapping_dir, mapping_save_file_name)
                            print_log_info(finished_number_problems, generated_hypothesis_count, evaluated_hypothesis_count, total_time_used_generation, total_time_used_evaluation)
                            start_time = datetime.now()
                            if os.path.exists(mapping_file):
                                with open(mapping_file, 'r') as f:
                                    evaluate_result = json.load(f)
                            else:
                                evaluate_result = generate_utils.evaluate_one_generation(cur_problem_io_pairs, assistant_response['content'], dataset_name, seed = seed, num_proc = num_proc)
                                with open(mapping_file, 'w') as f:
                                    json.dump(evaluate_result, f)
    
                            evaluation_history.append(evaluate_result)
                            continue_generation, _ = generate_utils.if_continue_generation(evaluation_history, unique_mapping_increase_threshold = unique_mapping_increase_threshold, print_process = False)
                            end_time = datetime.now()
                            cur_time_used_evaluation = end_time - start_time
                            total_time_used_evaluation += cur_time_used_evaluation
                            
                            evaluated_hypothesis_count += 1
                            
                            if not continue_generation:
                                break # with current criteria history message already have all the hypothesis generated
                        if len(cur_history_messages) >= limit_context_len: # safe protect, prevent unlimited generation
                            continue_generation = False 
        
                    # start generation
                    # ------- generate and save -------
                    while continue_generation:
                        print_log_info(
                            finished_number_problems,
                            generated_hypothesis_count,
                            evaluated_hypothesis_count,
                            total_time_used_generation,
                            total_time_used_evaluation
                        )
    
                        start_time = datetime.now()
                        cur_history_messages = generate_utils.generate_response_with_chat_history(
                            model_name,
                            problem['train'],
                            prompt_dict,
                            client,
                            history_messages=cur_history_messages,
                            model_use_flag=model_use_flag,
                            short_context=args.short_context,
                            super_short_context=args.super_short_context
                        )
                        hypothesis_generated_count_this_problem += 1
                        end_time = datetime.now()
                        total_time_used_generation += (end_time - start_time)
                        generated_hypothesis_count += 1
    
                        # save file
                        
                        assert cur_history_messages[-1]['role'] == 'assistant', \
                            f"returned message history should end with assistant's response, {cur_history_messages}"
    
                        new_response = cur_history_messages[-1]['content']
    
                        print_log_info(
                            finished_number_problems,
                            generated_hypothesis_count,
                            evaluated_hypothesis_count,
                            total_time_used_generation,
                            total_time_used_evaluation
                        )
    
                        hypothesis_idx = len([m for m in cur_history_messages if m['role'] == 'assistant']) - 1
                        mapping_file = os.path.join(cur_mapping_dir, f'{hypothesis_idx}.json')
    
                        start_time = datetime.now()
                        if os.path.exists(mapping_file):
                            with open(mapping_file, 'r') as f:
                                evaluate_result = json.load(f)
                        else:
                            evaluate_result = generate_utils.evaluate_one_generation(
                                cur_problem_io_pairs,
                                new_response,
                                dataset_name,
                                seed=seed,
                                num_proc=num_proc
                            )
                            with open(mapping_file, 'w') as f:
                                json.dump(evaluate_result, f)
                                
                        with open(save_hypothesis_file_path, 'w', encoding='utf-8') as f:
                            json.dump(cur_history_messages, f)
                        evaluation_history.append(evaluate_result)
                        end_time = datetime.now()
                        total_time_used_evaluation += (end_time - start_time)
                        evaluated_hypothesis_count += 1
    
                        
                        if len(cur_history_messages) >= limit_context_len:
                            continue_generation = False
                        if len([m for m in cur_history_messages if m['role'] == 'assistant']) >= limit_hypothesis_generation:
                            continue_generation = False
    
                    
                    total_hypothesis += hypothesis_generated_count_this_problem
                    finished_number_problems += 1
                    pbar.update(1)
        