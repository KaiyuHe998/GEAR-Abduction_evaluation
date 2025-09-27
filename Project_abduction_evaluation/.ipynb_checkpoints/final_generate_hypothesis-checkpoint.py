import os
from datasets import load_dataset
import transformers
from typing import *
from openai import OpenAI
import ast
from tqdm import tqdm 
import json
import argparse
import utils
from global_variables import *
import generate_utils
from datasets.utils.logging import disable_progress_bar
from datetime import datetime, timedelta
import itertools
import signal, sys
import argparse
import os
os.environ["HF_HOME"] = "your huggingface home path"                
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_key

save_root_dir = 'generate_log' # your generated hypothesis will be saved to this path
# ------------Parameters------------

group_indexs = [0,1,2,3,4] # you can run the experiment in groups
group_problem_count = 20
seed = 2025 # used for sample space collection in evaluation
num_proc = 8 # used for evaluation speed up
prompt_dict = code_prompt_dict_0_3_togather# select current prompt

def print_log_info(finished_number_problems, generated_hypothesis_count, evaluated_hypothesis_count, total_time_used_generation, total_time_used_evaluation):
    avg_time_evaluation_per_hypothesis = total_time_used_evaluation/evaluated_hypothesis_count if evaluated_hypothesis_count > 0 else '---NA---'
    avg_time_generation_per_hypothesis = total_time_used_generation/generated_hypothesis_count if generated_hypothesis_count > 0 else '---NA---'
    avg_hypo_generated_per_problem = round(total_hypothesis/(finished_number_problems),4) if finished_number_problems > 0 else '---NA---'
    total_time_used = datetime.now() - experiment_start_time
    avg_time_per_task = total_time_used/finished_number_problems if finished_number_problems > 0 else '---NA---'
    remaining_time_estimated = avg_time_per_task * (total_tasks-finished_number_problems) if avg_time_per_task!= '---NA---' else '---NA---'
    
    # TODO add current problem hypothesis count
    print(f'''generation_start_time: {experiment_start_time}, cur_time: {datetime.now()}, total_time_used:{total_time_used}, cur_task_progress: {finished_number_problems}/{total_tasks}, avg_time_per_task: {avg_time_per_task}, remaining_time: {remaining_time_estimated}, model_name: {model_name}, dataset_name: {dataset_name}, problem_idx(enumerate): {problem_idx}, cur_io_pair_count: {init_io_pair_count}, avg_time_evaluation_per_hypothesis:{avg_time_evaluation_per_hypothesis}, avg_time_generation_per_hypothesis:{avg_time_generation_per_hypothesis}, avg_hypo_generated_per_problem: {avg_hypo_generated_per_problem}, total_num_hypothesis_evaluated: {evaluated_hypothesis_count}, total_num_hypothesis_generated: {generated_hypothesis_count}, hypothesis_cur_problem: {hypothesis_generated_count_this_problem}, cur_problem_id: {cur_problem_id}, cur_group_idx:{group_index}''')    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g", "--group_idx",            
        type=str,                       
        required=True,                  
        help="choose the group of experiment that you want to run choose from 0,1,2,3,4",
    )
    parser.add_argument(
        "-m", "--model_name",           
        type=str,                       
        required=True,                  
        help=f"choose the model of experiment that you want to run choose from {model_mapping_dict}",
    )

    args = parser.parse_args()
    
    disable_progress_bar()
    # init generate client
    
    model_name = model_mapping_dict[args.model_name]
    if model_name in openai_models:
        client = OpenAI(api_key = open_ai_key)
        model_use_flag = 'use openai model'
    elif model_name in huggingface_models:
        client = generate_utils.HuggingFaceClient(model_name)
        model_use_flag = 'use huggingface model'
    else:
        assert False, f'''{model_name} not defined, please check the model you are using'''
    
    

    # safe usage parameters
    limit_hypothesis_generation = 25                    # at most generate x hypothesis
    limit_context_len = limit_hypothesis_generation*2+2 
    
    
    # experiment setting parameters
    selected_group_indexs = args.group_idx.split(',')
    selected_group_indexs = [int(i) for i in selected_group_indexs]
    
    unique_mapping_increase_threshold = 0.2
    print(f'current group_index : {selected_group_indexs}')
    
    for single_group_index in selected_group_indexs:
        assert single_group_index in group_indexs, f'''{group_index} not supported group_index'''

    
    finished_number_problems = 0
    generated_hypothesis_count = 0
    evaluated_hypothesis_count = 0
    total_hypothesis = 0 # update after each problem is finished
    total_time_used_generation = timedelta(0)
    total_time_used_evaluation = timedelta(0)

    total_tasks = 0
    cur_task = 0
    for group_index in selected_group_indexs:
        for dataset_name, init_io_pair_count in itertools.product(dataset_names, init_io_pair_counts):
            if dataset_name == 'acre' and init_io_pair_count == 1:
                continue
            io_pairs_path = f'../data/io_pairs/{dataset_name}/{dataset_name}_{init_io_pair_count}.json'
            with open(io_pairs_path, 'r') as f:
                n_all = len(json.load(f))
            n_selected = max(0, min(group_problem_count, n_all - group_problem_count*group_index))
            total_tasks += n_selected
    
    
    experiment_start_time = datetime.now()

    with tqdm(total=total_tasks, desc='TOTAL', unit='problem') as pbar:
        for group_index in selected_group_indexs:
            for dataset_name in dataset_names:
                for init_io_pair_count in init_io_pair_counts:
                    #get dataset io pairs
                    if dataset_name == 'acre' and init_io_pair_count == 1:
                        continue
                    io_pairs_path = f'../data/io_pairs/{dataset_name}/{dataset_name}_{init_io_pair_count}.json'
                    with open(io_pairs_path, 'r') as f:
                        loaded_io_pairs = json.load(f)
                    cur_problems = [{'problem_id':i[0],'io_pairs':i[1]} for i in loaded_io_pairs.items()]
                    cur_problems.sort(key=lambda x: x['problem_id'])
                    cur_problems = cur_problems[group_problem_count*group_index:group_problem_count*(group_index+1)]
            
                    # start generation for one problem
                    for problem_idx, problem in enumerate(cur_problems):
                        hypothesis_generated_count_this_problem = 0
                        continue_generation = True # by default continue generation
                        evaluation_history = []
                        # Todo assert type of io pairs should be 
                        cur_problem_id = problem['problem_id']
                        save_file_folder = os.path.join(save_root_dir, model_name, dataset_name, f'init_io_pair_count_{init_io_pair_count}')
                        save_file_path = os.path.join(save_file_folder, f'{cur_problem_id}.json')
                        if not os.path.exists(save_file_folder):
                            os.makedirs(save_file_folder)
                        cur_history_messages = None
                        # Check and load exist history file
                        if os.path.exists(save_file_path):
                            # exist a generated file, load that file and do evaluation with selected evaluation and early stopping criteria
                            with open(save_file_path, 'r') as f:
                                cur_history_messages = json.load(f)
                            cur_assistant_responses = [i for i in cur_history_messages if i['role'] == 'assistant']
                            assert len(cur_assistant_responses) > 0, f'''Saved file should have at least one assistant generated function'''
                            for assistant_response in cur_assistant_responses:
                                hypothesis_generated_count_this_problem += 1
                                print_log_info(finished_number_problems, generated_hypothesis_count, evaluated_hypothesis_count, total_time_used_generation, total_time_used_evaluation)
                                
                                start_time = datetime.now()
                                evaluate_result = generate_utils.evaluate_one_generation(problem['io_pairs'], assistant_response['content'], dataset_name, seed = seed, num_proc = num_proc)
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
                        while continue_generation:
                            print_log_info(finished_number_problems, generated_hypothesis_count, evaluated_hypothesis_count, total_time_used_generation, total_time_used_evaluation)
                            start_time = datetime.now()
                            cur_history_messages = generate_utils.generate_response_with_chat_history(model_name, 
                                                                                              problem['io_pairs'], 
                                                                                              prompt_dict, 
                                                                                              client,
                                                                                              history_messages = cur_history_messages,
                                                                                              model_use_flag = model_use_flag)
                            hypothesis_generated_count_this_problem += 1
                            end_time = datetime.now()
                            cur_time_used_generation = end_time - start_time
                            total_time_used_generation += cur_time_used_generation
                            generated_hypothesis_count += 1
        
                                
                            with open(save_file_path, 'w', encoding = 'utf-8') as f:
                                json.dump(cur_history_messages, f)
                            assert cur_history_messages[-1]['role'] == 'assistant', f'''returned message history should end with assistant's response, {cur_history_messages}'''
                            new_response = cur_history_messages[-1]['content']
                            print_log_info(finished_number_problems, generated_hypothesis_count, evaluated_hypothesis_count, total_time_used_generation, total_time_used_evaluation)
        
                            start_time = datetime.now()
                            evaluate_result = generate_utils.evaluate_one_generation(problem['io_pairs'], new_response, dataset_name, seed = seed, num_proc = num_proc)
                            evaluation_history.append(evaluate_result)
                            continue_generation, _ = generate_utils.if_continue_generation(evaluation_history, unique_mapping_increase_threshold = unique_mapping_increase_threshold, print_process = False)
                            end_time = datetime.now()
                            cur_time_used_evaluation = end_time - start_time
                            total_time_used_evaluation += cur_time_used_evaluation
                            evaluated_hypothesis_count += 1
                            
                            if len(cur_history_messages) >= limit_context_len:
                                continue_generation = False 
                        total_hypothesis += hypothesis_generated_count_this_problem
                        finished_number_problems += 1 
                        pbar.update(1)
                    
                            
                