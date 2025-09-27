import json
import os
from datetime import datetime
from tqdm import tqdm
from global_variables import * 
import random 
import copy
import itertools
import generate_utils
import utils
from pathlib import Path

def is_valid_hypothesis(hypothesis_data: dict):
    if 'is_consistant' not in hypothesis_data:
        return False
    assert 'is_consistant' in hypothesis_data and 'is_cheat' in hypothesis_data, f'''Should have this two keys'''
    
    is_consistant = hypothesis_data.get('is_consistant', False)
    is_cheat = hypothesis_data.get('is_cheat', False)
    return is_consistant and not is_cheat

def get_hypothesis_file_size_mb(hypothesis_meta: dict) -> float:
    """Constructs the file path for a hypothesis and returns its size in MB."""
    try:
        file_path = Path(MAPPING_FILE_ROOT) / \
                    hypothesis_meta['model_name'] / \
                    hypothesis_meta['dataset_name'] / \
                    f"init_io_{hypothesis_meta['init_io_pair_count']}" / \
                    hypothesis_meta['problem_id'] / \
                    f"{hypothesis_meta['hypothesis_idx']}.json"
        
        if not file_path.exists():
            return 0.0 # If file doesn't exist, treat it as size 0

        # Get size in bytes and convert to MB
        size_in_bytes = file_path.stat().st_size
        return size_in_bytes / (1024 * 1024)
        
    except (KeyError, FileNotFoundError):
        # Handle cases with missing keys or other errors gracefully
        return 0.0

prompt_dict = code_prompt_dict_0_3_togather
def get_hypothesis_log(hypothesis_meta_data):
    root_file_path = 'generate_log'
    init_pair_io_count = hypothesis_meta_data['init_io_pair_count']
    problem_id = hypothesis_meta_data['problem_id']
    file_path = os.path.join(root_file_path,
                             hypothesis_meta_data['model_name'], 
                             hypothesis_meta_data['dataset_name'],
                             f'init_io_pair_count_{init_pair_io_count}', 
                             f'{problem_id}.json')
    with open(file_path, 'r') as f:
        loaded_message = json.load(f)
    return [i['content'] for i in loaded_message if i['role'] == 'assistant'][hypothesis_meta_data['hypothesis_idx']]
    
def construct_context(context_data, dataset_name, io_count, problem_id):
    prompt_messages = []
    #load init io pairs
    io_pairs_save_path_dir = '../data/io_pairs'    
    io_pairs_path = f'{io_pairs_save_path_dir}/{dataset_name}/{dataset_name}_{io_count}.json'
    with open(io_pairs_path, 'r') as f:
        problem_io_pairs = json.load(f)
    cur_init_io_pairs = problem_io_pairs[problem_id]
    string_pairs = generate_utils.generate_string_io_pair(cur_init_io_pairs)
    context_message = [{"role": "user", "content": prompt_dict['Instruction_prompt']},
                       {"role": "user", "content": prompt_dict['Question_prompt_simple_mapping_init'].format(string_pairs,)}]
    
    generated_function_count = 0
    for hypothesis in context_data:
        assert hypothesis['dataset_name'] == dataset_name, f'''dataset name not match'''
        assert str(hypothesis['init_io_pair_count']) == str(io_count), f'''init_io_pair_count name not match'''
        assert hypothesis['problem_id'] == problem_id, f'''problem_id name not match'''
        generated_message = get_hypothesis_log(hypothesis)
        context_message.append({'role':'assistant', 'content':generated_message})
        generated_function_count += 1
        context_message.append({
                                "role": "user", "content": prompt_dict['Question_prompt_simple_mapping_iterative'].format(generated_function_count,string_pairs)
                               })
    return context_message

def load_hypothesis_mappings(single_context):
    mapping_save_dir = '/localdisk/kxh230002/hypothesis_mappings'
    io_count = single_context['init_io_pair_count']
    hypothesis_index = single_context['hypothesis_idx']
    cur_mapping_file = os.path.join(mapping_save_dir, 
                                    single_context['model_name'], 
                                    single_context['dataset_name'],
                                    f'init_io_{io_count}', 
                                    single_context['problem_id'], 
                                    f'{hypothesis_index}.json')
    with open(cur_mapping_file, 'r') as f:
        mapping_data = json.load(f)
    if isinstance(mapping_data, str):
        return []
    return [tuple(i) for i in mapping_data['unique_mapping_list']]

def compare_hypothesis(context_mappings, context_unique_set, answer_pair, print_process = False):

    answer_1_parse_error_score = (-1 if answer_pair[0]['parse_error'] is not None else 0)
    answer_2_parse_error_score = (-1 if answer_pair[1]['parse_error'] is not None else 0)
    if answer_1_parse_error_score == -1 and answer_2_parse_error_score == -1:
        return 'both parse error', None 
    elif answer_1_parse_error_score != answer_2_parse_error_score:
        winner = answer_pair[0] if answer_1_parse_error_score > answer_2_parse_error_score else answer_pair[1]
        return winner, 'parsing'  #  Return tuple with reason

    assert answer_pair[0]['init_io_pair_count'] == answer_pair[1]['init_io_pair_count'], f'''Selected answer pair do not have same io pair count'''
    assert answer_pair[0]['sample_space_len'] == answer_pair[1]['sample_space_len'], f'''Selected answer pair do not have same sample spave len'''
    
    # is cheating score
    answer_1_is_cheat_score = (-1 if answer_pair[0]['is_cheat'] else 0)
    answer_2_is_cheat_score = (-1 if answer_pair[1]['is_cheat'] else 0)
    if answer_1_is_cheat_score == -1 and answer_2_is_cheat_score == -1:
        return 'both cheating', None 
    elif answer_1_is_cheat_score != answer_2_is_cheat_score:
        winner = answer_pair[0] if answer_1_is_cheat_score > answer_2_is_cheat_score else answer_pair[1]
        return winner, 'cheating' #  Return tuple with reason
    
    # is consistant score (successfully explain all given io pairs)
    answer_1_consistant_score = (1 if answer_pair[0]['is_consistant'] else 0)
    answer_2_consistant_score = (1 if answer_pair[1]['is_consistant'] else 0)
    if answer_1_consistant_score == 0 and answer_2_consistant_score == 0:
        return 'both inconsistant', None 
    elif answer_1_consistant_score != answer_2_consistant_score:
        winner = answer_pair[0] if answer_1_consistant_score > answer_2_consistant_score else answer_pair[1]
        return winner, 'inconsistent' 



    answer_string_1_mapping = load_hypothesis_mappings(answer_pair[0])
    answer_string_2_mapping = load_hypothesis_mappings(answer_pair[1])
    
    # generalizablity score (defined domain size)
    answer_1_generalizablity_score = 1 - answer_pair[0]['undefined_count']/answer_pair[0]['sample_space_len']
    answer_2_generalizablity_score = 1 - answer_pair[1]['undefined_count']/answer_pair[1]['sample_space_len']

    # margin gamma diversity gain
    ori_set = copy.deepcopy(context_unique_set)
    new_set_ansewer_1 = ori_set.union(set(answer_string_1_mapping))
    answer_1_margin_gamma_gain_score = (len(new_set_ansewer_1) - len(ori_set))/answer_pair[0]['sample_space_len']
    
    new_set_answer_2 = ori_set.union(set(answer_string_2_mapping))
    answer_2_margin_gamma_gain_score = (len(new_set_answer_2) - len(ori_set))/answer_pair[1]['sample_space_len']

    # margin beta diversity gain
    if len(context_mappings) == 0:
        answer_1_avg_beta = 0
        answer_2_avg_beta = 0
    else:
        answer_1_hypothesis_list = copy.deepcopy(context_mappings)
        answer_1_hypothesis_list.append(answer_string_1_mapping)
        answer_1_avg_beta = utils.calculate_avg_beta_struct_simplified(answer_1_hypothesis_list)
        answer_2_hypothesis_list = copy.deepcopy(context_mappings)
        answer_2_hypothesis_list.append(answer_string_2_mapping)
        answer_2_avg_beta = utils.calculate_avg_beta_struct_simplified(answer_2_hypothesis_list)
        if print_process:
            print(f'-------------------------------------------------------------')
            print(answer_pair)
            print('generatlizablity:', answer_1_generalizablity_score, answer_2_generalizablity_score)
            print('beta:            ', answer_1_avg_beta, answer_2_avg_beta)
            print('gamma:           ', answer_1_margin_gamma_gain_score, answer_2_margin_gamma_gain_score)

    weight = [1,1,1]
    anwser_1_weighted_score = weight[0] * answer_1_generalizablity_score +  \
                              weight[1] * answer_1_margin_gamma_gain_score + \
                              weight[2] * answer_1_avg_beta
    anwser_2_weighted_score = weight[0] * answer_2_generalizablity_score +  \
                              weight[1] * answer_2_margin_gamma_gain_score + \
                              weight[2] * answer_2_avg_beta
                              
    if anwser_1_weighted_score == anwser_2_weighted_score:
        return 'equal good', None 

    # The final comparison based on the weighted score
    reason = 'fair comparsion'
    if anwser_1_weighted_score > anwser_2_weighted_score:
        if print_process:
            print('hypothesis 1 is better')
        return answer_pair[0], reason
    else:
        if print_process:
            print('hypothesis 2 is better')
        return answer_pair[1], reason




if __name__ == '__main__':
    print('-------------------------Step 1 preprocess all mappings data and log data--------------------------')
    print('-------------------------Step 1 preprocess all mappings data and log data--------------------------')
    print('-------------------------Step 1 preprocess all mappings data and log data--------------------------')
    print('-------------------------Step 1 preprocess all mappings data and log data--------------------------')
    mapping_save_dir = None #'your root path/hypothesis_mappings' # the path where you save hypothesis mappings
    final_labeled_data_save_dir = None # 'rl_data/final_labeled_data_v1_dpo.json'
    assert mapping_save_dir is not None and final_labeled_data_save_dir is not None, f'''specify your path'''
    log_save_dir = 'generate_log/'
    model_names = [i for i in model_mapping_dict.values() if 'lora' not in i] # train with non-tuned model
    all_hypothesis = []
    all_hypothesis_problem_group = {} # dataset_name, init_io_pair_count, problem_id
    
    
    # --- Step 1: Pre-calculate the total number of HYPOTHESES for the progress bar ---
    total_hypotheses_to_process = 0
    print("Calculating total number of hypotheses to process...")
    # This initial loop is just for the pre-calculation and can be slow if files are large or numerous
    for dataset_name in tqdm(dataset_names, desc="Scanning datasets"):
        for model_name in model_names:
            for init_io_pair_count in init_io_pair_counts:
                generate_log_folder = os.path.join(log_save_dir, model_name, dataset_name, f'init_io_pair_count_{init_io_pair_count}')
                if os.path.exists(generate_log_folder):
                    problem_files = [f for f in os.listdir(generate_log_folder) if not f.startswith('.') and f.endswith('.json')]
                    for problem_file_name in problem_files:
                        cur_problem_file = os.path.join(generate_log_folder, problem_file_name)
                        with open(cur_problem_file, 'r') as f:
                            log_data = json.load(f)
                            # Count items where role is 'assistant'
                            assistant_messages = [msg for msg in log_data if isinstance(msg, dict) and msg.get('role') == 'assistant']
                            total_hypotheses_to_process += len(assistant_messages)
    
    print(f"Found {total_hypotheses_to_process} total hypotheses.")
    
    # --- Step 2: Process files with a single, global progress bar for hypotheses ---
    # Initialize the global progress bar
    with tqdm(total=total_hypotheses_to_process, desc="Processing all hypotheses") as pbar:
        for dataset_name in dataset_names:
            if dataset_name not in all_hypothesis_problem_group:
                all_hypothesis_problem_group.update({dataset_name: {}})
            for model_name in model_names:
                for init_io_pair_count in init_io_pair_counts:
                    if dataset_name == 'acre' and init_io_pair_count == 1:
                        continue
                    if init_io_pair_count not in all_hypothesis_problem_group[dataset_name]:
                        all_hypothesis_problem_group[dataset_name].update({init_io_pair_count: {}})
                    generate_log_folder = os.path.join(log_save_dir, model_name, dataset_name, f'init_io_pair_count_{init_io_pair_count}')
    
                    problem_names = os.listdir(generate_log_folder)
                    problem_names = [i for i in problem_names if not i.startswith('.')]
                    
                    for problem_file_name in problem_names:
                        problem_id = problem_file_name.split('.')[0]
                        if problem_id not in all_hypothesis_problem_group[dataset_name][init_io_pair_count]:
                            all_hypothesis_problem_group[dataset_name][init_io_pair_count].update({problem_id:{'valid': [],
                                                                                                               'invalid': []}})
                        cur_problem_file = os.path.join(generate_log_folder, problem_file_name)
                        
                        with open(cur_problem_file, 'r') as f:
                            cur_problem_generate_log = json.load(f)
                        
                        all_current_hypothesis = [message for message in cur_problem_generate_log if message['role'] == 'assistant']
                        
                        for hypothesis_idx, hypothesis in enumerate(all_current_hypothesis):
                            pbar.update(1)
                            
                            hypothesis_mapping_file = os.path.join(mapping_save_dir, model_name, dataset_name, f'init_io_{init_io_pair_count}', problem_id, f'{hypothesis_idx}.json')
                            
    
                            with open(hypothesis_mapping_file, 'r') as f:
                                cur_hypothesis_mapping_information = json.load(f)
                            
                            cur_hypothesis_meta_data = {
                                'model_name': model_name,
                                'dataset_name': dataset_name,
                                'init_io_pair_count': init_io_pair_count,
                                'problem_id': problem_id,
                                'hypothesis_idx': hypothesis_idx,
                                'parse_error': None,
                            }
                            
                            if isinstance(cur_hypothesis_mapping_information, str):
                                cur_hypothesis_meta_data['parse_error'] = cur_hypothesis_mapping_information
                            else:
                                del cur_hypothesis_mapping_information['unique_mapping_list']
                                cur_hypothesis_meta_data.update(cur_hypothesis_mapping_information)
                            
                            all_hypothesis.append(cur_hypothesis_meta_data)
                            if is_valid_hypothesis(cur_hypothesis_meta_data):
                                all_hypothesis_problem_group[dataset_name][init_io_pair_count][problem_id]['valid'].append(cur_hypothesis_meta_data)
                            else:
                                all_hypothesis_problem_group[dataset_name][init_io_pair_count][problem_id]['invalid'].append(cur_hypothesis_meta_data)


    print('-------------------------Step 2 generate context and answer pairs----------------------------')
    print('-------------------------Step 2 generate context and answer pairs----------------------------')
    print('-------------------------Step 2 generate context and answer pairs----------------------------')
    print('-------------------------Step 2 generate context and answer pairs----------------------------')

    random_selector = random.Random(2025) 
    
    # answer pair sample parameters
    max_context_len = 2
    print_process = False
    
    keep_all_pair_context_len = [0]  # if context len = 0 or 1 retain all sampled pairs to keep data balance
    answer_pair_count_per_context = 15 # how many answer pairs at most each context
    # 40MB = 32016738 
    SIZE_CUTOFF_MB = 150                # drop hypothesis with huge mapping file
    
    
    structured_rl_data = {}            # intermediate samplede data
    structured_labeled_data = {}       # final preference data for DPO
    
    total_problems = sum(
        len(all_hypothesis_problem_group[ds][cnt])
        for ds in dataset_names
        for cnt in init_io_pair_counts
        if not (cnt == 1 and ds == 'acre')
    )
    
    
    
    MAPPING_FILE_ROOT = mapping_save_dir
    with tqdm(total=total_problems, desc="Building RL contexts") as pbar:
        for dataset_name in dataset_names:
            structured_rl_data[dataset_name] = {}
            for init_io_pair_count in init_io_pair_counts:
                if init_io_pair_count == 1 and dataset_name == 'acre':
                    continue
    
                structured_rl_data[dataset_name][init_io_pair_count] = {}
    
                for problem_id, hypothesis in all_hypothesis_problem_group[dataset_name][init_io_pair_count].items():
                    # 2. Filter hypotheses based on file size before using them  
                    original_valid_hypothesis = copy.deepcopy(hypothesis['valid'])
                    original_invalid_hypothesis = copy.deepcopy(hypothesis['invalid'])
    
                    # Filter the valid hypotheses
                    valid_hypothesis = [
                        h for h in original_valid_hypothesis 
                        if get_hypothesis_file_size_mb(h) <= SIZE_CUTOFF_MB
                    ]
                    
                    # Filter the invalid hypotheses
                    invalid_hypothesis = [
                        h for h in original_invalid_hypothesis
                        if get_hypothesis_file_size_mb(h) <= SIZE_CUTOFF_MB
                    ]
    
                    # Log the filtering action for monitoring
                    num_filtered = (len(original_valid_hypothesis) - len(valid_hypothesis)) + \
                                   (len(original_invalid_hypothesis) - len(invalid_hypothesis))
    
                    if num_filtered > 0:
                        # Calculate totals for the log message
                        total_before_filter = len(original_valid_hypothesis) + len(original_invalid_hypothesis)
                        total_after_filter = len(valid_hypothesis) + len(invalid_hypothesis)
                        
                        # Update the log message to be more comprehensive
                        pbar.write(
                            f"Dataset: {dataset_name}, IO_Pairs: {init_io_pair_count}, Problem: {problem_id} | "
                            f"Filtered: {num_filtered} (Before: {total_before_filter}, After: {total_after_filter})"
                        )
                    all_hypotheses = copy.deepcopy(valid_hypothesis) + copy.deepcopy(invalid_hypothesis)
        
                    context_answer_data = {}
        
                    for k in range(0, max_context_len + 1):
                        if len(valid_hypothesis) < k:
                            continue
                        if len(valid_hypothesis) >= 20:
                            sub_sample_valid_hypothesis = random_selector.sample(valid_hypothesis, 20)
                        else:
                            sub_sample_valid_hypothesis = valid_hypothesis
        
                        all_combos = list(itertools.combinations(sub_sample_valid_hypothesis, k))
        
                        for single_context in all_combos:
                            remaining_hypotheses = [i for i in all_hypotheses if i not in single_context]
                            all_possible_answer_pairs = list(itertools.combinations(remaining_hypotheses, 2))
    
                            if k in keep_all_pair_context_len: # here we want to add all possible mappings for context = 0 for better init generation and more balanced data
                                sampled_answer_pairs = all_possible_answer_pairs
                            else:
                                if len(all_possible_answer_pairs) > answer_pair_count_per_context:
                                    sampled_answer_pairs = random_selector.sample(all_possible_answer_pairs, answer_pair_count_per_context)
                                else:
                                    sampled_answer_pairs = all_possible_answer_pairs
                                
                            context_answer_data.update({json.dumps(single_context): sampled_answer_pairs})
    
                    structured_rl_data[dataset_name][init_io_pair_count][problem_id] = context_answer_data
                    
                    pbar.update(1) 
                    
    total_data_points = 0
    
    for dataset_name, io_counts in structured_rl_data.items():
        for io_count, problems in io_counts.items():
            for problem_id, context_data in problems.items():
                for answer_pairs_list in context_data.values():
                    total_data_points += len(answer_pairs_list)
    
    
    print(f"✅ pair data sampling finish")
    print(f" There are in total {total_data_points:,} init training pairs")


    print('--------------------Step 3 compare hypotheses pair and generate preference label-------------------')
    print('--------------------Step 3 compare hypotheses pair and generate preference label-------------------')
    print('--------------------Step 3 compare hypotheses pair and generate preference label-------------------')
    print('--------------------Step 3 compare hypotheses pair and generate preference label-------------------')

    print('Now start generating RL preference')
    from collections import defaultdict
    total_processed_pairs = 0
    total_discarded_pairs = 0
    reason_counts = defaultdict(int)
    progress_bar = tqdm(total=total_data_points, desc=f"generating lables, start time: {datetime.now()}")
    
    for dataset_name, io_counts in structured_rl_data.items():
        for io_count, problems in io_counts.items():
            for problem_id, context_data in problems.items():
                for context, answer_pairs_list in context_data.items():
                    context_data = json.loads(context) 
                    context = construct_context(context_data, dataset_name, io_count, problem_id)
                    context_mappings = []
                    context_len = len(context_data)
                    for single_context in context_data:
                        cur_context_mpping = load_hypothesis_mappings(single_context)
                        context_mappings.append(cur_context_mpping)
                    context_unique_set = {mapping for mapping_list in context_mappings for mapping in mapping_list}
                        
                    for idx, answer_pair in enumerate(answer_pairs_list):
                        total_processed_pairs += 1
                        good_hypothesis, reason = compare_hypothesis(context_mappings, context_unique_set, answer_pair, print_process)
                        progress_bar.update(1)
                        # Check if a winner was decided. If reason is None, it's a tie/error.
                        if reason is None:
                            total_discarded_pairs += 1
                            if print_process:
                                # 'good_hypothesis' now holds the string like 'both parse error'
                                print(idx, good_hypothesis, reason) 
                            continue # no preference provided this pair
                        reason_counts[reason] += 1
                        answer_pair = list(answer_pair)
                        answer_pair.remove(good_hypothesis)
                        assert len(answer_pair) == 1, f'''only one hypothesis should remain'''
                        bad_hypothesis = answer_pair[0]
                        good_answer_string = get_hypothesis_log(good_hypothesis)
                        bad_answer_string = get_hypothesis_log(bad_hypothesis)
                        labeled_data_tuple = (context, good_answer_string, bad_answer_string, reason)
                        structured_labeled_data.setdefault(dataset_name, {})\
                                           .setdefault(io_count, {})\
                                           .setdefault(problem_id, {})\
                                           .setdefault(context_len, [])\
                                           .append(labeled_data_tuple)
                        print(f'cur_time: {datetime.now()} | processed: {total_processed_pairs} | discarded: {total_discarded_pairs} | reasons: {json.dumps(reason_counts)}', end='\r')
                #break
            #break
        #break
    progress_bar.close()
    print("\n--- Final Statistics ---")
    print(f"Total pairs processed: {total_processed_pairs}")
    print(f"Total pairs discarded: {total_discarded_pairs}")
    print("Preference reasons breakdown:")
    for reason, count in reason_counts.items():
        print(f"  - {reason}: {count}")
    with open(final_labeled_data_save_dir, 'w') as f:
        json.dump(structured_labeled_data, f)
    
