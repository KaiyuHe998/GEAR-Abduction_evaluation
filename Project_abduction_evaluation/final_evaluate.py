import os
import generate_utils
import json
import utils
from tqdm.auto import tqdm
import argparse
from global_variables import *
from datetime import datetime

# evaluation_pipeline
# load_io_pairs


seed = 2025 # for sample space generation
unique_mapping_increase_threshold = 0.2
gen_save_root_dir = 'generate_log'
io_pairs_save_path_dir = '../data/io_pairs'

init_io_pair_count = 1
num_proc = 16

total_iters = len(dataset_names) * len(init_io_pair_counts) - 1
pbar = tqdm(total=total_iters, desc="Total")

# this function should evaluate one model one dataset one init_io_pair_count at a time
def evaluate_one_datafolder(model_name, 
                            dataset_name, 
                            init_io_pair_count, 
                            gen_save_root_dir,
                            num_proc = 16, 
                            unique_mapping_increase_threshold = 0.2,
                            io_pairs_save_path_dir = '../data/io_pairs',
                            print_process = True,
                            eval_sample_count = 5000,
                            mapping_save_dir = None,
                            cur_evaluation_mode = 'default',
                           ):
    
    io_pairs_path = f'{io_pairs_save_path_dir}/{dataset_name}/{dataset_name}_{init_io_pair_count}.json'
    with open(io_pairs_path, 'r') as f:
        cur_problems = json.load(f)
    
        
    save_file_folder = os.path.join(gen_save_root_dir, model_name, dataset_name, f'init_io_pair_count_{init_io_pair_count}')
    exist_hypothesis_gen_log_file_names = os.listdir(save_file_folder)
    exist_hypothesis_gen_log_file_names = [i for i in exist_hypothesis_gen_log_file_names if not i.startswith('.')]
    evaluation_results = []
    for save_file_name in tqdm(exist_hypothesis_gen_log_file_names):
        evaluation_history = []
        # load history message
        history_log_path = os.path.join(save_file_folder, save_file_name)
        with open(history_log_path, 'r') as f:
            history_messages = json.load(f)
        
        cur_problem_id = save_file_name.split('.')[0]
        assert cur_problem_id in cur_problems, f'''model_name: {model_name}, dataset_name:{dataset_name}, init_io_pair_count:{init_io_pair_count}, cur_problem_id:{cur_problem_id} do not have corresponding io_pairs'''
        cur_init_io_pairs = cur_problems[cur_problem_id]
        #print(f'''model_name: {model_name}, dataset_name:{dataset_name}, init_io_pair_count:{init_io_pair_count}, cur_problem_id:{cur_problem_id} do not have corresponding io_pairs''')
    
        generated_hypothesis = [i for i in history_messages if i['role'] == 'assistant']
        assert len(generated_hypothesis) > 0, f'''model_name: {model_name}, dataset_name:{dataset_name}, init_io_pair_count:{init_io_pair_count}, cur_problem_id:{cur_problem_id} do not have generated hypothesis'''
        total_num_hypothesis_generated = 0
        for hypothesis_idx, one_hypothesis in enumerate(generated_hypothesis):
            # first check if current mapping result exist, if do exists then load the existing one
            # if not generate all mappings and save the mapping results
            cur_mapping_dir = os.path.join(mapping_save_dir, model_name, dataset_name, f'init_io_{init_io_pair_count}', cur_problem_id)
            cur_mapping_file_path = os.path.join(cur_mapping_dir, f'{hypothesis_idx}.json')
            print(f'{os.path.join(model_name, dataset_name, f'init_io_{init_io_pair_count}', cur_problem_id,str(hypothesis_idx))}   Cur time: {datetime.now()}', end = '\r')
            if os.path.exists(cur_mapping_dir):
                pass
            else:
                os.makedirs(cur_mapping_dir)

            if os.path.exists(cur_mapping_file_path):
                # exist mapping file
                with open(cur_mapping_file_path, 'r') as f:
                    evaluate_result = json.load(f)
                if isinstance(evaluate_result, str):
                    pass
                else:
                    evaluate_result['unique_mapping_list'] = [tuple(pair) for pair in evaluate_result['unique_mapping_list']]
            else:
                # do not exists mapping file need to generate and save before call generation
                assert False, f'''do not exists mapping file: {cur_mapping_file_path}, please call final_generate_all_mappings first and store all mappings'''
                #evaluate_result = generate_utils.evaluate_one_generation(cur_init_io_pairs, one_hypothesis['content'], dataset_name, seed = seed, num_proc = num_proc)
                with open(cur_mapping_file_path, 'w') as f:
                    json.dump(evaluate_result, f)
            evaluation_history.append(evaluate_result)
            continue_generation, dynamic_evaluation = generate_utils.if_continue_generation(evaluation_history, unique_mapping_increase_threshold = unique_mapping_increase_threshold, print_process = print_process)
            total_num_hypothesis_generated += 1
            if not continue_generation:
                if hypothesis_idx < len(generated_hypothesis)-1:
                    # print(current problem have more generated hypothesis, but due to current stricter evaluation policy early stopped)
                    break
        assert dynamic_evaluation is not None, f'''Current evaluation have less than 3 hypothesis evaluated'''
        if continue_generation:
            pass
            #print(f'Under current evaluation and early stopping cretia, current problem can still generate new hypothesis: model_name: {model_name}, dataset_name:{dataset_name}, init_io_pair_count:{init_io_pair_count}, cur_problem_id:{cur_problem_id}')
        else:
            pass
        # finish one problem now return necessary information for visualization and analysis
        # evaluation_history should end here and all necessary information from unique_mappings should be extracted
        # TODO: record and calculate jaccard index in different settings
        # all count values
        #total_num_hypothesis_generated counted previously
        all_mapping_lists = [] # all mappings for all hypothesis generated 
        

        # valid hypothesis values
        total_num_valid_hypothesis_generated = 0
        mapping_lists_valid = [] # filter out cheat, parse error, and failed init_obv cases
        valid_hypothesis_index = []
        
        assert len(evaluation_history) == len(dynamic_evaluation['unique_mapping_learned']), f'''unique_mapping learned from dynamic evaluation should match the len of history evaluations, here we get {len(evaluation_history)} and {len(dynamic_evaluation['unique_mapping_learned'])}'''
        assert len(evaluation_history) == len(dynamic_evaluation['is_cheat']), f'''is_cheat learned from dynamic evaluation should match the len of history evaluations here we get {len(evaluation_history)} and {len(dynamic_evaluation['is_cheat'])}'''
        def is_valid(eval_result):
            if isinstance(eval_result, str):
                return False
            return eval_result.get('is_consistant', False) and not eval_result.get('is_cheat', False)
            
        for hypothesis_idx ,evaluate_result in enumerate(evaluation_history):
            if isinstance(evaluate_result, str):
                # parse error
                all_mapping_lists.append([])
                pass
            else:
                all_mapping_lists.append(evaluate_result['unique_mapping_list'])
                if is_valid(evaluate_result):
                    total_num_valid_hypothesis_generated += 1
                    mapping_lists_valid.append(evaluate_result['unique_mapping_list'])
                    valid_hypothesis_index.append(hypothesis_idx)


        # for all hypothesis statistic        
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
            assert False, f'''Should at least have 3 hypothesis generated for each problem, problem_id: {cur_problem_id}, io_count: {init_io_pair_count}, dataset_name:{dataset_name}, model_name: {model_name}, '''
        elif len(all_mapping_lists) == 1:
            avg_alpha_diversity = utils.calculate_avg_alpha_diversity(all_mapping_lists)
            avg_alpha_diversity_q_0 = avg_alpha_diversity['q0_richness']
            avg_alpha_diversity_q_1 = avg_alpha_diversity['q1_shannon']
            avg_alpha_diversity_q_2 = avg_alpha_diversity['q2_simpson']
            
            gamma_diversity_dist = utils.calculate_gamma_diversity_dist(all_mapping_lists)
            gamma_diversity_q_0_dist = gamma_diversity_dist['q0_richness']
            gamma_diversity_q_1_dist = gamma_diversity_dist['q1_shannon']
            gamma_diversity_q_2_dist = gamma_diversity_dist['q2_simpson']
            
            gamma_diversity_struct = utils.calculate_gamma_diversity_struct(all_mapping_lists)
            gamma_diversity_q_0_struct = gamma_diversity_struct['q0_richness']
            gamma_diversity_q_1_struct = gamma_diversity_struct['q1_shannon']
            gamma_diversity_q_2_struct = gamma_diversity_struct['q2_simpson']
        else:
            avg_alpha_diversity = utils.calculate_avg_alpha_diversity(all_mapping_lists)
            avg_alpha_diversity_q_0 = avg_alpha_diversity['q0_richness']
            avg_alpha_diversity_q_1 = avg_alpha_diversity['q1_shannon']
            avg_alpha_diversity_q_2 = avg_alpha_diversity['q2_simpson']

            avg_beta_diversity_dist = utils.calculate_avg_beta_dist(all_mapping_lists)
            avg_beta_diversity_struct = utils.calculate_avg_beta_struct(all_mapping_lists)

            gamma_diversity_dist = utils.calculate_gamma_diversity_dist(all_mapping_lists)
            gamma_diversity_q_0_dist = gamma_diversity_dist['q0_richness']
            gamma_diversity_q_1_dist = gamma_diversity_dist['q1_shannon']
            gamma_diversity_q_2_dist = gamma_diversity_dist['q2_simpson']
            
            gamma_diversity_struct = utils.calculate_gamma_diversity_struct(all_mapping_lists)
            gamma_diversity_q_0_struct = gamma_diversity_struct['q0_richness']
            gamma_diversity_q_1_struct = gamma_diversity_struct['q1_shannon']
            gamma_diversity_q_2_struct = gamma_diversity_struct['q2_simpson']

        # for valid hypothesis statistic
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
            avg_alpha_diversity_valid = utils.calculate_avg_alpha_diversity(mapping_lists_valid)
            avg_alpha_diversity_q_0_valid = avg_alpha_diversity_valid['q0_richness']
            avg_alpha_diversity_q_1_valid = avg_alpha_diversity_valid['q1_shannon']
            avg_alpha_diversity_q_2_valid = avg_alpha_diversity_valid['q2_simpson']
            
            gamma_diversity_valid_dist = utils.calculate_gamma_diversity_dist(mapping_lists_valid)
            gamma_diversity_q_0_valid_dist = gamma_diversity_valid_dist['q0_richness']
            gamma_diversity_q_1_valid_dist = gamma_diversity_valid_dist['q1_shannon']
            gamma_diversity_q_2_valid_dist = gamma_diversity_valid_dist['q2_simpson']
            
            gamma_diversity_valid_struct = utils.calculate_gamma_diversity_struct(mapping_lists_valid)
            gamma_diversity_q_0_valid_struct = gamma_diversity_valid_struct['q0_richness']
            gamma_diversity_q_1_valid_struct = gamma_diversity_valid_struct['q1_shannon']
            gamma_diversity_q_2_valid_struct = gamma_diversity_valid_struct['q2_simpson']
        else:
            avg_alpha_diversity_valid = utils.calculate_avg_alpha_diversity(mapping_lists_valid)
            avg_alpha_diversity_q_0_valid = avg_alpha_diversity_valid['q0_richness']
            avg_alpha_diversity_q_1_valid = avg_alpha_diversity_valid['q1_shannon']
            avg_alpha_diversity_q_2_valid = avg_alpha_diversity_valid['q2_simpson']
            
            avg_beta_diversity_dist_valid = utils.calculate_avg_beta_dist(mapping_lists_valid)
            avg_beta_diversity_struct_valid = utils.calculate_avg_beta_struct(mapping_lists_valid)
            
            gamma_diversity_valid_dist = utils.calculate_gamma_diversity_dist(mapping_lists_valid)
            gamma_diversity_q_0_valid_dist = gamma_diversity_valid_dist['q0_richness']
            gamma_diversity_q_1_valid_dist = gamma_diversity_valid_dist['q1_shannon']
            gamma_diversity_q_2_valid_dist = gamma_diversity_valid_dist['q2_simpson']
            
            gamma_diversity_valid_struct = utils.calculate_gamma_diversity_struct(mapping_lists_valid)
            gamma_diversity_q_0_valid_struct = gamma_diversity_valid_struct['q0_richness']
            gamma_diversity_q_1_valid_struct = gamma_diversity_valid_struct['q1_shannon']
            gamma_diversity_q_2_valid_struct = gamma_diversity_valid_struct['q2_simpson']
            
        undefined_count_valid = []
        sample_space_size_valid = []
        tem_unique_mapping_set = set()
        unique_mapping_learned_valid_io = []
        
        for i in evaluation_history:
            if isinstance(i,str):
                pass
            else:
                if is_valid(i):
                    cur_unique_mapping_set = set(i['unique_mapping_list'])
                    tem_tem_unique_mapping_set = tem_unique_mapping_set.union(cur_unique_mapping_set)
                    increase = len(tem_tem_unique_mapping_set) - len(tem_unique_mapping_set)
                    tem_unique_mapping_set = tem_tem_unique_mapping_set
                    unique_mapping_learned_valid_io.append(increase)
                    undefined_count_valid.append(i['undefined_count'])
                    sample_space_size_valid.append(i['sample_space_len'])
                    
                
        evaluation_result = {
                             'total_num_hypothesis_generated': total_num_hypothesis_generated,
                             'total_num_valid_hypothesis_generated': total_num_valid_hypothesis_generated,
                             'unique_mapping_learned': dynamic_evaluation['unique_mapping_learned'], # unique mapping added each generated hypothesis
                             'unique_mapping_learned_valid_io':unique_mapping_learned_valid_io,
                             'avg_alpha_diversity_q_0'   : avg_alpha_diversity_q_0,
                             'avg_alpha_diversity_q_1'   : avg_alpha_diversity_q_1,
                             'avg_alpha_diversity_q_2'   : avg_alpha_diversity_q_2,
                             'avg_beta_diversity_dist'   : avg_beta_diversity_dist,
                             'avg_beta_diversity_struct' : avg_beta_diversity_struct,
                             'gamma_diversity_q_0_dist'       : gamma_diversity_q_0_dist,
                             'gamma_diversity_q_1_dist'       : gamma_diversity_q_1_dist,
                             'gamma_diversity_q_2_dist'       : gamma_diversity_q_2_dist,
                             'gamma_diversity_q_0_struct'       : gamma_diversity_q_0_struct,
                             'gamma_diversity_q_1_struct'       : gamma_diversity_q_1_struct,
                             'gamma_diversity_q_2_struct'       : gamma_diversity_q_2_struct,
            
                             'avg_alpha_diversity_q_0_valid'   : avg_alpha_diversity_q_0_valid,
                             'avg_alpha_diversity_q_1_valid'   : avg_alpha_diversity_q_1_valid,
                             'avg_alpha_diversity_q_2_valid'   : avg_alpha_diversity_q_2_valid,
                             'avg_beta_diversity_dist_valid'   : avg_beta_diversity_dist_valid,
                             'avg_beta_diversity_struct_valid' : avg_beta_diversity_struct_valid,
                             'gamma_diversity_q_0_valid_dist'       : gamma_diversity_q_0_valid_dist,
                             'gamma_diversity_q_1_valid_dist'       : gamma_diversity_q_1_valid_dist,
                             'gamma_diversity_q_2_valid_dist'       : gamma_diversity_q_2_valid_dist,
                             'gamma_diversity_q_0_valid_struct'       : gamma_diversity_q_0_valid_struct,
                             'gamma_diversity_q_1_valid_struct'       : gamma_diversity_q_1_valid_struct,
                             'gamma_diversity_q_2_valid_struct'       : gamma_diversity_q_2_valid_struct,
            
                             'undefined_count': [i['undefined_count'] if not isinstance(i,str) else i for i in evaluation_history],
                             'undefined_count_valid': undefined_count_valid,
                             'sample_space_size': [i['sample_space_len'] if not isinstance(i,str) else i for i in evaluation_history],
                             'sample_space_size_valid': sample_space_size_valid,
                             'input_io_pair_count': init_io_pair_count,
                             'cur_problem_id': cur_problem_id,
                             'all_syntac_parse_error': dynamic_evaluation['parse_error_count'],
                             'all_consistancy_error':[len(i['failed_cases']) if not isinstance(i,str) else i for i in evaluation_history], 
                             'can_still_generate':continue_generation,
                            }
        evaluation_results.append(evaluation_result)
        #break # test use
    return evaluation_results

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_name",           
        type=str,                       
        required=True,                  
        help=f"choose the model of experiment that you want to run choose from {model_mapping_dict}",
    )
    args = parser.parse_args()
    model_name = model_mapping_dict[args.model_name]
    eval_root_save_dir = 'eval_results'
    mapping_save_dir = 'your mapping save root path' # something like'./hypothesis_mappings'
    assert mapping_save_dir is None, f'''please specify your mapping_save_dir, it should be the same in final_generate_all_mappings.py'''
    if model_name in openai_models:
        model_use_flag = 'use openai model'
        eval_save_file_name = f'final_eval_result_save_{model_name}.json'
    elif model_name in huggingface_models:
        model_use_flag = 'use huggingface model'
        model_name_list = model_name.split('/')
        assert len(model_name_list) == 2, f'''model name from huggingface should be corp/model_name'''
        eval_save_file_name = f'final_eval_result_save_{model_name_list[1]}.json'
    else:
        assert False, f'''{model_name} not defined, please check the model you are using'''
    eval_save_file_name = os.path.join(eval_root_save_dir,eval_save_file_name)
    eval_results = {}

    if os.path.exists(eval_save_file_name):
        print(f'''result file:{eval_save_file_name} already exists, pass''')
    else:
        path_dir = eval_save_file_name.split('/')[:-1]
        path_dir = '/'.join(path_dir)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        # cur_count = 0
        for dataset_name in dataset_names:
            eval_results_dataset = {}
            for init_pair_count in init_io_pair_counts:
                if dataset_name == 'acre' and init_pair_count == 1:
                    continue
                # cur_count += 1
                # if cur_count < 9:
                #     pbar.update(1)
                #     continue
                tem_eval_result = evaluate_one_datafolder(model_name, 
                                                          dataset_name,
                                                          init_pair_count,
                                                          gen_save_root_dir,
                                                          print_process = False,
                                                          mapping_save_dir = mapping_save_dir)
                eval_results_dataset.update({init_pair_count:tem_eval_result})
                pbar.update(1)
                
            eval_results.update({dataset_name:eval_results_dataset})
    
        with open(eval_save_file_name, 'w') as f:
            json.dump(eval_results,f)
        pbar.close()
            