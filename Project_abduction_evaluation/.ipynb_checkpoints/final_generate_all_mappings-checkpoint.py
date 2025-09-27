import os
import generate_utils
import json
import utils
from tqdm.auto import tqdm
import argparse
from global_variables import *
from datetime import datetime

# generate all mappings for all hypothesis


seed = 2025 # for sample space generation do not change this
gen_save_root_dir = 'generate_log'
io_pairs_save_path_dir = '../data/io_pairs'

init_io_pair_count = 1
num_proc = 16

total_iters = len(dataset_names) * len(init_io_pair_counts) - 1
pbar = tqdm(total=total_iters, desc="Total")

# this function should evaluate one model one dataset one init_io_pair_count at a time
def generate_one_datafolder(model_name, 
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
            if cur_evaluation_mode == 'default':
                file_name = f'{hypothesis_idx}.json'
            else:
                assert False, f'''cur_evaluation_mode: {cur_evaluation_mode} is not supported'''
            cur_mapping_file_path = os.path.join(cur_mapping_dir, file_name)
            print(f'{os.path.join(model_name, dataset_name, f'init_io_{init_io_pair_count}', cur_problem_id,str(hypothesis_idx))}   Cur time: {datetime.now()}', end = '\r')
            if os.path.exists(cur_mapping_dir):
                pass
            else:
                os.makedirs(cur_mapping_dir)

            if os.path.exists(cur_mapping_file_path):
                # exist mapping file
                pass
            else:
                # do not exists mapping file need to generate and save
                evaluate_result = generate_utils.evaluate_one_generation(cur_init_io_pairs, one_hypothesis['content'], dataset_name, seed = seed, num_proc = num_proc)
                with open(cur_mapping_file_path, 'w') as f:
                    json.dump(evaluate_result, f)

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
    mapping_save_dir = None # './hypothesis_mappings' This file could be large
    assert mapping_save_dir is not None, 'please specify your path to save the mappings'
    

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
            tem_eval_result = generate_one_datafolder(model_name, 
                                                      dataset_name,
                                                      init_pair_count,
                                                      gen_save_root_dir,
                                                      print_process = False,
                                                      mapping_save_dir = mapping_save_dir)
            eval_results_dataset.update({init_pair_count:tem_eval_result})
            pbar.update(1)
            
    pbar.close()
        