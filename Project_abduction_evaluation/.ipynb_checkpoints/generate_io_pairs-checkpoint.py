# use this file to sample init observations
import utils
import os
import json

io_save_folder = '../data/io_pairs'
seed = 2025
init_io_pair_counts = [1,2,3,4]
final_question_count = 100

if __name__ == '__main__':
    assert not os.path.exists(io_save_folder), f'''Should delete the save path before regenerating''' 
    os.makedirs(io_save_folder)

    dataset_name_gen_function_dict = {
        'list_function': utils.load_io_pairs_list_function,
        'mini_arc': utils.load_io_pairs_mini_arc,
        'arc_2025': utils.load_io_pairs_arc_2025,
        'acre': utils.load_io_pairs_acre
    }

    for dataset_name in dataset_name_gen_function_dict.keys():
        for init_io_pair_number in init_io_pair_counts:
            if dataset_name == 'acre' and init_io_pair_number == 1:
                continue # acre dataset need to have at least two pairs to show two output states have 'on' and 'off' state
            folder_name = os.path.join(io_save_folder, dataset_name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            file_name = f'{dataset_name}_{init_io_pair_number}.json'
            save_file_path = os.path.join(folder_name, file_name)
            io_pair_data_dict = dataset_name_gen_function_dict[dataset_name](init_io_pair_number, seed = seed, final_question_count = final_question_count)
            with open(save_file_path, 'w') as f:
                json.dump(io_pair_data_dict, f)
            