
from typing import *
from tqdm import tqdm

import ast, textwrap, re

import io, tokenize
import utils
import json
import torch
import os 
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import copy
from peft import PeftModel


def generate_response_with_chat_history(model_name:str, 
                                        init_io_pairs, 
                                        prompt_dict, 
                                        client,
                                        history_messages: List[Dict[str,str]] = None,
                                        model_use_flag: str = None,
                                        short_context: bool = False,
                                        super_short_context: bool = False,
                                        )->List[Dict[str, str]]:

    assert model_use_flag is not None, f'''Should set the flag string to specify the model platform using'''
    assert not (short_context and super_short_context), f'''super_short_context and short_context can not be True at the same time'''

    string_pairs = generate_string_io_pair(init_io_pairs)
    if history_messages is None:
        if 'google' in model_name:
            history_messages = [{"role": "user", "content": prompt_dict['Instruction_prompt'] + '\n' + prompt_dict['Question_prompt_simple_mapping_init'].format(string_pairs,)}]
        else:
            history_messages = [{"role": "user", "content": prompt_dict['Instruction_prompt']},
                        {"role": "user", "content": prompt_dict['Question_prompt_simple_mapping_init'].format(string_pairs,)},]
        if super_short_context:
            new_message = {"role": "user", "content": "Please generate as much hypothesis as possible (one at a time), and make sure new generated hypothesis is based on a fundamentally different principle from any of your previous hypotheses."}
            history_messages.append(new_message)
    else:
        assert history_messages[-1]['role'] == 'assistant', f'''given history_messages: {history_messages} should ends with an assistant reponse'''
        generated_functions = [i['content'] for i in history_messages if i['role'] == 'assistant']
        if short_context:
            history_messages.append({
                "role": "user", "content": 'Invent a brand-new hypothesis based on a fundamentally different principle from any of your previous hypotheses, while still matching every given input-output pair. Use exactly the same output format as before.'
            })
        elif super_short_context:
            history_messages.append({
                "role": "user", "content": 'Another.'
            })
            
        else:
            history_messages.append({
                "role": "user", "content": prompt_dict['Question_prompt_simple_mapping_iterative'].format(len(generated_functions),string_pairs)
            })

    if model_use_flag == 'use openai model':
        response = client.chat.completions.create(
            model= model_name,
            messages=history_messages
            )
        generated_message = response.choices[0].message.content

    elif model_use_flag == 'use huggingface model':
        generated_message = client.generate(history_messages)
        # load a client for open source model
        
    else:
        assert False, f'model_use_flag: {model_use_flag} is not correct, please check the naming of the flag'

    history_messages.append({'role':'assistant', 'content':generated_message})
    return history_messages

def evaluate_one_generation(init_io_pairs:str, input_str:str, dataset_name:str, seed = 2025, num_proc = 8, eval_mode = 'default'):
    assert eval_mode in ['default','artificial', 'dataset'],f'''cur eval mode is not defined'''
    sample_space_gen_function_dict = {
        'list_function': utils.generate_sample_space_list_function,
        'mini_arc': utils.generate_sample_space_mini_arc,
        'arc_2025': utils.generate_sample_space_arc_2025,
        'acre': utils.generate_sample_space_acre
    }
    assert dataset_name in sample_space_gen_function_dict.keys(), f'''{dataset_name} is not a defined dataset'''
    # step1: parse the generated sentence into tuple
    description, code_str = parse_string_tuple(input_str)
    if code_str == 'tuple parse error':
        return 'tuple parse error'

    # step2: parse the generate code_str into real callable function
    if not isinstance(code_str, str):
        return 'syntac parse error'
    code_str = utils.preprocess_code(code_str)
    #callable_function = utils.parse_string_code_to_function(code_str)
    callable_function = utils.instrument_with_local_guard(code_str)
    if isinstance(callable_function, str):
        return 'syntac parse error'

    #do_safe_call = False
    # step3: test on init observation io pairs and check consistency
    failed_cases = {}
    init_io_pairs_count = len(init_io_pairs)
    consistant_case_count = 0 # init_io_pairs_count == len(failed_cases) + consistant_case_count
    for init_pair_index,init_pair in enumerate(init_io_pairs):
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
            failed_cases.update({json.dumps(init_pair): [json.dumps(test_output), 'N/A', str(e)]}) #io_pair: [gold_output, generated_output, err_msg]
            # print(e)
            # if isinstance(e, TimeoutError) or 'timed-out' in str(e):
            #     do_safe_call = True
        else:
            if test_output == model_output:
                consistant_case_count += 1
            else:
                try:
                    failed_cases.update({json.dumps(init_pair): [json.dumps(test_output), json.dumps(model_output),'output not match']})
                except Exception as e:
                    failed_cases.update({json.dumps(init_pair): [test_output, 'Error: json can not dump the failed output result', 'output not match']})
                
    #while_true_test = utils.detect_while_true_risk(code_str)
    #do_safe_call = (while_true_test or do_safe_call)
    # step4: check if model cheate
    is_cheat = utils.is_cheating(code_str, init_io_pairs)

    # step5: generate sample_space and generate all mappings
    sample_space = sample_space_gen_function_dict[dataset_name](seed = seed)
    sample_space_len = len(sample_space)

    mapped_results = utils.generate_all_mappings(sample_space, callable_function, num_proc = num_proc, do_safe_call = False)

    #print(mapped_results['sample_mapping_pair'][:200])
    #disable_progress_bar()        
    
    #print(sample_space[:10], set([json.dumps(i) for i in sample_space])[])
    # step6: gather needed data from mappings
    unique_mapping_pairs = [tuple(pair) for pair in mapped_results['sample_mapping_pair'] if json.loads(pair[1]) != 'N/A: cur input is not defined on this function']

    unique_mapping_set = set(unique_mapping_pairs)
    assert len(unique_mapping_pairs) == len(unique_mapping_set), f'''These two numbers should be the same but we get {len(unique_mapping_pairs)} and {len(unique_mapping_set)}'''
    undefined_count = len(mapped_results) - len(unique_mapping_pairs)

    evaluation_result = {
        'failed_cases':failed_cases,
        'consistant_case_count':consistant_case_count,
        'is_consistant':len(failed_cases) == 0,
        'is_cheat':is_cheat,
        'sample_space_len': sample_space_len,
        'unique_mapping_list': unique_mapping_pairs,
        'undefined_count': undefined_count,
        'code_str':code_str,
    }

    return evaluation_result

def if_continue_generation(evaluation_history:List[Dict[str,Any]],  # for dynamic evaluation
                           stop_criteria:str = 'default', 
                           unique_mapping_increase_threshold:float = 0.1, # each hypothesis should at least bring 10% new unique mappings to the size of sample space
                           print_process = False,
                          )->bool:
    # evaluation_history stores all the result of the generated hypothesis at every step including all the mappings etc
    '''evaluation_history shoule be a list with each following key failed_cases, consistant_case_count, is_cheat, sample_space_len, undefined_count'''
    supported_criteria = ['default']
    assert stop_criteria in supported_criteria, f'''{stop_criteria} is not a supported criteria'''
    if stop_criteria == 'default': 
        result_dict =    {'cur_hypothesis_count':0, # for print log only
                          'parse_error_count':0,
                          'no_consistancy_count':0,
                          'cheat_count':0,
                          'no_enough_mapping_count':0,
                          'unique_mapping_learned': [],
                          'is_cheat': [],
                         }
        bad_count = 0
        # bad generation flag = 0, generated message is not able to parse into functions 1, not consistant (have failed_cases) 2, is_cheat by hard coding 3, bring no more than unique_mapping_increase_threshold new mappings
        # stop when have at least 3 bad generation
        if len(evaluation_history) < 3:
            return True, None
        
        cur_unique_mappings = set()
        for idx, tem_evaluation_history in enumerate(evaluation_history):
            result_dict['cur_hypothesis_count'] += 1
            if isinstance(tem_evaluation_history, str):
                bad_count += 1
                result_dict['parse_error_count'] += 1    
                result_dict['unique_mapping_learned'].append(0)
                result_dict['is_cheat'].append(False)
                if print_process:
                    print(tem_evaluation_history)
                    print(result_dict)
                if bad_count >= 3:
                    return False, result_dict
                continue
            flg_1 = len(tem_evaluation_history['failed_cases']) > 0
            if flg_1:
                result_dict['no_consistancy_count'] += 1
                if print_process:
                    print('-----------------------------------------------------------------------------------------')
                    print('---------------------------cur hypothesis have failed init case--------------------------')
                    print('-----------------------------------------------------------------------------------------')
                    print(result_dict)
                    for fail_case_io_pair, fail_case_result  in tem_evaluation_history['failed_cases'].items():
                        assert len(fail_case_result) == 3, f'''fail_case_result should have exact 3 elements, {fail_case_result}'''
                        test_output, model_output, err_msg = fail_case_result
                        code_str = tem_evaluation_history['code_str']
                        print(f'''input: {fail_case_io_pair}, function_str: {code_str}''')
                        print(f'''expected_output: {test_output}''')
                        print(f'''actual_output: {model_output}''')
                        print(f'''err_msg: {err_msg}''')
                    print('-----------------------------------------------------------------------------------------')
                    print('---------------------------cur hypothesis have failed init case--------------------------')
                    print('-----------------------------------------------------------------------------------------')
                    
                
            flg_2 = tem_evaluation_history['is_cheat']
            if flg_2:
                result_dict['cheat_count'] += 1
                result_dict['is_cheat'].append(True)
                if print_process:
                    print('----------------------------------------------------------------------------------------')
                    print('---------------------------cur hypothesis potentially cheating--------------------------')
                    print('----------------------------------------------------------------------------------------')
                    code_str = tem_evaluation_history['code_str']
                    print(code_str)
                    print(result_dict)
                    print('----------------------------------------------------------------------------------------')
                    print('---------------------------cur hypothesis potentially cheating--------------------------')
                    print('----------------------------------------------------------------------------------------')
            else:
                result_dict['is_cheat'].append(False)
                    
            tem_old_all_unique_mappings = cur_unique_mappings

            cur_mapping_pairs = tem_evaluation_history['unique_mapping_list']
            
            # list → tuple，避免 JSON 再次 dumps 造成双重编码
            formatted_mapping_pairs = []
            for p in cur_mapping_pairs:
                if isinstance(p, list):
                    p = tuple(p)
                formatted_mapping_pairs.append(p)
            
            unique_mapping_set = set(formatted_mapping_pairs)  # 先建集合
            cur_unique_mappings = cur_unique_mappings.union(unique_mapping_set)
            
            increase = len(cur_unique_mappings) - len(tem_old_all_unique_mappings)

            result_dict['unique_mapping_learned'].append(increase)
            sample_space_size = tem_evaluation_history['sample_space_len']
            flg_3 = (increase/sample_space_size) < unique_mapping_increase_threshold # if new hypothesis can not provide new unique mapping it is a bad hypothesis
            if flg_3:
                result_dict['no_enough_mapping_count'] += 1
                if print_process:
                    print('-------------------------------------------------------------------------------------------------')
                    print('---------------------------cur hypothesis generate too few new mappings--------------------------')
                    print('-------------------------------------------------------------------------------------------------')
                    print(f'''sample_size: {sample_space_size}, new_mapping_provided: {increase}, increase_ratio: {round(increase/sample_space_size, 4)}, cur_threshold_ratio: {unique_mapping_increase_threshold}''')
                    print(result_dict)
                    print('-------------------------------------------------------------------------------------------------')
                    print('---------------------------cur hypothesis generate too few new mappings--------------------------')
                    print('-------------------------------------------------------------------------------------------------')
                    
            else:
                if print_process:
                    print(f'''sample_size: {sample_space_size}, new_mapping_provided: {increase}, increase_ratio: {round(increase/sample_space_size, 4)}, cur_threshold_ratio: {unique_mapping_increase_threshold}''')
                    

            bad_count += int(flg_1 or flg_2 or flg_3)
            if bad_count >= 3:
                return False, result_dict
    return True, result_dict


def generate_string_io_pair(pairs:List[Tuple]):
    '''
    given input and output parse it into prompt strings
    '''
    return_string = ""
    for index, pair in enumerate(pairs):
        assert len(pair) == 2, f'''Each element in the list should be a tuple or a list with exactly 2 elements one is input and the other is output, current pair{pair}'''
        #assert isinstance(pair[0], str), f'''expect the input of one pair is a list but get {type(pair[0])} instead'''
        #assert isinstance(pair[1], str), f'''expect the output of one pair is a list but get {type(pair[1])} instead'''
        return_string += f"Pair{index}: ({pair[0]}, {pair[1]})\n"
    return return_string


class HuggingFaceClient:
    """
    A custom client for interacting with open-source LLMs from Hugging Face.
    It checks for a local copy of the model and downloads it if not found.
    This version is updated to be compatible with models like tencent/Hunyuan.
    """
    def __init__(self, model_name: str, ckp_count:int = None ,auth_token: str = None):
        """
        Initializes the client, loading the specified model and tokenizer.
        If the model does not exist locally, it will be downloaded from the Hub.

        Args:
            model_name (str): The Hugging Face Hub ID of the model to load.
                              e.g., "tencent/Hunyuan-Di-Instruct"
            auth_token (str, optional): Hugging Face token. If None, it's read from the environment.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.lora_adaptor_name = None
        if 'lora' in self.model_name:# Currently using a lora fine tuned model
            split_name = model_name.split('_')
            print(f'Now using lora fine tuned model, base_model_name: {split_name[0]}')
            if len(split_name) == 2: # debug test only
                self.model_name = split_name[0]
                self.using_lora = True
                self.lora_adaptor_name = self.model_name.split('/')[-1] 
                self.lora_adaptor_path = 'your path here'# something like: f"root_path/{self.lora_adaptor_name}/checkpoints/final_adapter"  only change root path
                if ckp_count is not None:
                    self.lora_adaptor_path = 'your path here'# something like: f"root_path/{self.lora_adaptor_name}/checkpoints/checkpoint-{ckp_count}"  only change root path
            elif len(split_name) == 3:  # debug test only
                ckp_step = int(split_name[2]) 
                self.model_name = split_name[0]
                self.using_lora = True
                self.lora_adaptor_name = self.model_name.split('/')[-1] + f'_{ckp_step}' 
                self.lora_adaptor_path = 'your path here'# f"root_path/{self.lora_adaptor_name}/checkpoints/final_adapter"
                if ckp_count is not None:
                    self.lora_adaptor_path = 'your path here' #f"root_path/{self.lora_adaptor_name}/checkpoints/checkpoint-{ckp_count}"
            elif 'sft' in split_name and 'dpo' in split_name:  # sft->dpo models
                self.using_lora = True
                self.model_name = split_name[0]
                postfix = '_'.join(split_name[1:])
                self.lora_adaptor_name = self.model_name.split('/')[-1] + f'_{postfix}' 
                self.lora_adaptor_path = 'your path here'# f"root_path/{self.lora_adaptor_name}/checkpoints/final_adapter/sft"
                if ckp_count is not None:
                    self.lora_adaptor_path = 'your path here'# f"root_path/{self.lora_adaptor_name}/checkpoints/checkpoint-{ckp_count}/sft"
            elif not 'sft' in split_name and 'dpo' in split_name:
                print("most recent experiment should load with this!")
                self.using_lora = True
                self.model_name = split_name[0]
                postfix = '_'.join(split_name[1:])
                self.lora_adaptor_name = self.model_name.split('/')[-1] + f'_{postfix}' 
                self.lora_adaptor_path = 'your path here'# f"root_path/{self.lora_adaptor_name}/checkpoints/final_adapter"
                if ckp_count is not None:
                    self.lora_adaptor_path = 'your path here'# f"root_path/{self.lora_adaptor_name}/checkpoints/checkpoint-{ckp_count}"
                
            else:                     
                self.model_name = split_name[0]
                split_name.remove('lora')
                postfix = '_'.join(split_name[1:])
                self.using_lora = True
                self.lora_adaptor_name = self.model_name.split('/')[-1] + f'_{postfix}' 
                self.lora_adaptor_path = 'your path here'# f"root_path/{self.lora_adaptor_name}/checkpoints/final_adapter"
                
            
        else:
            self.using_lora = False

        self.token = auth_token
        if self.token is None:
            self.token = os.getenv("HUGGINGFACE_HUB_TOKEN")

        print(f"Initializing model: {self.model_name}...")
        try:
            root_dir = 'your path here'  # Your local models directory
            model_folder_name = self.model_name.split('/')[-1]
            model_path = os.path.join(root_dir, model_folder_name)
            
            print(f"Expected local path: {model_path}")

            if not os.path.isdir(model_path):
                print(f"Model '{model_name}' not found locally. Starting download...")
                try:
                    snapshot_download(
                        repo_id=self.model_name,
                        local_dir=model_path,
                        token=self.token
                    )
                    print("Model download successful.")
                except Exception as e:
                    print(f"snapshot_download failed for model {self.model_name}: {e}")
                    raise e
            else:
                print("Found model in local directory.")

            print(f"Loading model from verified local path: {model_path}...")
            self._load_model_from_local(model_path)
            self.model.eval()
            
            print(f"Model loaded successfully from {model_path}.")

        except Exception as e:
            print(f"An error occurred while loading model {self.model_name}: {e}")
            raise
    def _load_model_from_local(self, model_path): # load huggingface models
        import torch
        if 'gemma-2-9b-it' in model_path:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 128 
            torch.set_float32_matmul_precision('high')
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )    
        else:
            # regular model like llama
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )    
        if self.using_lora:
            assert self.lora_adaptor_name is not None, f'''using lora but lora adaptor path is not defined'''
            # TODO here change the model path
            print(f'Try loading lora adaptor from: {self.lora_adaptor_path}')
            model = PeftModel.from_pretrained(self.model, self.lora_adaptor_path)
            print("Merging lora weights")
            self.model = model.merge_and_unload()
            print("Merging complete!")

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 4096) -> str:
    
        if not self.model or not self.tokenizer:
            raise RuntimeError("HuggingFaceClient is not initialized properly.")
    
        import torch
        with torch.no_grad():
            try:
                prompt_string = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # gemma model have different specification
                if 'gemma-2-9b-it' in self.model_name:
                    max_len = 8192
                    if len(prompt_string) > max_len - 256:
                        prompt_string = prompt_string[len(prompt_string) - (max_len - 256):]
    
                inputs = self.tokenizer(prompt_string, return_tensors="pt").to(self.model.device)
    
                num_input_tokens = inputs.input_ids.shape[1]
    
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                )
                newly_generated_tokens = outputs[0][num_input_tokens:]
                text = self.tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
                return text
            except Exception as e:
                print(f"An error occurred during text generation with {self.model_name}: {e}")
                raise

def parse_string_tuple(string_response: str):
    try:
        # 1. Clean the string: remove outer parentheses and surrounding whitespace
        # .strip() handles leading/trailing whitespace; [1:-1] removes '(' and ')'
        content = string_response.strip()[1:-1].strip()

        # 2. Define a robust regex to match any form of Python string
        #    - (['"]{1,3}): Captures the opening quote(s), either ' or " repeated 1 or 3 times.
        #    - (.*?): Lazily captures the string content.
        #    - \1: Matches the same quote(s) captured at the beginning.
        #    - re.DOTALL: Allows '.' to match newlines.
        string_pattern = re.compile(r"""(['"]{1,3})(.*?)\1""", re.DOTALL)
        
        # 3. Find all matched strings from the tuple content
        #    This yields a list of string contents without the quotes.
        #    e.g., for `('a', '''b''')`, it will produce ['a', 'b']
        all_strings = [match.group(2) for match in string_pattern.finditer(content)]

        # 4. Check whether exactly two strings were found
        if len(all_strings) == 2:
            NL_description = all_strings[0]
            string_code = all_strings[1]
        else:
            # If not, the format is more complex or corrupted than expected.
            # Fall back to the original approach as a backup.
            result_tuple = ast.literal_eval(string_response)
            NL_description = result_tuple[0]
            string_code = result_tuple[1]

    except Exception as e:
        # If any step fails, execute the fallback logic
        # print(f"Error parsing string tuple: {e}")
        # print(f"Problematic string: {string_response[:200]}...")
        NL_description = string_response
        string_code = 'tuple parse error'
            
    return NL_description, string_code



def generate_hypotheses_togather(model_name, initial_io_pairs, prompt_dict, client, additional_iter = 5, print_response = True):
    # Init solution
    string_pairs = generate_string_io_pair(initial_io_pairs)
    history_messages = [{"role": "user", "content": prompt_dict['Instruction_prompt']},
                {"role": "user", "content": prompt_dict['Question_prompt_simple_mapping_init'].format(string_pairs,)}]
    generated_functions = []
    response = client.chat.completions.create(
        model= model_name,
        messages=history_messages
        )
    if print_response:
        print(response)
    
    # parse the result
    NL_description, string_code = parse_string_tuple(response.choices[0].message.content)
    #function = parse_string_code_to_function(string_code)
    # save the generated results
    generated_functions.append((NL_description,string_code))
    # save the reulst
    history_messages.append({
        "role":"assistant", "content": response.choices[0].message.content
    })
    
    for i in tqdm(range(additional_iter)):
        history_messages.append({
            "role": "user", "content": prompt_dict['Question_prompt_simple_mapping_iterative'].format(len(generated_functions),string_pairs)
        })
        response = client.chat.completions.create(
            model = model_name,
            messages = history_messages
            )
        if print_response:
            print(response)
        
        NL_description, string_code = parse_string_tuple(response.choices[0].message.content)
        #function = parse_string_code_to_function(string_code)
        generated_functions.append((NL_description, string_code))
        history_messages.append({
            "role":"assistant", "content": response.choices[0].message.content
        })
    return generated_functions
    
def generate_hypotheses_code_only(model_name, initial_io_pairs, prompt_dict, client, additional_iter = 5, print_response = True):
    # Init solution
    string_pairs = generate_string_io_pair(initial_io_pairs)
    history_messages = [{"role": "user", "content": prompt_dict['Instruction_prompt']},
                {"role": "user", "content": prompt_dict['Question_prompt_simple_mapping_init'].format(string_pairs,)}]
    generated_functions = []
    response = client.chat.completions.create(
        model= model_name,
        messages=history_messages
        )
    if print_response:
        print(response)
    
    # parse the result
    string_code = response.choices[0].message.content
    #function = parse_string_code_to_function(string_code)
    # save the generated results
    generated_functions.append(('Code only mode, no description provided.',string_code))
    # save the reulst
    history_messages.append({
        "role":"assistant", "content": response.choices[0].message.content
    })
    
    for i in tqdm(range(additional_iter)):
        history_messages.append({
            "role": "user", "content": prompt_dict['Question_prompt_simple_mapping_iterative'].format(len(generated_functions),string_pairs)
        })
        response = client.chat.completions.create(
            model = model_name,
            messages = history_messages
            )
        if print_response:
            print(response)
        string_code = response.choices[0].message.content
        #function = parse_string_code_to_function(string_code)
        
        generated_functions.append(('Code only mode, no description provided.', string_code))
        history_messages.append({
            "role":"assistant", "content": response.choices[0].message.content
        })
    return generated_functions

def generate_hypotheses_separate(model_name, initial_io_pairs, prompt_dict, client, additional_iter = 5, print_response = True):
    # Init solution
    string_pairs = generate_string_io_pair(initial_io_pairs)
    history_messages_description = [{"role": "user", "content": prompt_dict['Instruction_prompt_description']},
                {"role": "user", "content": prompt_dict['Question_prompt_simple_mapping_init_description'].format(string_pairs,)}]
    response_description = client.chat.completions.create(
                                                          model = model_name,
                                                          messages = history_messages_description
                                                         )
    def generate_code_with_hypothesis(hypothesis_string):
        history_messages_code = [{"role": "user", "content": prompt_dict['Instruction_prompt_code']},
                    {"role": "user", "content": prompt_dict['Question_prompt_simple_mapping_init_code'].format(string_pairs,hypothesis_string)}]
        response_code = client.chat.completions.create(
                                                        model = model_name,
                                                        messages = history_messages_code
                                                        )
        return response_code
    
    generated_functions = []
    response_code = generate_code_with_hypothesis(response_description.choices[0].message.content)
    if print_response:
        print(response_code)
        print(response_description)
    
    # parse the result
    NL_description = response_description.choices[0].message.content
    string_code = response_code.choices[0].message.content
    
    #function = parse_string_code_to_function(string_code)
    # save the generated results
    generated_functions.append((NL_description,string_code))
    # save the reulst
    history_messages_description.append({
        "role":"assistant", "content": response_description.choices[0].message.content
    })
    
    for i in tqdm(range(additional_iter)):
        history_messages_description.append({
            "role": "user", "content": prompt_dict['Question_prompt_simple_mapping_iterative_description'].format(len(generated_functions),string_pairs)
        })
        response_description = client.chat.completions.create(
            model = model_name,
            messages = history_messages_description
            )
        NL_description = response_description.choices[0].message.content
        response_code = generate_code_with_hypothesis(NL_description)
        if print_response:
            print(response_code)
            print(response_description)
        #function = parse_string_code_to_function(string_code)
        generated_functions.append((NL_description, string_code))
        history_messages_description.append({
            "role":"assistant", "content": response_description.choices[0].message.content
        })
    return generated_functions

