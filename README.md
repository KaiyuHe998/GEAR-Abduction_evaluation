# ⚙️GEAR: General Evaluation for Abductive Reasoning

This repository accompanies our paper **GEAR** (General Evaluation for Abductive Reasoning).  
GEAR provides a **fully automated, label-free, and scalable framework** to evaluate abductive reasoning in large language models.  
It measures hypothesis sets along three classical dimensions:

- **Consistency** – hypotheses must correctly explain given observations  
- **Generalizability** – consistent hypotheses should make testable predictions on unseen inputs  
- **Diversity** – hypothesis sets should cover distinct perspectives and non-redundant patterns  

We benchmarked 9 LLMs on 4 popular datasets (MINI-ARC, ACRE, LIST FUNCTIONS, ARC-2025), generating over 50k hypotheses.
We also propose a **momentum-based curriculum training strategy** that improves all three objectives without gold-label supervision

![Figure1](figs/Figure1.png)

# $\textcolor{purple}{\text{GEAR evaluation on 9 LLMs}}$

### Step 1, Generate initial observation(io) pairs (See in paper section 4 Sampling initial observations.)
- Call `python generate_io_pairs.py` to sample the initial observations to seed abduction.
- We have random sampled io paris form 4 dataset saved in ../data/io_pairs
- However if you want to generate new io_pairs with your random seed you can use function `utils.load_io_pairs_{dataset_name}`


```python
import utils
for n in range(1,5): # random io count from 1~4
    init_io_count_list_function = utils.load_io_pairs_list_function(n, seed = 42)
list(init_io_count_list_function.items())[:3]
```




    [(163,
      [('[27, 33, 10]', '[11, 13, 7]'),
       ('[1, 6, 83, 99, 41, 30]', '[5, 6, 25, 29, 15, 12]'),
       ('[14, 72, 2, 89, 97, 7, 6]', '[8, 23, 5, 27, 29, 6, 6]'),
       ('[52, 65, 67, 8, 54, 85]', '[18, 21, 21, 7, 18, 26]')]),
     (28,
      [('[3, 9, 7, 5, 4, 0, 2, 6]', '[7, 5, 4, 0, 2, 6]'),
       ('[0, 1, 9, 4, 2, 3, 5]', '[9, 4, 2, 3, 5]'),
       ('[4, 6, 0, 8, 9, 7, 5, 3]', '[0, 8, 9, 7, 5, 3]'),
       ('[8, 6, 2]', '[2]')]),
     (6,
      [('[8, 1, 4, 9]', '[8, 1]'),
       ('[0, 2, 7, 4, 5, 9, 3, 1, 8]', '[0, 2]'),
       ('[6, 8, 3, 2, 9, 4, 5, 1]', '[6, 8]'),
       ('[7, 4, 3, 9, 2, 8, 5, 0, 1, 6]', '[7, 4]')])]



### step 2 generate hypothesis
- For generating hypothesis with 'o1' use the following command. (You can define new models and their name in  `global_variables.py` and **make sure to provide your api key and huggingface key**)
    - `python final_generate_hypothesis.py -m o1, -g 0,1,2,3,4` (Change the model name based on your definition in `global_variables.py`)
    - It saves while generating, if it stops, recall the same command.

### Step 3 generate prediction space
- After step 2, you need to generate corresponding prediction space for each generated hypothesis.
- Use the following command to generate prediction space and save them.
    - `python final_generate_all_mappings.py -m o1` 

### Step 4 evaluate
- After all prediction space is generated you can call the following to evaluate
    - `python final_evaluate.py -m o1`

### Downloadable Content

We provide generated **hypotheses** and **preference data** used in our paper, so you can skip earlier steps if you don't want to generate from scratch.

1. Download the archive from **[Google Drive](https://drive.google.com/file/d/126DPVwlWdnPzNT2zQFJTtsUk3KjQwuk-/view?usp=sharing)**.
2. **Extract** the archive; you will see two folders: `generate_log/` and `training_data/`.
3. **Move** both folders into the project root (e.g., `Project_abduction_evaluation/`), so the structure looks like:
- Project_abduction_evaluation/
    - generate_log/
    - training_data/
4. Proceed directly to **Step 3**.

# $\textcolor{purple}{\text{Preference data generation}}$
- Both RL training and similation study rely on the pre-generated preference data.
- Call `python final_rl_data_gen.py` to generate preference data based on all LLM hypothesis you've generated.

# $\textcolor{purple}{\text{Simulation study}}$

### Simulation Study 1 (See in paper section 5.2)
- `python final_simulation_study_defeasible.py -tc 1 -tr 5` (tc: how many test cases you want to use, tr: how many round do you willing to run this experiment)
- Since MINI-ARC and ARC-2025 do not provide enough observations in our experiment $tc \leq 4$

### Simulation Study 2 (See in paper section 5.3)
- `python final_simulation_study_general.py -tc 1 -tr 5`

# $\textcolor{purple}{\text{Train, validation, held out split}}$
- We provide the train/validation/held-out split in `training_data`. You may also use your own split. The format is simple: for each dataset, list the problem IDs assigned to train, validation, and held-out sets.


```python
import json
with open('training_data/dpo_train_ids.json', 'r') as f:
    train_split = json.load(f)
print(train_split['acre'])
# shown is the train ids for ACRE dataset
```

    ['0', '12', '13', '14', '17', '18', '2', '21', '22', '27', '3', '32', '38', '39', '4', '40', '41', '42', '44', '49', '5', '51', '52', '53', '59', '6', '81', '83', '84', '85', '87', '89', '93', '95', '98', '69', '72', '74', '77', '80']


# $\textcolor{purple}{\text{Reinforcement Learning and Evaluation}}$

### Step 1: Model training:
- Call ```python final_dpo_default_curriculum -m llama-3.1-8b -g 0 -p example_train -t scratch``` where
    -  -m: is the model name you provided in the global_variables.py
    -  -g: is the gpu you want to use 0,2,3 means you want to use GPU 0, 2 and 3 (in our experiment we use single A100 80G GPU)
    -  -p: is the postfix of the save_dir of the check point
    -  -t: don't need to change this.

### Step2: Generate hypothesis on held out problems and generate prediction space
- Call `python final_generate_hypothesis_auto_save_mappings_t3.py -m model_name -g 0 -r 5` where
-  -m is the model name you specified in global_variables.py
-  -g is the gpu you want to use for generation
-  -r stands for how many round do you want to run the evaluation
-  The scrip saves while running, you can rerun the same command if it stopps.

### Step3: Evaluate
- Call `python final_evaluate_t3 -m model_name -r 5` where
-  -m is the model name you specified in global_variables.py
-  -r stands for how many round of hypothesis and prediction space you previously generated

# $\textcolor{purple}{\text{Ablation study}}$

- Call `python final_dpo_default_simple_single_preference.py -m model_name -g 0 -p example_train -t scratch -pr parsing` where
    - -m: is the base model you want to train
    - -g: is the gpu you want to use
    - -p: is the postfix of the save_dir
    - -t: don't need to change this.
    - -pr: is the single preference type you want to train
- Then same to previous experiment run `final_generate_hypothesis_auto_save_mappings_t3.py` and `python final_evaluate_t3` for evaluation

# $\textcolor{purple}{\text{Final Results}}$


```python
import re
import json
import os
import utils
import copy
import numpy as np
import ast
from global_variables import *
import example_utils
from collections import OrderedDict
import generate_utils
import pandas as pd
with open('training_data/held_out_ids_t3.json', 'r') as f:
    t3_held_out = json.load(f)

held_out_problem_id = t3_held_out
datasets = {}
for dataset_name in held_out_problem_id.keys():
    if dataset_name == 'arc_2025':
        cur_dataset = utils.load_arc_2025_dataset()
    else:
        cur_dataset = utils.load_other_dataset(dataset_name)
    # do filter
    cur_dataset = {item[0]:item[1] for item in cur_dataset.items() if item[0] in held_out_problem_id[dataset_name]}
    datasets.update({
        dataset_name:cur_dataset
    })

llama_groups = {
    'Llama-3.1-8b': [
        ('llama-3.1-8b',None,'',1),('llama-3.1-8b',None,'',2),('llama-3.1-8b',None,'',3),('llama-3.1-8b',None,'',4),('llama-3.1-8b',None,'',5)
    ],
    'Llama-3.1-8b-dpo-fixed-ratio': [
        ("llama-fix",None,'',1),("llama-fix",None,'',2),("llama-fix",None,'',3),("llama-fix",None,'',4),("llama-fix",None,'',5)
    ],
    'Llama-3.1-8b-dpo-momentum-curriulum': [
        ("llama-momentum",None,'',1),("llama-momentum",None,'',2),("llama-momentum",None,'',3),("llama-momentum",None,'',4),("llama-momentum",None,'',5)
    ],
}
qwen_groups = {
    'qwen-2.5-7b': [
        ('qwen2.5-7b',None,'',1),('qwen2.5-7b',None,'',2),('qwen2.5-7b',None,'',3),('qwen2.5-7b',None,'',4),('qwen2.5-7b',None,'',5)
    ],
    'qwen-2.5-7b-dpo-fixed-ratio': [
        ('qwen-fix',None,'',1),('qwen-fix',None,'',2),('qwen-fix',None,'',3),('qwen-fix',None,'',4),('qwen-fix',None,'',5)
    ],
    'qwen-2.5-7b-dpo-momentum-curriulum': [
        ("qwen-momentum",None,'',1),("qwen-momentum",None,'',2),("qwen-momentum",None,'',3),("qwen-momentum",None,'',4),("qwen-momentum",None,'',5)
    ],
}
nextcoder_groups = {
    'nextcoder-7b': [
        ('mic-coder-7b',None,'',1),('mic-coder-7b',None,'',2),('mic-coder-7b',None,'',3),('mic-coder-7b',None,'',4),('mic-coder-7b',None,'',5)
    ],
    'nextcoder-7b-dpo-fixed-ratio': [
        ("nextcoder-fix",None,'',1),("nextcoder-fix",None,'',2),("nextcoder-fix",None,'',3),("nextcoder-fix",None,'',4),("nextcoder-fix",None,'',5)
    ],
    'nextcoder-7b-momentum-curriulum': [
        ('nextcoder-momentum',None,'',1),('nextcoder-momentum',None,'',2),('nextcoder-momentum',None,'',3),('nextcoder-momentum',None,'',4),('nextcoder-momentum',None,'',5)
    ],
}


postfix_mapping_dict = {
    "parsing-ablation":["llama-ablation-parsing","llama-ablation-consistent","llama-ablation-fair"],
    "consistent-ablation":["qwen-ablation-parsing","qwen-ablation-consistent","qwen-ablation-fair"],
    "fair-ablation":["nextcoder-ablation-parsing","nextcoder-ablation-consistent","nextcoder-ablation-fair"]
}

for model_index, (model_name,mapping_grounp) in enumerate(zip(['Llama-3.1-8b','qwem-2.5-7b','nextcoder-7b'],(llama_groups, qwen_groups, nextcoder_groups))):
    for postfix in postfix_mapping_dict:
        adaptor_index = postfix_mapping_dict[postfix][model_index]
        
        mapping_grounp.update({
            f'{model_name}-{postfix}':[(adaptor_index,None,'',i) for i in range(1,6)]
        })
    

grouped_cfgs = OrderedDict({
    **llama_groups,
    **qwen_groups,
    **nextcoder_groups,
})


multi_model_data, group_map_eval, group_order = example_utils._load_eval_data_for_named_groups(grouped_cfgs)

div_group_df = example_utils.generate_performance_table_t3_v2(
    multi_model_data,
    dataset_name='all',
    held_out_map=held_out_problem_id, 
    subset='held_out',             
    group_map=group_map_eval,  
)

div_group_df.index = [f'[{g}] (Agg)' for g in div_group_df.index]


final_results, group_map_res, group_order_res = example_utils._load_results_for_named_groups(grouped_cfgs)

pass_df_all = example_utils.generate_passrate_table_v3(
    final_results,
    dataset_name='all'
)
pass_df_with_agg = example_utils.append_group_summaries_passrate(
    pass_df_all,
    group_map=group_map_res,
    group_order=group_order_res,
    summary_suffix="(Agg)"
)

agg_index = [f'[{label}] (Agg)' for label in group_order_res]  
pass_group_df = pass_df_with_agg.loc[agg_index]

final_agg_only = div_group_df.join(pass_group_df, how='outer').loc[agg_index].round(4)

from IPython.display import display
display(final_agg_only)
final_agg_only.to_csv('named_groups_all_metrics_AGG_ONLY_9rows.csv')

import pandas as pd
import numpy as np

div_group_df.index = [f'[{g}] (Agg)' for g in div_group_df.index]


keep_cols_map = {
    'Instruction Following': 'Instruction Following Rate',
    'Consistency': 'Consistency',
    'Generalizability (Valid)': 'Generalizability',
    'Beta Struct Diversity (Valid)': 'Beta Diversity',
    'Normalized Gamma Diversity (Valid, q=0)': 'Gamma Diversity',
    'Avg Train Pass Rate': 'Avg Train Pass Rate',
    'Avg Test Pass Rate': 'Avg Test Pass Rate',
    'Top1 Accuracy': 'Top-1 Accuracy',
    'Top2 Accuracy': 'Top-2 Accuracy',
    'Top3 Accuracy': 'Top-3 Accuracy',
}

latex_split = example_utils.make_split_agg_latex(
    final_agg_only=final_agg_only,
    grouped_cfgs=grouped_cfgs,
    caption='Aggregated results across nine settings (group averages only).',
    label='tab:agg_9rows_split',
    as_percent=False,
    decimals=3,
    cross_column=False,           
    strip_brackets_in_index=True, 
    div_block_title='Diversity block',
    acc_block_title='T3 accuracy block'
)
print(latex_split)
# Some of the LLMs generated hypothesis may print something out during execution
```

    Loading data from meta-llama/Llama-3.1-8B-Instruct_round1
    Loading data from meta-llama/Llama-3.1-8B-Instruct_round2
    Loading data from meta-llama/Llama-3.1-8B-Instruct_round3
    Loading data from meta-llama/Llama-3.1-8B-Instruct_round4
    Loading data from meta-llama/Llama-3.1-8B-Instruct_round5
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round1
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round2
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round3
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round4
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round5
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round1
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round2
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round3
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round4
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round5
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round1
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round2
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round3
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round4
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round5
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round1
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round2
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round3
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round4
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round5
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round1
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round2
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round3
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round4
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation_round5
    Loading data from Qwen/Qwen2.5-7B-Instruct_round1
    Loading data from Qwen/Qwen2.5-7B-Instruct_round2
    Loading data from Qwen/Qwen2.5-7B-Instruct_round3
    Loading data from Qwen/Qwen2.5-7B-Instruct_round4
    Loading data from Qwen/Qwen2.5-7B-Instruct_round5
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round1
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round2
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round3
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round4
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1_round5
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round1
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round2
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round3
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round4
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum_round5
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round1
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round2
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round3
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round4
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round5
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round1
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round2
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round3
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round4
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round5
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round1
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round2
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round3
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round4
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation_round5
    Loading data from microsoft/NextCoder-7B_round1
    Loading data from microsoft/NextCoder-7B_round2
    Loading data from microsoft/NextCoder-7B_round3
    Loading data from microsoft/NextCoder-7B_round4
    Loading data from microsoft/NextCoder-7B_round5
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_longfinal_round1
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_longfinal_round2
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_longfinal_round3
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_longfinal_round4
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_longfinal_round5
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_momentum_long_steady_new_1_round1
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_momentum_long_steady_new_1_round2
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_momentum_long_steady_new_1_round3
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_momentum_long_steady_new_1_round4
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_momentum_long_steady_new_1_round5
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round1
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round2
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round3
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round4
    Loading data from meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round5
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round1
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round2
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round3
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round4
    Loading data from Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round5
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round1
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round2
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round3
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round4
    Loading data from microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation_round5
    --- Starting Performance Analysis for All Datasets (Averaged) (Held-out Subset) ---
    --- Analysis Complete. Performance Table: ---


    <ast_guarded>:7: SyntaxWarning: 'generator' object is not subscriptable; perhaps you missed a comma?


    0 0 5 0 0 0 6 0 3 0 0 0 3 0 0 0 0 8 0 6 0 8 0 0 0 
    0 7 0 0 0 0 9 0 0 0 9 0 0 7 2 0 0 3 0 0 0 0 0 3 0 
    0 0 5 0 3 0 4 0 0 0 0 0 0 0 4 8 0 3 0 0 0 0 0 8 0 
    0 5 0 0 0 0 0 7 4 0 0 0 7 0 0 1 0 0 4 0 0 0 1 0 0 
    0 0 0 3 0 0 6 0 0 7 7 0 6 0 0 0 0 1 1 3 0 0 0 0 0 
    0 0 9 7 0 0 0 0 0 9 3 4 7 0 0 0 0 0 4 0 0 3 0 0 0 


    <unknown>:3: SyntaxWarning: invalid decimal literal
    <unknown>:3: SyntaxWarning: invalid decimal literal
    <unknown>:4: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:3: SyntaxWarning: invalid decimal literal
    <unknown>:2: SyntaxWarning: invalid decimal literal
    <unknown>:6: SyntaxWarning: invalid decimal literal


    -1
    -1
    -1
    -1
    3
    3
    -1
    -1
    -1
    -1
    -1
    -1
    -1
    3
    3
    4
    4
    -1
    -1
    -1
    -1
    -1
    -1
    -1
    3
    5
    5
    -1
    -1
    -1
    -1
    -1
    -1
    4
    2
    4
    -1
    -1
    -1
    -1
    -1
    -1
    -1
    4
    4
    5
    -1
    -1
    -1
    -1
    --- Generating Pass-Rate Table for dataset='all' ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Instruction Following</th>
      <th>Consistency</th>
      <th>Normalized Gamma Diversity (Valid, q=0)</th>
      <th>Beta Struct Diversity (Valid)</th>
      <th>Generalizability (Valid)</th>
      <th>Hypotheses Generated (Valid)</th>
      <th>Avg Train Pass Rate</th>
      <th>Avg Test Pass Rate</th>
      <th>Top1 Accuracy</th>
      <th>Top2 Accuracy</th>
      <th>Top3 Accuracy</th>
      <th>#Hypotheses</th>
      <th>#Follow</th>
      <th>#Non-Follow</th>
      <th>Follow Rate</th>
      <th>#Inconsistent</th>
      <th>#Consistent</th>
      <th>Inconsistent Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[Llama-3.1-8b] (Agg)</th>
      <td>0.1488</td>
      <td>0.0021</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0</td>
      <td>0.0115</td>
      <td>0.0089</td>
      <td>0.0</td>
      <td>0.0025</td>
      <td>0.0037</td>
      <td>2400</td>
      <td>357</td>
      <td>2043</td>
      <td>0.1488</td>
      <td>352</td>
      <td>5</td>
      <td>0.9877</td>
    </tr>
    <tr>
      <th>[Llama-3.1-8b-dpo-fixed-ratio] (Agg)</th>
      <td>0.9679</td>
      <td>0.0142</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>6.6</td>
      <td>0.0963</td>
      <td>0.0761</td>
      <td>0.0275</td>
      <td>0.0425</td>
      <td>0.0487</td>
      <td>2400</td>
      <td>2323</td>
      <td>77</td>
      <td>0.9679</td>
      <td>2290</td>
      <td>33</td>
      <td>0.9858</td>
    </tr>
    <tr>
      <th>[Llama-3.1-8b-dpo-momentum-curriulum] (Agg)</th>
      <td>0.9700</td>
      <td>0.0204</td>
      <td>1.1795</td>
      <td>0.1959</td>
      <td>1.0000</td>
      <td>9.8</td>
      <td>0.1285</td>
      <td>0.1005</td>
      <td>0.035</td>
      <td>0.0525</td>
      <td>0.065</td>
      <td>2400</td>
      <td>2328</td>
      <td>72</td>
      <td>0.97</td>
      <td>2279</td>
      <td>49</td>
      <td>0.9789</td>
    </tr>
    <tr>
      <th>[Llama-3.1-8b-parsing-ablation] (Agg)</th>
      <td>0.9412</td>
      <td>0.0092</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>4.4</td>
      <td>0.1295</td>
      <td>0.0986</td>
      <td>0.0263</td>
      <td>0.0312</td>
      <td>0.0412</td>
      <td>2400</td>
      <td>2259</td>
      <td>141</td>
      <td>0.9412</td>
      <td>2237</td>
      <td>22</td>
      <td>0.9903</td>
    </tr>
    <tr>
      <th>[Llama-3.1-8b-consistent-ablation] (Agg)</th>
      <td>0.9962</td>
      <td>0.0321</td>
      <td>1.0077</td>
      <td>0.0138</td>
      <td>1.0000</td>
      <td>15.4</td>
      <td>0.1979</td>
      <td>0.1537</td>
      <td>0.045</td>
      <td>0.055</td>
      <td>0.0662</td>
      <td>2400</td>
      <td>2391</td>
      <td>9</td>
      <td>0.9963</td>
      <td>2314</td>
      <td>77</td>
      <td>0.9678</td>
    </tr>
    <tr>
      <th>[Llama-3.1-8b-fair-ablation] (Agg)</th>
      <td>0.9917</td>
      <td>0.0196</td>
      <td>1.1403</td>
      <td>0.2217</td>
      <td>1.0000</td>
      <td>9.4</td>
      <td>0.162</td>
      <td>0.1251</td>
      <td>0.045</td>
      <td>0.055</td>
      <td>0.0625</td>
      <td>2400</td>
      <td>2380</td>
      <td>20</td>
      <td>0.9917</td>
      <td>2333</td>
      <td>47</td>
      <td>0.9802</td>
    </tr>
    <tr>
      <th>[qwen-2.5-7b] (Agg)</th>
      <td>0.9883</td>
      <td>0.0371</td>
      <td>1.0014</td>
      <td>0.0318</td>
      <td>0.9869</td>
      <td>17.8</td>
      <td>0.1982</td>
      <td>0.1543</td>
      <td>0.035</td>
      <td>0.0588</td>
      <td>0.0675</td>
      <td>2400</td>
      <td>2372</td>
      <td>28</td>
      <td>0.9883</td>
      <td>2283</td>
      <td>89</td>
      <td>0.9625</td>
    </tr>
    <tr>
      <th>[qwen-2.5-7b-dpo-fixed-ratio] (Agg)</th>
      <td>0.9958</td>
      <td>0.0417</td>
      <td>1.0459</td>
      <td>0.0437</td>
      <td>0.9869</td>
      <td>20.0</td>
      <td>0.2053</td>
      <td>0.1597</td>
      <td>0.0537</td>
      <td>0.0638</td>
      <td>0.0775</td>
      <td>2400</td>
      <td>2390</td>
      <td>10</td>
      <td>0.9958</td>
      <td>2290</td>
      <td>100</td>
      <td>0.9581</td>
    </tr>
    <tr>
      <th>[qwen-2.5-7b-dpo-momentum-curriulum] (Agg)</th>
      <td>0.9971</td>
      <td>0.0408</td>
      <td>1.0518</td>
      <td>0.0732</td>
      <td>1.0000</td>
      <td>19.6</td>
      <td>0.2027</td>
      <td>0.1626</td>
      <td>0.0662</td>
      <td>0.0825</td>
      <td>0.095</td>
      <td>2400</td>
      <td>2393</td>
      <td>7</td>
      <td>0.9971</td>
      <td>2295</td>
      <td>98</td>
      <td>0.959</td>
    </tr>
    <tr>
      <th>[qwem-2.5-7b-parsing-ablation] (Agg)</th>
      <td>0.3221</td>
      <td>0.0087</td>
      <td>1.0108</td>
      <td>0.0136</td>
      <td>1.0000</td>
      <td>4.2</td>
      <td>0.0844</td>
      <td>0.0658</td>
      <td>0.0175</td>
      <td>0.0225</td>
      <td>0.0275</td>
      <td>2400</td>
      <td>773</td>
      <td>1627</td>
      <td>0.3221</td>
      <td>752</td>
      <td>21</td>
      <td>0.9744</td>
    </tr>
    <tr>
      <th>[qwem-2.5-7b-consistent-ablation] (Agg)</th>
      <td>0.9883</td>
      <td>0.0467</td>
      <td>1.0098</td>
      <td>0.0146</td>
      <td>0.9745</td>
      <td>22.2</td>
      <td>0.2241</td>
      <td>0.1761</td>
      <td>0.0587</td>
      <td>0.075</td>
      <td>0.0862</td>
      <td>2400</td>
      <td>2372</td>
      <td>28</td>
      <td>0.9883</td>
      <td>2261</td>
      <td>111</td>
      <td>0.9532</td>
    </tr>
    <tr>
      <th>[qwem-2.5-7b-fair-ablation] (Agg)</th>
      <td>0.9446</td>
      <td>0.0337</td>
      <td>1.0395</td>
      <td>0.0536</td>
      <td>1.0000</td>
      <td>16.2</td>
      <td>0.2004</td>
      <td>0.1534</td>
      <td>0.06</td>
      <td>0.0737</td>
      <td>0.08</td>
      <td>2400</td>
      <td>2267</td>
      <td>133</td>
      <td>0.9446</td>
      <td>2186</td>
      <td>81</td>
      <td>0.9642</td>
    </tr>
    <tr>
      <th>[nextcoder-7b] (Agg)</th>
      <td>0.8625</td>
      <td>0.0125</td>
      <td>1.0811</td>
      <td>0.1395</td>
      <td>1.0000</td>
      <td>5.6</td>
      <td>0.1512</td>
      <td>0.1206</td>
      <td>0.025</td>
      <td>0.0438</td>
      <td>0.055</td>
      <td>2400</td>
      <td>2360</td>
      <td>40</td>
      <td>0.9833</td>
      <td>2325</td>
      <td>35</td>
      <td>0.9851</td>
    </tr>
    <tr>
      <th>[nextcoder-7b-dpo-fixed-ratio] (Agg)</th>
      <td>0.9654</td>
      <td>0.0277</td>
      <td>1.0658</td>
      <td>0.0783</td>
      <td>1.0000</td>
      <td>13.2</td>
      <td>0.1731</td>
      <td>0.1356</td>
      <td>0.0513</td>
      <td>0.065</td>
      <td>0.0738</td>
      <td>2400</td>
      <td>2317</td>
      <td>83</td>
      <td>0.9654</td>
      <td>2251</td>
      <td>66</td>
      <td>0.9715</td>
    </tr>
    <tr>
      <th>[nextcoder-7b-momentum-curriulum] (Agg)</th>
      <td>0.9912</td>
      <td>0.0300</td>
      <td>1.1507</td>
      <td>0.1824</td>
      <td>0.9906</td>
      <td>14.4</td>
      <td>0.1893</td>
      <td>0.1495</td>
      <td>0.065</td>
      <td>0.08</td>
      <td>0.0925</td>
      <td>2400</td>
      <td>2379</td>
      <td>21</td>
      <td>0.9912</td>
      <td>2307</td>
      <td>72</td>
      <td>0.9698</td>
    </tr>
    <tr>
      <th>[nextcoder-7b-parsing-ablation] (Agg)</th>
      <td>0.2579</td>
      <td>0.0054</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>2.6</td>
      <td>0.0389</td>
      <td>0.0299</td>
      <td>0.0112</td>
      <td>0.0162</td>
      <td>0.02</td>
      <td>2400</td>
      <td>619</td>
      <td>1781</td>
      <td>0.2579</td>
      <td>606</td>
      <td>13</td>
      <td>0.9802</td>
    </tr>
    <tr>
      <th>[nextcoder-7b-consistent-ablation] (Agg)</th>
      <td>0.9812</td>
      <td>0.0342</td>
      <td>1.0124</td>
      <td>0.0232</td>
      <td>1.0000</td>
      <td>15.6</td>
      <td>0.1834</td>
      <td>0.141</td>
      <td>0.0388</td>
      <td>0.0563</td>
      <td>0.065</td>
      <td>2400</td>
      <td>2355</td>
      <td>45</td>
      <td>0.9813</td>
      <td>2277</td>
      <td>78</td>
      <td>0.9669</td>
    </tr>
    <tr>
      <th>[nextcoder-7b-fair-ablation] (Agg)</th>
      <td>0.9796</td>
      <td>0.0150</td>
      <td>1.1453</td>
      <td>0.2023</td>
      <td>0.9757</td>
      <td>7.2</td>
      <td>0.1497</td>
      <td>0.118</td>
      <td>0.0325</td>
      <td>0.0425</td>
      <td>0.0487</td>
      <td>2400</td>
      <td>2351</td>
      <td>49</td>
      <td>0.9796</td>
      <td>2315</td>
      <td>36</td>
      <td>0.9847</td>
    </tr>
  </tbody>
</table>
</div>


    \begin{table}[t]
        \centering
        \setlength{\tabcolsep}{3pt}
        \renewcommand{\arraystretch}{0.95}
        \caption{Aggregated results across nine settings (group averages only).}
        \label{tab:agg_9rows_split}
        {\itshape Diversity block}\par
        \vspace{0.25em}
        \resizebox{\linewidth}{!}{%
    \begin{tabular}{lccccc}
    \toprule
     & Instruction Following Rate & Consistency & Generalizability & Beta Diversity & Gamma Diversity \\
    \midrule
    Llama-3.1-8b & 0.149 & 0.002 & 1.000 & 0.000 & 1.000 \\
    Llama-3.1-8b-dpo-fixed-ratio & 0.968 & 0.014 & 1.000 & 0.000 & 1.000 \\
    Llama-3.1-8b-dpo-momentum-curriulum & 0.970 & 0.020 & 1.000 & 0.196 & 1.179 \\
    Llama-3.1-8b-parsing-ablation & 0.941 & 0.009 & 1.000 & 0.000 & 1.000 \\
    Llama-3.1-8b-consistent-ablation & 0.996 & 0.032 & 1.000 & 0.014 & 1.008 \\
    Llama-3.1-8b-fair-ablation & 0.992 & 0.020 & 1.000 & 0.222 & 1.140 \\
    qwen-2.5-7b & 0.988 & 0.037 & 0.987 & 0.032 & 1.001 \\
    qwen-2.5-7b-dpo-fixed-ratio & 0.996 & 0.042 & 0.987 & 0.044 & 1.046 \\
    qwen-2.5-7b-dpo-momentum-curriulum & 0.997 & 0.041 & 1.000 & 0.073 & 1.052 \\
    qwem-2.5-7b-parsing-ablation & 0.322 & 0.009 & 1.000 & 0.014 & 1.011 \\
    qwem-2.5-7b-consistent-ablation & 0.988 & 0.047 & 0.975 & 0.015 & 1.010 \\
    qwem-2.5-7b-fair-ablation & 0.945 & 0.034 & 1.000 & 0.054 & 1.040 \\
    nextcoder-7b & 0.863 & 0.013 & 1.000 & 0.140 & 1.081 \\
    nextcoder-7b-dpo-fixed-ratio & 0.965 & 0.028 & 1.000 & 0.078 & 1.066 \\
    nextcoder-7b-momentum-curriulum & 0.991 & 0.030 & 0.991 & 0.182 & 1.151 \\
    nextcoder-7b-parsing-ablation & 0.258 & 0.005 & 1.000 & 0.000 & 1.000 \\
    nextcoder-7b-consistent-ablation & 0.981 & 0.034 & 1.000 & 0.023 & 1.012 \\
    nextcoder-7b-fair-ablation & 0.980 & 0.015 & 0.976 & 0.202 & 1.145 \\
    \bottomrule
    \end{tabular}
        }
        \vspace{0.6em}
        {\itshape T3 accuracy block}\par
        \vspace{0.25em}
        \resizebox{\linewidth}{!}{%
    \begin{tabular}{lccccc}
    \toprule
     & Avg Train Pass Rate & Avg Test Pass Rate & Top-1 Accuracy & Top-2 Accuracy & Top-3 Accuracy \\
    \midrule
    Llama-3.1-8b & 0.011 & 0.009 & 0.000 & 0.003 & 0.004 \\
    Llama-3.1-8b-dpo-fixed-ratio & 0.096 & 0.076 & 0.028 & 0.043 & 0.049 \\
    Llama-3.1-8b-dpo-momentum-curriulum & 0.129 & 0.101 & 0.035 & 0.052 & 0.065 \\
    Llama-3.1-8b-parsing-ablation & 0.130 & 0.099 & 0.026 & 0.031 & 0.041 \\
    Llama-3.1-8b-consistent-ablation & 0.198 & 0.154 & 0.045 & 0.055 & 0.066 \\
    Llama-3.1-8b-fair-ablation & 0.162 & 0.125 & 0.045 & 0.055 & 0.062 \\
    qwen-2.5-7b & 0.198 & 0.154 & 0.035 & 0.059 & 0.068 \\
    qwen-2.5-7b-dpo-fixed-ratio & 0.205 & 0.160 & 0.054 & 0.064 & 0.077 \\
    qwen-2.5-7b-dpo-momentum-curriulum & 0.203 & 0.163 & 0.066 & 0.083 & 0.095 \\
    qwem-2.5-7b-parsing-ablation & 0.084 & 0.066 & 0.018 & 0.022 & 0.028 \\
    qwem-2.5-7b-consistent-ablation & 0.224 & 0.176 & 0.059 & 0.075 & 0.086 \\
    qwem-2.5-7b-fair-ablation & 0.200 & 0.153 & 0.060 & 0.074 & 0.080 \\
    nextcoder-7b & 0.151 & 0.121 & 0.025 & 0.044 & 0.055 \\
    nextcoder-7b-dpo-fixed-ratio & 0.173 & 0.136 & 0.051 & 0.065 & 0.074 \\
    nextcoder-7b-momentum-curriulum & 0.189 & 0.149 & 0.065 & 0.080 & 0.092 \\
    nextcoder-7b-parsing-ablation & 0.039 & 0.030 & 0.011 & 0.016 & 0.020 \\
    nextcoder-7b-consistent-ablation & 0.183 & 0.141 & 0.039 & 0.056 & 0.065 \\
    nextcoder-7b-fair-ablation & 0.150 & 0.118 & 0.033 & 0.043 & 0.049 \\
    \bottomrule
    \end{tabular}
        }
    \end{table}

