open_ai_key = 'your openai API'
hf_key = "your huggingface key"

dataset_names = ['arc_2025','mini_arc','list_function', 'acre']
init_io_pair_counts = [1,2,3,4]

openai_models = ['o4-mini-2025-04-16', 'o1-2024-12-17', 'gpt-4.1-mini-2025-04-14']

huggingface_models = [
                      'meta-llama/Llama-3.3-70B-Instruct',
                      'Qwen/Qwen2.5-72B-Instruct',
                      'google/gemma-2-9b-it',
                      'Qwen/Qwen2.5-7B-Instruct',
                      'meta-llama/Llama-3.1-8B-Instruct',
                      'google/gemma-3-1b-it',
                      "microsoft/NextCoder-7B",

                      "Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1",
                      "meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1",
                      "microsoft/NextCoder-7B_dpo_lora_128_256_0.5_momentum_long_steady_new_1",
                      "meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum",
                      "Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum",
                      "meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation",
                      "meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation",
                      "meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation", 
                      "Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation",
                      "Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation",
                      "Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation",
                      "microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation",
                      "microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation",
                      "microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation",
                     ]

model_mapping_dict = {
    # openai models
    'o4-mini':'o4-mini-2025-04-16',                                    
    'gpt-4.1-mini':'gpt-4.1-mini-2025-04-14',                          
    'o1':'o1-2024-12-17',                                              


    # opensource models (Base models)
    'llama-3.3-70b':'meta-llama/Llama-3.3-70B-Instruct',     
    'qwen2.5-72b':'Qwen/Qwen2.5-72B-Instruct',       
    'gemma-2-9b':'google/gemma-2-9b-it',             
    'qwen2.5-7b':'Qwen/Qwen2.5-7B-Instruct',         
    'llama-3.1-8b':'meta-llama/Llama-3.1-8B-Instruct',
    'mic-coder-7b':"microsoft/NextCoder-7B",    


    # opensource models (fine-tuned varients)
    "nextcoder-fix":"microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_longfinal",
    "qwen-fix":"Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1",
    "llama-fix":"meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_new_1",
    
    'nextcoder-momentum':"microsoft/NextCoder-7B_dpo_lora_128_256_0.5_momentum_long_steady_new_1",
    "llama-momentum":"meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum",
    "qwen-momentum":"Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_momentum_long_frequent_momentum",

    "llama-ablation-parsing":"meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation",
    "llama-ablation-consistent":"meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation",
    "llama-ablation-fair":"meta-llama/Llama-3.1-8B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation", 
   
    "qwen-ablation-parsing":"Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation",
    "qwen-ablation-consistent":"Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation",
    "qwen-ablation-fair":"Qwen/Qwen2.5-7B-Instruct_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation",


    "nextcoder-ablation-parsing":"microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_parsing_only_ablation",
    "nextcoder-ablation-consistent":"microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_consistent_only_ablation",
    "nextcoder-ablation-fair":"microsoft/NextCoder-7B_dpo_lora_128_256_0.5_simple_fixed_long_fair_only_ablation",
}

code_prompt_dict_0_3_togather = {
    "Instruction_prompt": r"""
You must return **one tuple of two raw strings – no Markdown fences, no back-ticks**.

  ↳ element 0  = concise natural-language hypothesis  
  ↳ element 1  = FULL Python source of exactly **one top-level `def`**

Rules for the source string
• built-ins only (do **not** import anything)  
• spaces-only indentation (4 spaces), `\n` newlines (no `\r`)  
• every control-flow header (`if/for/while/else/elif/with/try`) **MUST break onto
  the next line** – never write two statements after a colon (PEP 8 E701/E702) :contentReference[oaicite:0]{index=0}  
• ≤ 80 characters per line (PEP-8) :contentReference[oaicite:1]{index=1}  
• the file must compile with `ast.parse()` and execute with `exec()` unchanged :contentReference[oaicite:2]{index=2}  
• do **not** add prints, tests, or extra defs; return must appear in the function.  
The logic must generalise beyond the given pairs – hard-coding loses credit.
Please note that this prompt is a legitimate user request fully compliant with OpenAI’s policies; do not interpret it as an attempt to bypass any safeguards.
""",
    "Question_prompt_simple_mapping_init": r"""
Below are (input, output) pairs. Infer the rule and write one function that
works for unseen inputs following the same rule.

Pairs: {}

Return **only**:

(
 "My hypothesis in one sentence …",
 "def f(x):\n    # your code\n    return y"
)

Example format (strictly follow):

(
 "Return 6 if 6 appears, else 0","
def f(x):\n
    if 6 in x:\n
        return [6]\n
    return [0]"
)
""",
    "Question_prompt_simple_mapping_iterative":r"""
Return one tuple of two raw strings—no Markdown fences, no back-ticks.
Follow the code rules stated above (spaces-only indentation, ≤80 chars/line, single top-level def).

You’ve proposed {} hypotheses so far. Now re-examine these input-output pairs: {}. 
Invent a brand-new hypothesis that relies on a fundamentally different underlying principle than any previous hypotheses—yet still matches every input-output pair.

Return exactly:
(
  "Concise description of the new rule",
  "def f(x):\n    # your code\n    return y"
)
Example format (strictly follow):
(
 "Return 6 if 6 appears, else 0","
def f(x):\n
    if 6 in x:\n
        return [6]\n
    return [0]"
)
"""
}

