"""
Simplified DPO training with **fixed-ratio sampling by reason** and **balanced per-dataset sourcing**.
- Removes curriculum/adaptive sampling and custom trainer.
- Uses a single fixed draw with ratios: parsing:inconsistent:fair comparsion = 2:4:4 (configurable).
- Balances samples across datasets within each reason as evenly as possible.

Run example:
  python dpo_training_simple_fixed_ratio.py \
      -m llama_3_8b \
      -g 0 \
      -t scratch \
      --short_context

Notes:
- Keeps your short/super_short context trimming logic.
- Preserves SFT→DPO path (model_adapter_name/ref_adapter_name) when -t sft.
- Evaluation is standard TRL DPO evaluation on the whole eval set (no per-reason slicing).
"""

import json
import random
from typing import *
from collections import defaultdict
import argparse
import os
import copy

from global_variables import *  # expects model_mapping_dict

# ---------------------------
# Paths & static data
# ---------------------------
preference_label_data_path = 'training_data/rl_data/final_labeled_data_v1_dpo.json'
with open(preference_label_data_path, 'r') as f:
    structured_labeled_data = json.load(f)

with open('training_data/held_out_ids.json', 'r') as f:
    leftover_pids_per_dataset = json.load(f)

with open('training_data/dpo_train_ids.json', 'r') as f:
    dpo_train_ids = json.load(f)

with open('training_data/dpo_val_ids.json', 'r') as f:
    dpo_val_ids = json.load(f)

# ---------------------------
# Utilities
# ---------------------------

def print_split_headcounts(structured_data: Dict[str, Any]) -> None:
    for ds_name, io_counts in structured_data.items():
        pid_set = set()
        for io_count, problems in io_counts.items():
            pid_set.update(problems.keys())
        print(f"🔍 数据集 '{ds_name}' 独立问题数: {len(pid_set)}")


def create_problem_based_splits(
    structured_data: Dict[str, Any],
    random_seed: int = 2025,
    train_pids_per_dataset: Optional[Dict[str, List[str]]] = None,
    test_pids_per_dataset: Optional[Dict[str, List[str]]] = None,
    leftover_pids_per_dataset: Optional[Dict[str, List[str]]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Your original splitter, unchanged except no curriculum bits.
    Respects externally provided train/test/leftover sets.
    Uses global 'args' for short/super_short context trimming, as in your script.
    """
    rng = random.Random(random_seed)

    # --- Stats ---
    print_split_headcounts(structured_data)

    # External split path
    use_external_split = (
        train_pids_per_dataset is not None and
        test_pids_per_dataset is not None and
        leftover_pids_per_dataset is not None
    )

    if use_external_split:
        print("⚡ 使用外部分组（train/test/leftover）进行划分")
        train_pids_per_dataset = {k: set(v) for k, v in train_pids_per_dataset.items()}
        test_pids_per_dataset = {k: set(v) for k, v in test_pids_per_dataset.items()}
    else:
        # Fallback: simple proportional split if externals are missing
        unique_per_dataset: Dict[str, List[str]] = {}
        for ds_name, io_counts in structured_data.items():
            pid_set = set()
            for io_count, problems in io_counts.items():
                pid_set.update(problems.keys())
            unique_per_dataset[ds_name] = list(pid_set)
        raw_train_sizes = {ds: int(len(pids) * 0.5) for ds, pids in unique_per_dataset.items()}
        raw_test_sizes  = {ds: int(len(pids) * 0.1) for ds, pids in unique_per_dataset.items()}
        min_train = min(raw_train_sizes.values())
        min_test  = min(raw_test_sizes.values())
        print(f"ℹ️ 全局最小训练集大小: {min_train}，全局最小测试集大小: {min_test}")

        train_pids_per_dataset = {}
        test_pids_per_dataset = {}
        leftover_local = {}
        for ds, pids in unique_per_dataset.items():
            rng.shuffle(pids)
            train_pids = set(rng.sample(pids, min_train))
            remaining = [pid for pid in pids if pid not in train_pids]
            test_pids  = set(rng.sample(remaining, min_test))
            leftover = [pid for pid in pids if pid not in train_pids and pid not in test_pids]
            train_pids_per_dataset[ds] = train_pids
            test_pids_per_dataset[ds]  = test_pids
            leftover_local[ds]         = leftover
        leftover_pids_per_dataset = leftover_local

    # Build rows
    train_dataset: List[Dict] = []
    test_dataset:  List[Dict] = []

    for ds_name, io_counts in structured_data.items():
        for io_count, problems in io_counts.items():
            for problem_id, contexts in problems.items():
                if problem_id in train_pids_per_dataset.get(ds_name, set()):
                    target = train_dataset
                elif problem_id in test_pids_per_dataset.get(ds_name, set()):
                    target = test_dataset
                else:
                    continue
                for ctx_len, labeled_list in contexts.items():
                    for context, chosen, rejected, reason in labeled_list:
                        # context shortening (same as your original)
                        if args.short_context:
                            keep_context_index = [0, 1]
                            for index, context_message in enumerate(context):
                                if index in keep_context_index:
                                    continue
                                if context_message['role'] == 'user':
                                    context_message['content'] = 'Invent a brand-new hypothesis based on a fundamentally different principle from any of your previous hypotheses, while still matching every given input-output pair. Use exactly the same output format as before.'
                        if args.super_short_context:
                            new_context = [context[0], context[1]]
                            new_instruction = {"role": "user", "content": "Please generate as much hypothesis as possible (one at a time), and make sure new generated hypothesis is based on a fundamentally different principle from any of your previous hypotheses."}
                            new_context.append(new_instruction)
                            keep_context_index = [0, 1]
                            for index, context_message in enumerate(context):
                                if index in keep_context_index:
                                    continue
                                if context_message['role'] == 'user':
                                    context_message['content'] = 'Another.'
                            new_context.extend(context[2:])
                            context = new_context

                        target.append({
                            "prompt": context,
                            "chosen": chosen,
                            "rejected": rejected,
                            "reason": reason,
                            "dataset": ds_name,
                            "io_pair_count": str(io_count),
                            "problem_id": problem_id,
                            "context_length": str(ctx_len),
                        })

    print("✅ 按传入/自动采样分组后的 train/test 划分完成！")
    return train_dataset, test_dataset


def summarize_distribution(rows: List[Dict], title: str) -> None:
    print(f"\n📊 {title}")
    by_reason = defaultdict(int)
    by_reason_ds = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_reason[r['reason']] += 1
        by_reason_ds[r['reason']][r['dataset']] += 1
    total = len(rows)
    print(f"  Total: {total}")
    for reason, cnt in by_reason.items():
        pct = 100.0 * cnt / max(1, total)
        print(f"  - {reason}: {cnt} ({pct:.1f}%)")
        for ds, c in sorted(by_reason_ds[reason].items()):
            print(f"      · {ds}: {c}")

def sample_training_data_balanced(
    train_data: List[Dict],
    n_samples: int,
    ratios: Dict[str, float],
    seed: int = 2025,
) -> List[Dict]:
    """Fixed-ratio sampling by reason with **balanced per-dataset** allocation.

    - For each reason, allocate quota = floor(...) except the last reason uses the remainder to make sum exact.
    - Within a reason, split its quota evenly across datasets that have that reason.
    - If a dataset lacks enough examples for its quota, we redistribute the remainder to other datasets of the *same reason*.
    - If the reason as a whole has fewer examples than its quota, we take all from that reason (no cross-reason top-up).
    """
    rng = random.Random(seed)

    # Bucket by reason→dataset
    by_reason_ds: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in train_data:
        by_reason_ds[rec['reason']][rec['dataset']].append(rec)

    # Compute integer quotas per reason, ensuring exact total equals n_samples
    reasons = list(ratios.keys())
    tot_ratio = sum(ratios.values())
    quotas: Dict[str, int] = {}
    used = 0
    for i, r in enumerate(reasons):
        if i < len(reasons) - 1:
            q = int(n_samples * (ratios[r] / tot_ratio))
            quotas[r] = q
            used += q
        else:
            quotas[r] = max(0, n_samples - used)

    print("\n🎯 固定采样目标配比 (期望):", quotas)

    sampled: List[Dict] = []

    for reason in reasons:
        per_ds = by_reason_ds.get(reason, {})
        if not per_ds:
            print(f"  ⚠️ 原始数据中不存在类别 '{reason}'，跳过。")
            continue

        target = quotas[reason]
        avail_total = sum(len(v) for v in per_ds.values())
        if avail_total <= target:
            print(f"  ⚠️ 类别 '{reason}' 数据不足: 仅 {avail_total} 条，目标 {target} 条，全部选用。")
            # sample all (shuffle for determinism)
            for ds, rows in per_ds.items():
                # deterministic order
                rows_copy = rows[:]
                rng.shuffle(rows_copy)
                sampled.extend(rows_copy)
            continue

        # Initial even split across datasets
        ds_keys = list(per_ds.keys())
        base = target // len(ds_keys)
        remainder = target - base * len(ds_keys)

        alloc = {ds: min(len(per_ds[ds]), base) for ds in ds_keys}
        # Distribute remainder with capacity-aware round-robin
        order = ds_keys[:]
        rng.shuffle(order)
        rem = remainder
        # First pass: give +1 if capacity allows
        for ds in order:
            if rem == 0:
                break
            cap = len(per_ds[ds])
            if alloc[ds] < cap:
                alloc[ds] += 1
                rem -= 1
        # If still remainder exists (rare), do additional passes until filled
        while rem > 0:
            progressed = False
            for ds in order:
                if rem == 0:
                    break
                cap = len(per_ds[ds])
                if alloc[ds] < cap:
                    alloc[ds] += 1
                    rem -= 1
                    progressed = True
            if not progressed:
                # Should not happen since avail_total > target, but guard anyway
                break

        # If some datasets were at capacity and we under-allocated others, redistribute leftover
        need = target - sum(alloc.values())
        if need > 0:
            # fill from any dataset with spare capacity
            for ds in order:
                if need == 0:
                    break
                cap = len(per_ds[ds])
                spare = cap - alloc[ds]
                if spare > 0:
                    take = min(spare, need)
                    alloc[ds] += take
                    need -= take

        # Now sample per dataset
        for ds in ds_keys:
            k = alloc[ds]
            rows = per_ds[ds]
            picks = rng.sample(rows, k)
            sampled.extend(picks)

        print(f"  ✅ '{reason}' 采样 {target} 条 (按数据集分配: "
              + ", ".join([f"{ds}:{alloc[ds]}" for ds in sorted(ds_keys)]) + ")")

    rng.shuffle(sampled)
    print(f"\n✅ 采样完成，总计 {len(sampled)} 条。")
    return sampled


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        help=f"choose the model of experiment that you want to run choose from {model_mapping_dict}")
    parser.add_argument("-s", "--short_context", action="store_true", help="Enable short context mode")
    parser.add_argument("-ss", "--super_short_context", action="store_true", help="Enable super short context mode")
    parser.add_argument("-g", "--gpu", type=str, required=True, help="GPU want to use")
    parser.add_argument("-p", "--postfix", type=str, required=False,default='', help="postfix for the model")
    parser.add_argument("-t", "--train_from", type=str, required=True, choices=["scratch", "sft"],
                        help="train from sft or train from scratch")
    parser.add_argument("-pr", "--preference_reason", type=str, required=True, choices=["parsing", "inconsistent", "fair"],
                        help="preference reason want to train")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path_full = model_mapping_dict[args.model_name]
    print(f"Train model with short context? {args.short_context}")
    print(f"Train model with super short context? {args.super_short_context}")
    assert not (args.short_context and args.super_short_context), "super_short_context and short_context cannot be both True"

    large_model = ['Qwen/Qwen2.5-72B-Instruct', 'meta-llama/Llama-3.3-70B-Instruct']
    small_model = ['Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', 'microsoft/NextCoder-7B']
    use_large_model = model_path_full in large_model
    assert model_path_full in large_model or model_path_full in small_model

    print(f"cur model_name: {model_path_full}")
    base_model_name = model_path_full.split('/')[-1]

    # ---------------------------
    # Split
    # ---------------------------
    full_train_data, test_data = create_problem_based_splits(
        structured_labeled_data,
        random_seed=2025,
        train_pids_per_dataset=dpo_train_ids,
        test_pids_per_dataset=dpo_val_ids,
        leftover_pids_per_dataset=leftover_pids_per_dataset,
    )

    print(f"\n📊 划分后数据集大小:\n  - 完整训练集: {len(full_train_data)} 条\n  - 完整测试集: {len(test_data)} 条")

    # ---------------------------
    # Fixed-ratio sampling (2:4:4)
    # ---------------------------
    N_SAMPLES = int(51200/3)
    ratio_dict = {
        'parsing': 1 if args.preference_reason == 'parsing' else 0,
        'inconsistent': 1 if args.preference_reason == 'inconsistent' else 0,
        'fair comparsion': 1 if args.preference_reason == 'fair' else 0,
        
    }

        
    sampled_train_data = sample_training_data_balanced(
        full_train_data,
        n_samples=N_SAMPLES,
        ratios=ratio_dict,
        seed=2025,
    )

    VAL_sampling_ratio = {
        'parsing': 1,
        'inconsistent': 1,
        'fair comparsion': 1,  # keep the exact key spelling used in your data
    }
    sampled_test_data = sample_training_data_balanced(
        test_data,
        n_samples=32*20,
        ratios=VAL_sampling_ratio,
        seed=2025,
    )

    summarize_distribution(sampled_train_data, "采样后训练分布 (reason & dataset)")

    # ---------------------------
    # HF/TRL setup
    # ---------------------------
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer
    from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel

    class CustomDPOTrainer(DPOTrainer):
            
        def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
            print(f"\n--- [CustomDPOTrainer] Custom Evaluation at Step {self.state.global_step} ---")
            
            all_metrics = {}
            weighted_loss_components = []
    
            for reason in ['parsing', 'inconsistent', 'fair comparsion']:
                print(f"\n  ▶️  Evaluating category: '{reason}'...")
    
                # 步骤 1: 带缓存的数据集过滤
                if reason not in self.reason_dataset_cache:
                    print(f"  (Filtering and caching dataset for '{reason}' for the first time...)")
                    self.reason_dataset_cache[reason] = self.eval_dataset.filter(
                        lambda example: example['reason'] == reason,
                        num_proc=self.dataset_num_proc
                    )
                reason_dataset = self.reason_dataset_cache[reason]
    
                assert len(reason_dataset) != 0, f'''reason dataset is none after filtering'''
    
                # 步骤 2: 对该子集进行采样
                # 注意：datasets 对象不能直接用 random.sample，需要用 .shuffle().select()
                
                sampled_subset = reason_dataset
                
                print(f"  (Using a subsample of {len(sampled_subset)} examples for evaluation)")

                reason_sanitized = reason.replace(' ', '_')
                current_metric_prefix = f"eval/{reason_sanitized}"
                # DPOTrainer 返回的损失指标键名是 'loss'，加上前缀后变为 'eval/category/loss'
                loss_key_for_reason = f"{current_metric_prefix}_loss"
    
                # 步骤 3: 调用父类的 evaluate 方法进行评估
                reason_metrics = super().evaluate(
                    eval_dataset=sampled_subset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=current_metric_prefix # 使用我们定义的前缀
                )
                
                # 步骤 4: 收集指标并计算加权损失部分
                # reason_metrics 字典里是类似 'eval_loss': 0.5 这样的键
                # ### 修改点 2: 使用正确的键名检查和获取损失，并简化指标收集 ###
                if loss_key_for_reason in reason_metrics:
                    reason_loss = reason_metrics[loss_key_for_reason]
                    reason_weight = 1
                    
                    print(f"  (Category '{reason}' Loss: {reason_loss:.4f}, Weight: {reason_weight})")
                    
                    weighted_loss_components.append(reason_loss * reason_weight)
                    
                    # 直接将带有正确前缀的指标更新到总字典中，更简洁且不会出错
                    all_metrics.update(reason_metrics)
                else:
                     print(f"  ⚠️  Warning: Loss key '{loss_key_for_reason}' not found in metrics for reason '{reason}' existing key: {reason_metrics.keys()}.")
                     all_metrics.update(reason_metrics) # 即使没有loss，也更新其他指标
                self.custom_metrics_history[reason].append(reason_metrics)

    
            # 步骤 5: 计算最终的加权损失
            if weighted_loss_components:
                # ### 修改点 3: 确保加权损失被正确添加和记录 ###
                final_weighted_loss = sum(weighted_loss_components)
                all_metrics['eval_weighted_loss'] = final_weighted_loss
                print(f"\n  ✅ Calculated final eval_weighted_loss: {final_weighted_loss:.4f}")
            else:
                # 如果由于某种原因没有任何类别被成功评估，则提供一个默认值以避免KeyError
                # 并发出警告
                print("\n  ❌ ERROR: Could not calculate weighted loss. No categories were evaluated successfully.")
                all_metrics['eval_weighted_loss'] = float('inf') # 使用一个很大的值，表示这是一个糟糕的结果

            self.log(all_metrics)
            
            if self.control.should_training_stop:  
                print(f"Early stopping triggered at step {self.state.global_step}")  
            else:
                print(f'Cur early stop state: {self.control.should_training_stop} keep training')
                
            return all_metrics

    lora_rank = 128
    lora_alpha = lora_rank * 2
    dpo_beta = 0.5

    base_model_path = f"/localdisk/models/{base_model_name}"

    if args.short_context:
        long_short_context_str = 'short'
    elif args.super_short_context:
        long_short_context_str = 'spuer_short'
    else:
        long_short_context_str = 'long'

    if args.train_from == 'scratch':
        sft_adapter_path = None
        output_root = f"/localdisk/kxh230002/DPO_abduction_ckp/{base_model_name}_dpo_lora_{lora_rank}_{lora_alpha}_{dpo_beta}_simple_fixed_{long_short_context_str}{args.postfix}"
    else:  # sft
        sft_adapter_path = f"/localdisk/kxh230002/SFT_abduction_ckp/{base_model_name}_sft_lora_{lora_rank}_{lora_alpha}_curriculum/checkpoints/final_adapter"
        assert os.path.exists(sft_adapter_path), f"path {sft_adapter_path} does not exist; run SFT first"
        output_root = f"/localdisk/kxh230002/DPO_abduction_ckp/{base_model_name}_sft_dpo_lora_{lora_rank}_{lora_alpha}_{dpo_beta}_simple_fixed_{long_short_context_str}{args.postfix}"

    print(f"Currently train from {args.train_from}")

    # 4-bit quant
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.train_from == 'scratch':
        model = prepare_model_for_kbit_training(model)
    else:
        # attach SFT adapter as trainable, and also load a frozen reference adapter
        model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True, adapter_name='sft')
        model.load_adapter(sft_adapter_path, adapter_name='reference')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Datasets → apply chat template to prompt only
    train_ds = Dataset.from_list(sampled_train_data)
    eval_ds  = Dataset.from_list(sampled_test_data)

    def to_tokenized_chatml(example):
        example["prompt"] = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        return example

    formatted_train_ds = train_ds.map(to_tokenized_chatml)
    formatted_eval_ds  = eval_ds.map(to_tokenized_chatml)

    # LoRA config (for scratch case)
    peft_config = None
    if args.train_from == 'scratch':
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    # Training args
    per_device_train_batch_size = 4 if use_large_model else 8
    gradient_accumulation_steps = 8 if use_large_model else 4

    train_epoch_count  = 1
    eval_step_freq     = 50

    common_cfg = dict(
        output_dir=os.path.join(output_root, "checkpoints"),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        beta=dpo_beta,
        num_train_epochs=train_epoch_count,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=eval_step_freq,
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
    )

    if args.train_from == 'scratch':
        training_args = DPOConfig(**common_cfg)
    else:
        training_args = DPOConfig(
            **common_cfg,
            model_adapter_name="sft",
            ref_adapter_name="reference",
        )

    # Trainer (standard TRL DPOTrainer)
    if args.train_from == 'scratch':
        trainer = CustomDPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=formatted_train_ds,
            eval_dataset=formatted_eval_ds,
            peft_config=peft_config,
        )
    else:
        trainer = CustomDPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=formatted_train_ds,
            eval_dataset=formatted_eval_ds,
        )

    # Train & save adapter
    trainer.custom_metrics_history = {'parsing': [],
                                      'inconsistent': [],
                                      'fair comparsion': []}
    trainer.reason_dataset_cache = {}
    trainer.train()

    final_adapter_dir = os.path.join(output_root, 'checkpoints', "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    trainer.save_model(final_adapter_dir)
    print(f"✅ LoRA adapter saved to {final_adapter_dir}")

    history = trainer.state.log_history
    save_history_path = os.path.join(output_root, 'train_history.json')
    with open(save_history_path, 'w') as f:
        json.dump(history, f)
    print(f"📁 Training history saved to {save_history_path}")
