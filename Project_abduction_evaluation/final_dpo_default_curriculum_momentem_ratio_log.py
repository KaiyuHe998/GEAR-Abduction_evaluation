# final train with momentum (uses latest sample_training_data_balanced)

import os
import json
import random
import copy
import argparse
from typing import *
from collections import defaultdict

from global_variables import *
from dowhen import do

# ---------------------------
# Data IO
# ---------------------------
preference_label_data_path = None
assert preference_label_data_path is not None, f'''Please use final_rl_data_gen.py to generate rl data first! And specify your path'''
with open(preference_label_data_path, 'r') as f:
    structured_labeled_data = json.load(f)

with open('your held_out_id file path', 'r') as f:
    leftover_pids_per_dataset = json.load(f)

with open('your train_id file path', 'r') as f:
    dpo_train_ids = json.load(f)

with open('your validation file path', 'r') as f:
    dpo_val_ids = json.load(f)


# ---------------------------
# Split helpers
# ---------------------------
def print_split_headcounts(structured_data: Dict[str, Any]) -> None:
    for ds_name, io_counts in structured_data.items():
        pid_set = set()
        for io_count, problems in io_counts.items():
            pid_set.update(problems.keys())
        print(f"🔍 Data set '{ds_name}' sample count: {len(pid_set)}")


def create_problem_based_splits(
    structured_data: Dict[str, Any],
    random_seed: int = 2025,
    train_pids_per_dataset: Optional[Dict[str, List[str]]] = None,
    test_pids_per_dataset: Optional[Dict[str, List[str]]] = None,
    leftover_pids_per_dataset: Optional[Dict[str, List[str]]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Your latest splitter (verbatim behavior)."""
    rng = random.Random(random_seed)
    print_split_headcounts(structured_data)

    use_external_split = (
        train_pids_per_dataset is not None and
        test_pids_per_dataset is not None and
        leftover_pids_per_dataset is not None
    )

    if use_external_split:
        print("⚡ using predefined problem split (train/test/leftover)")
        train_pids_per_dataset = {k: set(v) for k, v in train_pids_per_dataset.items()}
        test_pids_per_dataset = {k: set(v) for k, v in test_pids_per_dataset.items()}
    else:
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
        print(f"ℹ️ all train size (min): {min_train}, all test size (min): {min_test}")

        train_pids_per_dataset = {}
        test_pids_per_dataset  = {}
        leftover_local = {}
        for ds, pids in unique_per_dataset.items():
            rng.shuffle(pids)
            train_pids = set(rng.sample(pids, min_train))
            remaining = [pid for pid in pids if pid not in train_pids]
            test_pids  = set(rng.sample(remaining, min_test))
            leftover   = [pid for pid in pids if pid not in train_pids and pid not in test_pids]
            train_pids_per_dataset[ds] = train_pids
            test_pids_per_dataset[ds]  = test_pids
            leftover_local[ds]         = leftover
        leftover_pids_per_dataset = leftover_local

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
                        # context shortening (same behavior as your original)
                        if args.short_context:
                            keep_context_index = [0, 1]
                            for index, context_message in enumerate(context):
                                if index in keep_context_index:
                                    continue
                                if context_message['role'] == 'user':
                                    context_message['content'] = (
                                        'Invent a brand-new hypothesis based on a fundamentally different principle '
                                        'from any of your previous hypotheses, while still matching every given '
                                        'input-output pair. Use exactly the same output format as before.'
                                    )
                        if args.super_short_context:
                            new_context = [context[0], context[1]]
                            new_instruction = {
                                "role": "user",
                                "content": "Please generate as much hypothesis as possible (one at a time), "
                                           "and make sure new generated hypothesis is based on a fundamentally "
                                           "different principle from any of your previous hypotheses."
                            }
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

    print("✅ train test split finished")
    return train_dataset, test_dataset


def summarize_distribution(rows: List[Dict], title: str) -> None:
    print(f"📊 {title}")
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
    """LATEST balanced sampler by reason & dataset."""
    rng = random.Random(seed)

    # Bucket by reason→dataset
    by_reason_ds: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in train_data:
        by_reason_ds[rec['reason']][rec['dataset']].append(rec)

    # Integer quotas per reason, exact sum == n_samples
    reasons = list(ratios.keys())
    tot_ratio = sum(ratios.values())
    quotas: Dict[str, int] = {}
    used = 0
    for i, r in enumerate(reasons):
        if i < len(reasons) - 1:
            q = int(n_samples * (ratios[r] / tot_ratio)) if tot_ratio > 0 else 0
            quotas[r] = q
            used += q
        else:
            quotas[r] = max(0, n_samples - used)

    print("🎯 expected data sample ratio", quotas)

    sampled: List[Dict] = []

    for reason in reasons:
        per_ds = by_reason_ds.get(reason, {})
        if not per_ds:
            print(f"  ⚠️ no data found for preference:'{reason}' skip.")
            continue

        target = quotas[reason]
        avail_total = sum(len(v) for v in per_ds.values())
        if avail_total <= target:
            print(f"  ⚠️ preference type '{reason}' not enough: only {avail_total} samples, expected {target} samples, select all samples.")
            for ds, rows in per_ds.items():
                rows_copy = rows[:]
                rng.shuffle(rows_copy)
                sampled.extend(rows_copy)
            continue

        ds_keys = list(per_ds.keys())
        base = target // len(ds_keys)
        remainder = target - base * len(ds_keys)

        alloc = {ds: min(len(per_ds[ds]), base) for ds in ds_keys}
        order = ds_keys[:]
        rng.shuffle(order)
        rem = remainder
        for ds in order:
            if rem == 0:
                break
            cap = len(per_ds[ds])
            if alloc[ds] < cap:
                alloc[ds] += 1
                rem -= 1
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
                break

        need = target - sum(alloc.values())
        if need > 0:
            for ds in order:
                if need == 0:
                    break
                cap = len(per_ds[ds])
                spare = cap - alloc[ds]
                if spare > 0:
                    take = min(spare, need)
                    alloc[ds] += take
                    need -= take

        for ds in ds_keys:
            k = alloc[ds]
            rows = per_ds[ds]
            picks = rng.sample(rows, k) if k <= len(rows) else rows[:]
            sampled.extend(picks)

        print(f"  ✅ '{reason}' select {target} samples (based on dataset: "
              + ", ".join([f"{ds}:{alloc[ds]}" for ds in sorted(ds_keys)]) + ")")

    rng.shuffle(sampled)
    print(f"✅ sample finished, in total {len(sampled)} samples.")
    return sampled


# ---------------------------
# Monitoring wrapper
# ---------------------------
class MonitoringDataLoader:
    def __init__(self, original_dataloader, print_distribution_func):
        self.original_dataloader = original_dataloader
        self.print_distribution_func = print_distribution_func

    def __iter__(self):
        print(f"🔍 [DataLoader Monitor] Starting new iteration...")
        self.print_distribution_func(self.original_dataloader)
        return iter(self.original_dataloader)

    def __len__(self):
        return len(self.original_dataloader)

    def __getattr__(self, name):
        return getattr(self.original_dataloader, name)


def print_dataloader_distribution(dataloader, sample_batches: int = None):
    from collections import defaultdict
    reason_counts = defaultdict(int)
    total_samples = 0
    print(f"📊 [DataLoader Distribution Check - Cur Epoch]")
    if hasattr(dataloader.dataset, '__getitem__'):
        dataset_size = len(dataloader.dataset)
        print(f"  Checking all {dataset_size} samples...")
        for i in range(dataset_size):
            try:
                sample = dataloader.dataset[i]
                if 'reason' in sample:
                    reason_counts[sample['reason']] += 1
                    total_samples += 1
            except Exception:
                continue
    if total_samples > 0:
        print(f"  Total samples checked: {total_samples}")
        print(f"  Distribution by reason:")
        for reason, count in reason_counts.items():
            percentage = (count / total_samples) * 100
            print(f"    - '{reason}': {count} samples ({percentage:.1f}%)")
    else:
        print(f"  ⚠️ No samples found or 'reason' field missing")
    return dict(reason_counts)


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
    parser.add_argument("-uf", "--update_frequency", type=int, required=True, help="update_frequency used when update ratio of the training data")
    parser.add_argument("-p", "--postfix", type=str, required=False, default='', help="postfix for the model")
    parser.add_argument("-t", "--train_from", type=str, required=True, choices=["scratch", "sft"],
                        help="train from sft or train from scratch")
    args = parser.parse_args()
    update_frequency = int(args.update_frequency)
    print(f'Currently update training ratio in every {update_frequency} epoch')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path_full = model_mapping_dict[args.model_name]
    print(f"Train model with short context? {args.short_context}")
    print(f"Train model with super short context? {args.super_short_context}")
    assert not (args.short_context and args.super_short_context), \
        "super_short_context and short_context cannot be both True"

    large_model = ['Qwen/Qwen2.5-72B-Instruct', 'meta-llama/Llama-3.3-70B-Instruct']
    small_model = ['Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', 'microsoft/NextCoder-7B']
    use_large_model = model_path_full in large_model
    assert model_path_full in large_model or model_path_full in small_model

    print(f"cur model_name: {model_path_full}")
    base_model_name = model_path_full.split('/')[-1]

    # ---------------------------
    # Split (external splits respected)
    # ---------------------------
    full_train_data, test_data = create_problem_based_splits(
        structured_labeled_data,
        random_seed=2025,
        train_pids_per_dataset=dpo_train_ids,
        test_pids_per_dataset=dpo_val_ids,
        leftover_pids_per_dataset=leftover_pids_per_dataset,
    )
    print(f"📊 Dataset sizes after split:\n  - Train set (full): {len(full_train_data)} samples\n  - Test set (full): {len(test_data)} samples")

    # ---------------------------
    # Initial pool sampling (LATEST sampler)
    #   You may change N_SAMPLES and ratios as needed; defaults match prior large-scale training
    # ---------------------------
    N_SAMPLES = 300000
    SAMPLING_RATIOS = {
        'parsing': 1,
        'inconsistent': 1,
        'fair comparsion': 1,  # Note: keep original key spelling
    }
    sampled_train_data = sample_training_data_balanced(
        full_train_data,
        n_samples=N_SAMPLES,
        ratios=SAMPLING_RATIOS,
        seed=2025,
    )

    # The validation set also uses the latest sampler (smaller sample size for faster evaluation)
    VAL_sampling_ratio = {
        'parsing': 1,
        'inconsistent': 1,
        'fair comparsion': 1,
    }
    sampled_test_data = sample_training_data_balanced(
        test_data,
        n_samples=100000,
        ratios=VAL_sampling_ratio,
        seed=2025,
    )

    summarize_distribution(sampled_train_data, "Post-sampling train distribution (reason & dataset)")
    summarize_distribution(sampled_test_data,  "Post-sampling validation distribution (reason & dataset)")

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

    # LoRA & DPO basic configuration
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
        sft_adapter_path = None # Don't need this
        output_root = f"your root save path/{base_model_name}_dpo_lora_{lora_rank}_{lora_alpha}_{dpo_beta}_simple_fixed_{long_short_context_str}{args.postfix}"
        assert not output_root.startswith('your root save path'), f'''Please specify your model save path'''
    
    else:  # sft
        sft_adapter_path = None # Use this only if you use sft, in our paper we do not use sft
        assert os.path.exists(sft_adapter_path), f"path {sft_adapter_path} does not exist; run SFT first"
        output_root = None

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
        model = PeftModel.from_pretrained(model, sft_adapter_path,
                                          is_trainable=True, adapter_name='sft')
        model.load_adapter(sft_adapter_path, adapter_name='reference')

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Build datasets (apply chat template to "prompt" only)
    train_ds = Dataset.from_list(sampled_train_data)
    eval_ds  = Dataset.from_list(sampled_test_data)

    def to_tokenized_chatml(example):
        example["prompt"] = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        return example

    # train_ds = train_ds.train_test_split(test_size = 0.01, seed = 2025)['test'] # For debug
    # eval_ds = eval_ds.train_test_split(test_size = 0.01, seed = 2025)['test']   # comment out this line if you set a small eval sample size and want to random draw eval samples during evaluation

    formatted_train_ds = train_ds.map(to_tokenized_chatml)
    formatted_eval_ds  = eval_ds.map(to_tokenized_chatml)

    # LoRA config (for scratch)
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

    # ---------------------------
    # Custom DPOTrainer with momentum curriculum (probe-loss on TRAIN set)
    # ---------------------------
    class CustomDPOTrainer(DPOTrainer):
        def __init__(
            self,
            *args,
            reasons_to_eval: List[str],
            eval_sample_size: int,
            n_samples_per_epoch: int,
            reason_init_weight: Dict[str, float],
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            self.reasons_to_eval = reasons_to_eval
            self.eval_sample_size = eval_sample_size
            self.custom_metrics_history = {r: [] for r in reasons_to_eval}
            self.reason_init_weight = reason_init_weight
            self.n_samples_per_epoch = n_samples_per_epoch

            # map: reason -> indices (on the full formatted training dataset)
            self.indices_by_reason = defaultdict(list)
            self.train_dataset_source = self.train_dataset
            for i, example in enumerate(self.train_dataset_source):
                self.indices_by_reason[example['reason']].append(i)
            for reason, indices in self.indices_by_reason.items():
                print(f"  - Found {len(indices)} examples for reason '{reason}'")
            print("✅ Index mapping complete.")

            self.tokenized_eval_dataset_source = self.eval_dataset
            if not self.tokenized_eval_dataset_source:
                raise ValueError("CustomDPOTrainer requires an eval_dataset to be provided.")
            if 'reason' not in self.tokenized_eval_dataset_source.column_names:
                raise ValueError("The 'reason' column was not found in the tokenized evaluation dataset.")

            self.dataset_num_proc = getattr(self.args, "dataset_num_proc", None)
            self.ratio_history: List[Dict[str, Any]] = []

            # ---- momentum/EMA state ----
            self.probe_size_per_reason = eval_sample_size
            self.ema_alpha = 0.1
            self.weight_floor = 0.8
            self.weight_cap   = 1.2
            self.eps_for_inv  = 0.1
            self.momentum_cap = 0.03
            self.update_per_epoch = update_frequency
            
            # Use probe samples from the EVAL set
            self._probe_cache_eval: Dict[str, Dataset] = {}
            self._ema_by_reason: Dict[str, float] = {}
            self._last_ema_by_reason: Dict[str, float] = {}
            
            for reason in self.reasons_to_eval:
                ds_reason_eval = self.tokenized_eval_dataset_source.filter(
                    lambda ex: ex['reason'] == reason,
                    num_proc=self.dataset_num_proc
                )
                if len(ds_reason_eval) == 0:
                    print(f"[Probe] Warning: no EVAL samples for reason '{reason}'")
                    continue
                k = min(self.probe_size_per_reason, len(ds_reason_eval))
                probe = ds_reason_eval.shuffle(seed=42).select(range(k))
                self._probe_cache_eval[reason] = probe
                print(f"[Probe] reason '{reason}': cached {len(probe)} EVAL-probe examples.")

            # optional: keep your hook
            _ = do('train_dataloader = self.get_train_dataloader()').when(
                self._inner_training_loop, "epoch_dataloader = train_dataloader"
            )

            # caches for eval split-by-reason
            self._bootstrap_probe_baseline()
            self.reason_dataset_cache = {}


        def _bootstrap_probe_baseline(self):
            print("[Bootstrap] Running initial probe baseline...")
            for reason in self.reasons_to_eval:
                loss0 = self._compute_probe_loss_on_eval(reason, step_seed=0)
                if loss0 is None:
                    continue
                self._ema_by_reason[reason] = float(loss0)
                self._last_ema_by_reason[reason] = float(loss0)
                rs = reason.replace(' ', '_')
                step = int(self.state.global_step or 0)  # NEW: explicitly include step
                self.log({
                    f"bootstrap/loss/{rs}": float(loss0),
                    f"ema/{rs}": float(loss0),
                    f"momentum/{rs}": 0.0,
                    "step": step,
                })
        def _compute_probe_loss_on_eval(self, reason: str, step_seed: int = 0) -> Optional[float]:
            if reason not in self._probe_cache_eval:
                return None
            probe_src = self._probe_cache_eval[reason]
            probe_ds = probe_src.shuffle(seed=step_seed).select(
                range(min(len(probe_src), self.probe_size_per_reason))
            )
            reason_sanitized = reason.replace(' ', '_')
            prefix = f"probe_eval/{reason_sanitized}"
            metrics = super(CustomDPOTrainer, self).evaluate(
                eval_dataset=probe_ds, metric_key_prefix=prefix
            )
            loss_key = f"{prefix}_loss"
            if loss_key not in metrics:
                print(f"[Probe] Loss key '{loss_key}' not found for reason '{reason}'. Keys: {list(metrics.keys())}")
                return None
        
            return float(metrics[loss_key])

        def _ratios_from_momentum(self) -> Dict[str, float]:
            ratios = {}
            ema_snapshot = {}
            cur_epoch = int(self.state.epoch or 0)
        
            for reason in self.reasons_to_eval:
                cur_loss = self._compute_probe_loss_on_eval(reason, step_seed=cur_epoch)  # changed here
                if cur_loss is None:
                    ratios[reason] = 1.0
                    continue
        
                if reason not in self._ema_by_reason:
                    self._ema_by_reason[reason] = cur_loss
                    self._last_ema_by_reason[reason] = cur_loss
                    momentum = 0.0
                else:
                    prev_ema = self._ema_by_reason[reason]
                    new_ema = self.ema_alpha * cur_loss + (1 - self.ema_alpha) * prev_ema
                    # momentum = max(0.0, min(self.momentum_cap, new_ema - self._last_ema_by_reason[reason])) # inverse loss 
                    momentum = max(0.0, min(self.momentum_cap, self._last_ema_by_reason[reason] - new_ema))
                    self._last_ema_by_reason[reason] = self._ema_by_reason[reason]
                    self._ema_by_reason[reason] = new_ema
        
                ema_snapshot[reason] = {"probe_loss": cur_loss, "ema": self._ema_by_reason[reason], "momentum": momentum}
        
                # Also log for easier debugging/analysis
                rs = reason.replace(' ', '_')
                step = int(self.state.global_step or 0)
                self.log({
                            f"momentum/{rs}": float(momentum),
                            f"ema/{rs}": float(self._ema_by_reason[reason]),
                            "step": step,  # include step
                        })
        
                inv = self.eps_for_inv + momentum  # try inverse later
                ratios[reason] = inv
        
            s = sum(ratios.values()) or 1.0
            for r in ratios:
                ratios[r] = max(self.weight_floor, min(self.weight_cap, ratios[r] / s * len(ratios)))
            print(f"[Momentum] snapshot per reason: {ema_snapshot}")
            print(f"[Momentum] raw ratios: {ratios}")

            step = int(self.state.global_step or 0)
            epoch = int(self.state.epoch or 0)
            for r, v in ratios.items():
                rs = r.replace(' ', '_')
                self.log({f"ratio/{rs}": float(v), "step": step})
        
            # NEW: append to ratio_history (including ema/momentum snapshot for auditing)
            self.ratio_history.append({
                "source": "momentum_update",
                "step": step,
                "epoch": epoch,
                "ratios": {k: float(v) for k, v in ratios.items()},
                "ema_snapshot": {k: {"probe_loss": float(v["probe_loss"]), "ema": float(v["ema"]), "momentum": float(v["momentum"])}
                                 for k, v in ema_snapshot.items()}
            })
            return ratios

        # def training_step(self, model, inputs, num_items_in_batch=None):  
        #     """Override training_step to decode and print detailed batch data"""  
        # 
        #     # Control print frequency to avoid excessive logs  
        #     if self.state.global_step % 10 == 0:  # print every 10 steps  
        #         print(f"🔍 [Training Step {self.state.global_step}] Detailed Batch Data:")  
        # 
        #         batch_size = len(inputs.get('prompt_input_ids', []))  
        #         print(f"  - Batch size: {batch_size}")  
        # 
        #         # Print details for each sample  
        #         for i in range(min(batch_size, 2)):  # only show first 2 samples  
        #             print(f"  📝 Sample {i+1}:")  
        # 
        #             # Print metadata  
        #             if 'reason' in inputs:  
        #                 print(f"    - Reason: {inputs['reason'][i]}")  
        #             if 'problem_id' in inputs:  
        #                 print(f"    - Problem ID: {inputs['problem_id'][i]}")  
        #             if 'dataset' in inputs:  
        #                 print(f"    - Dataset: {inputs['dataset'][i]}")  
        # 
        #             # Decode prompt  
        #             if 'prompt_input_ids' in inputs:  
        #                 prompt_ids = inputs['prompt_input_ids'][i]  
        #                 # Remove padding tokens  
        #                 if 'prompt_attention_mask' in inputs:  
        #                     mask = inputs['prompt_attention_mask'][i]  
        #                     prompt_ids = prompt_ids[mask.bool()]  
        # 
        #                 decoded_prompt = self.processing_class.decode(prompt_ids, skip_special_tokens=True)  
        # 
        #                 print(f"    - Prompt ({len(decoded_prompt)}): {decoded_prompt}")  
        # 
        #             # Decode chosen response  
        #             if 'chosen_input_ids' in inputs:  
        #                 chosen_ids = inputs['chosen_input_ids'][i]  
        #                 if 'chosen_attention_mask' in inputs:  
        #                     mask = inputs['chosen_attention_mask'][i]  
        #                     chosen_ids = chosen_ids[mask.bool()]  
        # 
        #                 decoded_chosen = self.processing_class.decode(chosen_ids, skip_special_tokens=True)  
        #                 print(f"    - Chosen ({len(decoded_chosen)}): {decoded_chosen}")  
        # 
        #             # Decode rejected response  
        #             if 'rejected_input_ids' in inputs:  
        #                 rejected_ids = inputs['rejected_input_ids'][i]  
        #                 if 'rejected_attention_mask' in inputs:  
        #                     mask = inputs['rejected_attention_mask'][i]  
        #                     rejected_ids = rejected_ids[mask.bool()]  
        # 
        #                 decoded_rejected = self.processing_class.decode(rejected_ids, skip_special_tokens=True)  
        #                 print(f"    - Rejected ({len(decoded_rejected)}): {decoded_rejected}")  
        # 
        #             # Print tensor shapes  
        #             print(f"    - Shapes: prompt={inputs['prompt_input_ids'][i].shape}, "  
        #                   f"chosen={inputs['chosen_input_ids'][i].shape}, "  
        #                   f"rejected={inputs['rejected_input_ids'][i].shape}")  
        #         print('--------------------------------------------step end!--------------------------------------------')
        #         print('--------------------------------------------step end!--------------------------------------------')
        #         print('--------------------------------------------step end!--------------------------------------------')
        # 
        #     # Call parent training_step  
        #     return super().training_step(model, inputs, num_items_in_batch)

        def get_train_dataloader(self):
            if self.state.epoch is None:
                print(f"🚀 [Adaptive Curriculum Engine] Preparing data for Epoch {1}...")
            else:
                print(f"🚀 [Adaptive Curriculum Engine] Preparing data for Epoch {int(self.state.epoch)}...")

            if self.state.epoch is None or int(self.state.epoch) == 0:
                current_ratios = dict(self.reason_init_weight)
                self._last_ratios = current_ratios
                print(f"  - Warmup ratios (init): {current_ratios}")
            else:
                if int(self.state.epoch) % self.update_per_epoch == 0:
                    current_ratios = self._ratios_from_momentum()
                    self._last_ratios = current_ratios  # remember this round
                else:
                    # If not at an update point, reuse the ratios from the previous round
                    current_ratios = getattr(self, "_last_ratios", dict(self.reason_init_weight))

            self.early_stopping_weights = current_ratios

            if self.state.epoch is None:
                sampler = random.Random(self.args.seed)
            else:
                sampler = random.Random(self.args.seed + int(self.state.epoch))

            total_ratio = sum(current_ratios.values()) or 1.0
            sampled_indices = []
            sampling_composition = defaultdict(int)

            for reason, ratio in current_ratios.items():
                num_to_sample = int(self.n_samples_per_epoch * (ratio / total_ratio))
                available_indices = self.indices_by_reason.get(reason, [])
                if not available_indices or num_to_sample == 0:
                    continue
                if len(available_indices) <= num_to_sample:
                    sampled_for_reason = available_indices
                else:
                    sampled_for_reason = sampler.sample(available_indices, num_to_sample)
                sampled_indices.extend(sampled_for_reason)
                sampling_composition[reason] = len(sampled_for_reason)

            sampler.shuffle(sampled_indices)
            total_sampled = len(sampled_indices)
            comp = ", ".join([f"'{r}': {c}" for r, c in sampling_composition.items()])
            print(f"  - Sampled a total of {total_sampled} examples for this epoch.")
            print(f"  - Composition: {{ {comp} }}")

            epoch_dataset = self.train_dataset_source.select(sampled_indices)
            self.train_dataset = epoch_dataset
            step = int(self.state.global_step or 0)
            epoch = int(self.state.epoch or 0)
            for r, v in current_ratios.items():
                rs = r.replace(' ', '_')
                self.log({f"ratio/{rs}": float(v), "step": step})
            for r, c in sampling_composition.items():
                rs = r.replace(' ', '_')
                self.log({f"epoch_comp/{rs}": int(c), "step": step})
            self.log({"epoch_total_sampled": int(total_sampled), "epoch": epoch, "step": step})
        
            self.ratio_history.append({
                "source": "epoch_sampling",
                "step": step,
                "epoch": epoch,
                "ratios": {k: float(v) for k, v in current_ratios.items()},
                "composition": {k: int(v) for k, v in sampling_composition.items()},
                "total_sampled": int(total_sampled),
            })

            standard_dataloader = super().get_train_dataloader()
            monitoring_dataloader = MonitoringDataLoader(standard_dataloader, print_dataloader_distribution)
            return monitoring_dataloader

        # Keep the evaluation structure that evaluates by reason on the validation set
        def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
            print(f"--- [CustomDPOTrainer] Probe-Only Evaluation at Step {self.state.global_step} ---")
            all_metrics = {}
            weighted_loss_components = []
        
            for reason in self.reasons_to_eval:
                print(f"  ▶️  Evaluating category (probe from EVAL): '{reason}'...")
                if reason not in self._probe_cache_eval:
                    print(f"  ⚠️  Skipped '{reason}' (no probe cached)")
                    continue
        
                # Use the cached eval-probe; shuffle with step seed for a bit of randomness
                src = self._probe_cache_eval[reason]
                k = min(self.eval_sample_size, len(src))
                probe_subset = src.shuffle(seed=self.state.global_step).select(range(k))
        
                reason_sanitized = reason.replace(' ', '_')
                current_metric_prefix = f"eval_probe/{reason_sanitized}"
                loss_key_for_reason = f"{current_metric_prefix}_loss"
        
                reason_metrics = super().evaluate(
                    eval_dataset=probe_subset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=current_metric_prefix
                )
        
                # Log all probe metrics (entering log_history)
                self.custom_metrics_history[reason].append(reason_metrics)
        
                if loss_key_for_reason in reason_metrics:
                    reason_loss = reason_metrics[loss_key_for_reason]
                    reason_weight = self.early_stopping_weights.get(reason, 1.0)
                    print(f"  (Category '{reason}' Loss: {reason_loss:.4f}, Weight: {reason_weight})")
                    weighted_loss_components.append(reason_loss * reason_weight)
                    all_metrics.update(reason_metrics)
                else:
                    print(f"  ⚠️  Warning: Loss key '{loss_key_for_reason}' not found. Keys: {reason_metrics.keys()}")
                    all_metrics.update(reason_metrics)
        
            if weighted_loss_components:
                final_weighted_loss = sum(weighted_loss_components)
                all_metrics['eval_weighted_loss'] = final_weighted_loss
                print(f"  ✅ Calculated final eval_weighted_loss (probe-only): {final_weighted_loss:.4f}")
            else:
                print("❌ ERROR: Could not calculate weighted loss. No categories were evaluated successfully.")
                all_metrics['eval_weighted_loss'] = float('inf')
        
            # Write the aggregate once more
            self.log(all_metrics)
            return all_metrics

    # ---------------------------
    # Training args
    # ---------------------------
    per_device_train_batch_size = 4 if use_large_model else 8
    gradient_accumulation_steps = 8 if use_large_model else 4

    train_epoch_count  = 40
    train_batch_sample = 32 * 40          # per epoch sample up to 2560 examples into training (from pooled train)
    eval_step_freq     = 100
    eval_sample_size   = 300

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
        save_steps=1000,
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

    reason_init_weight = {
        'parsing': 1,
        'inconsistent': 1,
        'fair comparsion': 1,
    }

    # ---------------------------
    # Init trainer
    # ---------------------------
    if args.train_from == 'scratch':
        trainer = CustomDPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=formatted_train_ds,
            eval_dataset=formatted_eval_ds,
            peft_config=peft_config,
            reasons_to_eval=['parsing', 'inconsistent', 'fair comparsion'],
            eval_sample_size=eval_sample_size,
            reason_init_weight=reason_init_weight,
            n_samples_per_epoch=train_batch_sample,
        )
    else:
        trainer = CustomDPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=formatted_train_ds,
            eval_dataset=formatted_eval_ds,
            reasons_to_eval=['parsing', 'inconsistent', 'fair comparsion'],
            eval_sample_size=eval_sample_size,
            reason_init_weight=reason_init_weight,
            n_samples_per_epoch=train_batch_sample,
        )

    # ---------------------------
    # Train & save
    # ---------------------------
    trainer.train()

    final_adapter_dir = os.path.join(output_root, 'checkpoints', "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    trainer.save_model(final_adapter_dir)
    print(f"✅ LoRA adapter saved to {final_adapter_dir}")

    history = trainer.state.log_history

    history.append({
        "custom_metrics_history": trainer.custom_metrics_history,
        "ratio_history": trainer.ratio_history,  # NEW
    })
    save_history_path = os.path.join(output_root, 'train_history.json')
    with open(save_history_path, 'w') as f:
        json.dump(history, f)
    print(f"📁 Training history saved to {save_history_path}")
