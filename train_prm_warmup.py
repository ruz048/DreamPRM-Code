# pip install transformers accelerate datasets scipy peft
import json, math, os, argparse, re
from typing import Dict, Any, List
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------
# Parse command-line arguments
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Warm-up training for PRM (Process Reward Model)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B",
                        help="Base model name (e.g., Qwen/Qwen2.5-Coder-7B, Qwen/Qwen3-Coder-14B)")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--meta_data", type=str, required=True,
                        help="Path to meta dataset JSON file (with code_list and graded_list)")
    parser.add_argument("--output_dir", type=str, default="/data1/ruiyi/funprm/checkpoints_warmup",
                        help="Output directory for warm-up checkpoints")
    parser.add_argument("--text_max_len", type=int, default=4096,
                        help="Maximum text length for tokenization")
    parser.add_argument("--use_bce", action="store_true",
                        help="Use BCEWithLogits loss (for scores in [0,1]); otherwise use MSE")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size for warm-up")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps for warm-up")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for warm-up")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer (L2 regularization)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for reward head")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of warm-up epochs")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--lora_r", type=int, default=4,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj",
                        help="Comma-separated list of target modules for LoRA (e.g., q_proj,v_proj)")
    return parser.parse_args()

args = parse_args()

# -----------------------
# Set random seed
# -----------------------
def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# -----------------------
# Config
# -----------------------
MODEL_NAME = args.model_name
DATA_JSON = args.train_data
TEXT_MAX_LEN = args.text_max_len
USE_BCE = args.use_bce

# Set device based on local rank for distributed training
import torch.distributed as dist

# Initialize distributed training if launched with torchrun
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    # torchrun sets these environment variables
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    is_distributed = True
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0
    print(f"[Rank {local_rank}] Using device: {device} (Physical GPU: {torch.cuda.current_device()})")
elif dist.is_available() and dist.is_initialized():
    # Already initialized by some other means
    local_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    is_distributed = True
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0
    print(f"[Rank {local_rank}] Using device: {device}")
else:
    # Single GPU or CPU mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_distributed = False
    local_rank = 0
    world_size = 1
    is_main_process = True
    print(f"Using device: {device}")

# -----------------------
# Prompt template
# -----------------------
PROMPT_TMPL = """<task>
Platform: {platform}
Question ID: {question_id}
Title: {title}
</task>

<problem>
{content}
</problem>

<starter_code>
{starter}
</starter_code>""".strip()

def build_text(ex: Dict[str, Any]) -> str:
    return PROMPT_TMPL.format(
        platform=ex.get("platform",""),
        question_id=ex.get("question_id",""),
        title=ex.get("question_title",""),
        content=ex.get("question_content",""),
        starter=ex.get("starter_code",""),
    )

# -----------------------
# Extract base question ID (without step suffix)
# -----------------------
def get_base_question_id(question_id: str) -> str:
    """
    Extract the base question ID without the step number.
    E.g., '1873_A-0' -> '1873_A', '1873_A-1' -> '1873_A', '1873_A-2' -> '1873_A'
    """
    # Match pattern: everything before the last '-' followed by a digit
    match = re.match(r'^(.+)-(\d+)$', question_id)
    if match:
        return match.group(1)
    return question_id  # If no match, return as is

def get_step_number(question_id: str) -> int:
    """
    Extract the step number from question_id.
    E.g., '1873_A-0' -> 0, '1873_A-1' -> 1, '1873_A-2' -> 2
    """
    match = re.match(r'^.+-(\d+)$', question_id)
    if match:
        return int(match.group(1))
    return -1  # If no match, return -1

# -----------------------
# Load training data (exclude last-step data, same as BLO stage)
# -----------------------
import random

# Load training dataset
with open(DATA_JSON, "r", encoding="utf-8") as f:
    train_raw = json.load(f)

if is_main_process:
    print(f"Loaded {len(train_raw)} training samples from {DATA_JSON}")

# Group data by base question ID to identify last steps
question_groups = defaultdict(list)
for idx, ex in enumerate(train_raw):
    question_id = ex.get("question_id", "")
    base_id = get_base_question_id(question_id)
    step_num = get_step_number(question_id)

    score = float(ex.get("pass@1", 0.0))
    data_item = {
        "text": build_text(ex),
        "score": score,
        "question_id": question_id,
        "base_id": base_id,
        "step_num": step_num,
        "original_idx": idx
    }
    question_groups[base_id].append(data_item)

# Sort each group by step number
for base_id in question_groups:
    question_groups[base_id].sort(key=lambda x: x["step_num"])

# Process training data - exclude last steps (same as BLO)
train_rows = []
for base_id, items in question_groups.items():
    if len(items) == 0:
        continue
    elif len(items) == 1:
        # If only one step, still use it for training
        train_rows.append({
            "text": items[0]["text"],
            "score": items[0]["score"],
        })
    else:
        # Only use non-last steps for training
        for item in items[:-1]:
            train_rows.append({
                "text": item["text"],
                "score": item["score"],
            })

if is_main_process:
    print(f"Train samples (excluding last steps): {len(train_rows)}")

# -----------------------
# Load meta dataset from separate JSON file (same as BLO stage)
# -----------------------
META_JSON = args.meta_data

with open(META_JSON, "r", encoding="utf-8") as f:
    meta_raw = json.load(f)

if is_main_process:
    print(f"Loaded {len(meta_raw)} entries from meta dataset: {META_JSON}")

# Process meta dataset: extract code_list and graded_list
meta_rows = []
for entry in meta_raw:
    code_list = entry.get("code_list", [])
    graded_list = entry.get("graded_list", [])

    # Ensure code_list and graded_list have the same length
    if len(code_list) != len(graded_list):
        if is_main_process:
            print(f"Warning: Skipping entry with mismatched code_list ({len(code_list)}) and graded_list ({len(graded_list)})")
        continue

    # Create a sample for each code in code_list with corresponding grade
    for code, grade in zip(code_list, graded_list):
        # Create a modified entry where the code from code_list becomes the starter_code
        entry_with_code = entry.copy()
        entry_with_code["starter_code"] = code

        # Build text from the entry (using the code as starter_code)
        text = build_text(entry_with_code)

        # Convert boolean grade to binary (True -> 1.0, False -> 0.0)
        score = 1.0 if grade else 0.0

        meta_rows.append({
            "text": text,
            "score": score
        })

if is_main_process:
    print(f"Meta samples (after extracting from code_list): {len(meta_rows)}")

# Combine train and meta data for warm-up
data_rows = train_rows + meta_rows

if is_main_process:
    print(f"Total samples for warm-up (train + meta): {len(data_rows)}")

# Create dataset
dataset = Dataset.from_list(data_rows)

# -----------------------
# Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess(ex):
    enc = tokenizer(
        ex["text"],
        max_length=TEXT_MAX_LEN,
        truncation=True,
        add_special_tokens=True,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label": float(ex["score"])
    }

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# -----------------------
# Collator (pad to batch max length)
# -----------------------
@dataclass
class RMDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []

        for f in features:
            ids = f["input_ids"]
            am  = f["attention_mask"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attn.append(am + [0] * pad_len)
            labels.append(f["label"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

collator = RMDataCollator(pad_token_id=tokenizer.pad_token_id)

# -----------------------
# Model: Causal LM + scalar head on last token (per sequence)
# -----------------------
class RewardScalarModel(nn.Module):
    def __init__(self, base_name: str, use_bce: bool = True, dropout: float = 0.1,
                 lora_config: LoraConfig = None):
        super().__init__()
        # Load model in float32 for LoRA fine-tuning
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float32,
        )

        # Apply LoRA if config is provided
        if lora_config is not None:
            if is_main_process:
                print("Applying LoRA to base model...")
            self.lm = get_peft_model(self.lm, lora_config)
            if is_main_process:
                self.lm.print_trainable_parameters()
        else:
            # If no LoRA, enable gradient computation for all parameters (full fine-tuning)
            for param in self.lm.parameters():
                param.requires_grad = True

        hidden = self.lm.config.hidden_size
        # Smaller reward head with bias
        self.reward_head = nn.Linear(hidden, 1, bias=True)
        self.use_bce = use_bce

    def forward(self, input_ids, attention_mask):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        h = out.hidden_states[-1]  # [B, T, H]

        # Pool at last non-pad token per sequence
        lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_idx = torch.arange(h.size(0), device=h.device)
        last_h = h[batch_idx, lengths]  # [B, H]

        # Get logits and apply sigmoid
        logits = self.reward_head(last_h).squeeze(-1)  # [B]
        scores = torch.sigmoid(logits)  # [B], values in [0, 1]
        return scores

# -----------------------
# Create model
# -----------------------
if is_main_process:
    print("Loading reward model for LoRA fine-tuning...")

# Create LoRA configuration
lora_target_modules = args.lora_target_modules.split(',')
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=lora_target_modules,
    bias="none",
)

if is_main_process:
    print(f"LoRA Configuration:")
    print(f"  - Rank (r): {args.lora_r}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Dropout: {args.lora_dropout}")
    print(f"  - Target modules: {lora_target_modules}")

# Create reward model with LoRA
reward_model = RewardScalarModel(MODEL_NAME, use_bce=USE_BCE, dropout=args.dropout,
                                lora_config=lora_config)

# Print model info
if is_main_process:
    total_params = sum(p.numel() for p in reward_model.parameters())
    trainable_params = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Reward head parameters: {sum(p.numel() for p in reward_model.reward_head.parameters()):,}")

# -----------------------
# Data loaders
# -----------------------
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if is_distributed:
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collator)
else:
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator)

# -----------------------
# Loss functions
# -----------------------
# Label smoothing to prevent overconfident predictions
LABEL_SMOOTHING = 0.1

def apply_label_smoothing(labels, smoothing=LABEL_SMOOTHING):
    """Apply label smoothing: move labels slightly away from 0 and 1"""
    return labels * (1 - smoothing) + smoothing * 0.5

# Since model outputs sigmoid (0-1), use BCE (not BCEWithLogits) or MSE
if USE_BCE:
    base_loss_fn = nn.BCELoss(reduction="none")  # For sigmoid outputs
else:
    base_loss_fn = nn.MSELoss(reduction="none")

# -----------------------
# Warm-up training
# -----------------------
if is_main_process:
    print(f"\n{'='*60}")
    print(f"Starting Warm-up Training: {args.epochs} epoch(s)")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    if is_distributed:
        print(f"Distributed training: {world_size} GPUs")
        effective_batch_size = args.batch_size * world_size * args.gradient_accumulation_steps
        print(f"Effective batch size: {effective_batch_size}")
    print(f"{'='*60}\n")

# Move model to device
reward_model.to(device)

# Wrap model with DDP if in distributed mode
if is_distributed:
    from torch.nn.parallel import DistributedDataParallel as DDP
    reward_model = DDP(reward_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

reward_model.train()

# Create optimizer
trainable_params = [p for p in reward_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)

# Calculate total steps for scheduler (accounting for gradient accumulation)
total_steps = (args.epochs * len(dataloader)) // args.gradient_accumulation_steps
warmup_steps = 0  # No LR warmup needed for warm-up stage

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_training_steps=total_steps,
    num_warmup_steps=warmup_steps
)

global_step = 0
grad_accum_step = 0

for epoch in range(args.epochs):
    # Set epoch for DistributedSampler to ensure proper shuffling
    if is_distributed and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

    epoch_loss = 0.0
    epoch_steps = 0

    # Only show progress bar on main process
    if is_main_process:
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    else:
        progress_bar = dataloader

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Apply label smoothing
        labels_smoothed = apply_label_smoothing(labels)

        # Forward pass
        scores = reward_model(input_ids, attention_mask)

        # Compute loss (uniform weighting)
        loss_vector = base_loss_fn(scores, labels_smoothed)
        loss = torch.mean(loss_vector)

        # Scale loss by accumulation steps
        loss = loss / args.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Track metrics (use unscaled loss for logging)
        epoch_loss += loss.item() * args.gradient_accumulation_steps
        epoch_steps += 1
        grad_accum_step += 1

        # Perform optimizer step after accumulating gradients
        if grad_accum_step % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Update progress bar (only on main process)
        if is_main_process and hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{epoch_loss/epoch_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'grad_accum': f'{grad_accum_step % args.gradient_accumulation_steps}/{args.gradient_accumulation_steps}'
            })

    # Handle remaining gradients at end of epoch
    if grad_accum_step % args.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    avg_epoch_loss = epoch_loss / epoch_steps

    # Synchronize loss across all processes
    if is_distributed:
        loss_tensor = torch.tensor([avg_epoch_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_epoch_loss = loss_tensor.item()

    if is_main_process:
        print(f"Epoch {epoch+1}/{args.epochs} completed - Average Loss: {avg_epoch_loss:.6f}")

# Synchronize all processes before saving
if is_distributed:
    dist.barrier()

# -----------------------
# Save checkpoint
# -----------------------
if is_main_process:
    print(f"\n{'='*60}")
    print(f"Warm-up training completed after {global_step} steps")
    print(f"Saving checkpoint...")

    os.makedirs(args.output_dir, exist_ok=True)

    # Get the underlying model (unwrap DDP if necessary)
    model_to_save = reward_model.module if isinstance(reward_model, torch.nn.parallel.DistributedDataParallel) else reward_model

    # Save the model state dict
    checkpoint_path = os.path.join(args.output_dir, "warmup_checkpoint.pt")
    torch.save(model_to_save.state_dict(), checkpoint_path)

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"{'='*60}\n")

if is_distributed:
    dist.barrier()
    # Clean up distributed process group
    if is_main_process:
        print("Cleaning up distributed training...")
    dist.destroy_process_group()

print("Warm-up training completed successfully!")
