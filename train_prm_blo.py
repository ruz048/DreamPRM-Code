# pip install transformers accelerate datasets scipy betty-optimizer peft
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

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

# -----------------------
# Parse command-line arguments
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train PRM (Process Reward Model) with Bilevel Optimization")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B",
                        help="Base model name (e.g., Qwen/Qwen2.5-Coder-7B, Qwen/Qwen3-Coder-14B)")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--meta_data", type=str, required=True,
                        help="Path to meta dataset JSON file (with code_list and graded_list)")
    parser.add_argument("--output_dir", type=str, default="/data1/ruiyi/funprm/checkpoints_blo",
                        help="Output directory for model checkpoints")
    parser.add_argument("--text_max_len", type=int, default=4096,
                        help="Maximum text length for tokenization")
    parser.add_argument("--use_bce", action="store_true",
                        help="Use BCEWithLogits loss (for scores in [0,1]); otherwise use MSE")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate for main model (reduced to prevent overfitting)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for main model optimizer (L2 regularization)")
    parser.add_argument("--meta_lr", type=float, default=5e-6,
                        help="Learning rate for meta-network")
    parser.add_argument("--meta_weight_decay", type=float, default=0.01,
                        help="Weight decay for meta-network optimizer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for reward head")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--train_iters", type=int, default=750,
                        help="Number of training iterations")
    parser.add_argument("--warmup_iters", type=int, default=250,
                        help="Number of warmup iterations")
    parser.add_argument("--valid_step", type=int, default=5,
                        help="Validation frequency")
    parser.add_argument("--unroll_steps", type=int, default=1,
                        help="Number of unroll steps for implicit differentiation")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline without meta-learning")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--precision", type=str, default="fp32",
                        help="Training precision (fp32, fp16, bf16)")
    parser.add_argument("--strategy", type=str, default="default",
                        help="Training strategy (default, distributed, zero)")
    parser.add_argument("--lora_r", type=int, default=4,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj",
                        help="Comma-separated list of target modules for LoRA (e.g., q_proj,v_proj)")
    parser.add_argument("--warmup_checkpoint", type=str, default=None,
                        help="Path to warm-up checkpoint file (e.g., /path/to/warmup_checkpoint.pt)")
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

device = "cuda" if torch.cuda.is_available() else "cpu"

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
# Load training data (exclude last-step data)
# -----------------------
import random

# Load training dataset
with open(DATA_JSON, "r", encoding="utf-8") as f:
    train_raw = json.load(f)

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

# Process training data - exclude last steps
train_rows = []
sample_idx = 0
for base_id, items in question_groups.items():
    if len(items) == 0:
        continue
    elif len(items) == 1:
        # If only one step, still use it for training
        train_rows.append({
            "text": items[0]["text"],
            "score": items[0]["score"],
            "sample_idx": sample_idx
        })
        sample_idx += 1
    else:
        # Only use non-last steps for training
        for item in items[:-1]:
            train_rows.append({
                "text": item["text"],
                "score": item["score"],
                "sample_idx": sample_idx
            })
            sample_idx += 1

print(f"Train samples (excluding last steps): {len(train_rows)}")

# -----------------------
# Load meta dataset from separate JSON file
# -----------------------
META_JSON = args.meta_data

with open(META_JSON, "r", encoding="utf-8") as f:
    meta_raw = json.load(f)

print(f"Loaded {len(meta_raw)} entries from meta dataset: {META_JSON}")

# Process meta dataset: extract code_list and graded_list
meta_rows = []
for entry in meta_raw:
    code_list = entry.get("code_list", [])
    graded_list = entry.get("graded_list", [])

    # Ensure code_list and graded_list have the same length
    if len(code_list) != len(graded_list):
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

print(f"Meta samples (after extracting from code_list): {len(meta_rows)}")

# Create datasets
train_ds = Dataset.from_list(train_rows)
meta_ds = Dataset.from_list(meta_rows)

print(f"After split - Train: {len(train_ds)}, Meta: {len(meta_ds)}")

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
    result = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label": float(ex["score"])
    }
    # Preserve sample_idx for training data (not present in meta/val)
    if "sample_idx" in ex:
        result["sample_idx"] = ex["sample_idx"]
    return result

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
meta_ds = meta_ds.map(preprocess, remove_columns=meta_ds.column_names)

# -----------------------
# Collator (pad to batch max length)
# -----------------------
@dataclass
class RMDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        sample_indices = []

        for f in features:
            ids = f["input_ids"]
            am  = f["attention_mask"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attn.append(am + [0] * pad_len)
            labels.append(f["label"])
            # Collect sample indices if present (only for training data)
            if "sample_idx" in f:
                sample_indices.append(f["sample_idx"])

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

        # Add sample indices if present (only for training batches)
        if sample_indices:
            batch["sample_indices"] = torch.tensor(sample_indices, dtype=torch.long)

        return batch

collator = RMDataCollator(pad_token_id=tokenizer.pad_token_id)

# -----------------------
# Model: Causal LM + scalar head on last token (per sequence)
# -----------------------
class RewardScalarModel(nn.Module):
    def __init__(self, base_name: str, use_bce: bool = True, dropout: float = 0.1,
                 lora_config: LoraConfig = None):
        super().__init__()
        # Load model in float32 for LoRA fine-tuning with Betty
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float32,
        )

        # Apply LoRA if config is provided
        if lora_config is not None:
            print("Applying LoRA to base model...")
            self.lm = get_peft_model(self.lm, lora_config)
            self.lm.print_trainable_parameters()
        else:
            # If no LoRA, enable gradient computation for all parameters (full fine-tuning)
            for param in self.lm.parameters():
                param.requires_grad = True

        hidden = self.lm.config.hidden_size
        # No dropout to avoid architecture mismatch between training and eval
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

        # No dropout applied
        logits = self.reward_head(last_h).squeeze(-1)  # [B]

        # Apply sigmoid to bound output between 0 and 1
        scores = torch.sigmoid(logits)  # [B], values in [0, 1]
        return scores

# -----------------------
# Meta-weight table (per-sample learnable weights)
# -----------------------
class MetaWeightTable(nn.Module):
    """
    Instance-level meta-weight table that assigns a learnable weight parameter to each training sample.
    Uses clipping to avoid extreme values instead of softmax normalization.

    Reference: Instance Table strategy from the paper, where each training example x∈Dtr
    is associated with a learnable scalar weight αx.
    """
    def __init__(self, num_samples, alpha_min=0.0, alpha_max=2.0):
        super(MetaWeightTable, self).__init__()
        # Initialize weights to 1.0 (uniform weighting at start)
        self.weight_table = nn.Parameter(torch.ones(num_samples))
        self.num_samples = num_samples
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def forward(self, indices):
        """
        Args:
            indices: tensor of sample indices [B]
        Returns:
            weights: tensor of clipped weights [B] in range [alpha_min, alpha_max]
        """
        # Look up raw weights for the given indices
        raw_weights = self.weight_table[indices]  # [B]
        # Apply clipping to avoid extreme values
        weights = torch.clamp(raw_weights, self.alpha_min, self.alpha_max)
        return weights

# -----------------------
# Create models
# -----------------------
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

print(f"LoRA Configuration:")
print(f"  - Rank (r): {args.lora_r}")
print(f"  - Alpha: {args.lora_alpha}")
print(f"  - Dropout: {args.lora_dropout}")
print(f"  - Target modules: {lora_target_modules}")

# Create reward model with LoRA
reward_model = RewardScalarModel(MODEL_NAME, use_bce=USE_BCE, dropout=args.dropout,
                                lora_config=lora_config)

# Load warm-up checkpoint if provided
if args.warmup_checkpoint is not None and os.path.exists(args.warmup_checkpoint):
    print(f"\n{'='*60}")
    print(f"Loading warm-up checkpoint from: {args.warmup_checkpoint}")
    checkpoint = torch.load(args.warmup_checkpoint, map_location='cpu')
    reward_model.load_state_dict(checkpoint)
    print(f"Warm-up checkpoint loaded successfully!")
    print(f"{'='*60}\n")
elif args.warmup_checkpoint is not None:
    print(f"\nWARNING: Warm-up checkpoint not found at {args.warmup_checkpoint}")
    print(f"Starting BLO training from scratch...\n")

# Print model info
total_params = sum(p.numel() for p in reward_model.parameters())
trainable_params = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print(f"\nModel Parameters:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {frozen_params:,}")
print(f"Reward head parameters: {sum(p.numel() for p in reward_model.reward_head.parameters()):,}")

meta_net = MetaWeightTable(
    num_samples=len(train_ds)
)

# Print meta-net info
print(f"Meta-weight table initialized with {len(train_ds)} learnable weights")

# -----------------------
# Data loaders
# -----------------------
from torch.utils.data import DataLoader

train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=collator)
meta_loader = DataLoader(meta_ds, shuffle=True, batch_size=args.batch_size, collate_fn=collator)

# -----------------------
# Optimizers
# -----------------------
# Use Adam instead of AdamW for better compatibility with Betty framework
# Only optimize parameters that require gradients (LoRA adapters + reward head)
trainable_params = [p for p in reward_model.parameters() if p.requires_grad]
print(f"Optimizer will update {len(trainable_params)} parameter tensors")

optimizer = torch.optim.Adam(
    trainable_params,
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_training_steps=args.train_iters,
    num_warmup_steps=args.warmup_iters
)

meta_optimizer = torch.optim.Adam(
    meta_net.parameters(),
    lr=args.meta_lr,
    weight_decay=args.meta_weight_decay,
)

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
# Define bilevel optimization problems
# -----------------------
class Finetune(ImplicitProblem):
    """Lower-level problem: Fine-tune the reward model with weighted loss"""

    def trainable_parameters(self):
        """
        Override to return only parameters that have requires_grad=True.
        Since we freeze the base LM and only train the reward head,
        this ensures the optimizer only updates the reward head parameters.
        """
        return [p for p in self.module.parameters() if p.requires_grad]

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        sample_indices = batch.get("sample_indices", None)  # Get sample indices

        # Apply label smoothing to prevent overconfident predictions
        labels_smoothed = apply_label_smoothing(labels)

        # Forward pass through reward model (outputs sigmoid scores in [0, 1])
        scores = self.module(input_ids, attention_mask)  # [B], values in [0, 1]

        # Compute per-sample loss with smoothed labels
        loss_vector = base_loss_fn(scores, labels_smoothed)  # [B]

        if args.baseline:
            # Baseline: uniform weighting
            loss = torch.mean(loss_vector)
        else:
            # Meta-learning: weighted by meta-weight-table using sample indices
            weights = self.reweight(sample_indices)  # [B], values in [0, 1]
            loss = torch.mean(weights * loss_vector)
        #print('Finetune loss: ', loss.item())
        return loss

class Reweight(ImplicitProblem):
    """Upper-level problem: Learn to reweight samples to minimize meta loss"""

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward pass through reward model (trained in lower-level)
        # Model outputs sigmoid scores in [0, 1]
        scores = self.finetune(input_ids, attention_mask)  # [B]

        # Compute loss on meta dataset
        if USE_BCE:
            loss = F.binary_cross_entropy(scores, labels)  # For sigmoid outputs
        else:
            loss = F.mse_loss(scores, labels)

        #print('Reweight loss: ', loss.item())
        return loss

# -----------------------
# Validation (monitoring only, no early stopping)
# -----------------------
class PRMEngine(Engine):
    @torch.no_grad()
    def validation(self):
        # Compute loss on train set
        train_loss = 0.0
        '''
        print(f"\n[Step {self.global_step}] Computing train set loss...")
        for batch in tqdm(train_loader, desc="Train Set", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            scores = self.finetune(input_ids, attention_mask)

            if USE_BCE:
                loss = F.binary_cross_entropy(scores, labels)
            else:
                loss = F.mse_loss(scores, labels)

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        '''
        # Compute loss on meta set
        meta_loss = 0.0
        print(f"[Step {self.global_step}] Computing meta set loss...")
        for batch in tqdm(meta_loader, desc="Meta Set", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            scores = self.finetune(input_ids, attention_mask)

            if USE_BCE:
                loss = F.binary_cross_entropy(scores, labels)
            else:
                loss = F.mse_loss(scores, labels)

            meta_loss += loss.item()

        meta_loss = meta_loss / len(meta_loader)

        # Print losses
        print(f"Step {self.global_step} - Train Loss: {train_loss:.6f} | Meta Loss: {meta_loss:.6f}")

        # Save the newest checkpoint (delete previous if exists)
        if not args.baseline:
            os.makedirs(args.output_dir, exist_ok=True)

            # Delete previous checkpoint to save storage
            prev_step = self.global_step - args.valid_step
            if prev_step > 0:
                prev_reward_model = os.path.join(args.output_dir, f"reward_model_step_{prev_step}.pt")
                prev_meta_net = os.path.join(args.output_dir, f"meta_net_step_{prev_step}.pt")
                if os.path.exists(prev_reward_model):
                    os.remove(prev_reward_model)
                if os.path.exists(prev_meta_net):
                    os.remove(prev_meta_net)

            # Save current checkpoint
            torch.save(self.finetune.state_dict(),
                      os.path.join(args.output_dir, f"reward_model_step_{self.global_step}.pt"))
            torch.save(self.reweight.state_dict(),
                      os.path.join(args.output_dir, f"meta_net_step_{self.global_step}.pt"))
            print(f"Saved checkpoint: reward_model_step_{self.global_step}.pt")

        return {
            "train_loss": train_loss,
            "meta_loss": meta_loss,
        }

# -----------------------
# Configure bilevel optimization
# -----------------------
engine_config = EngineConfig(
    train_iters=args.train_iters,
    valid_step=args.valid_step,
    strategy=args.strategy,
)

finetune_config = Config(
    type="darts",  # Use DARTS for implicit differentiation
    precision=args.precision,
    retain_graph=True,
    log_step=args.valid_step,
    unroll_steps=args.unroll_steps,
    gradient_clipping=args.grad_clip,  # Add gradient clipping
    gradient_accumulation=args.gradient_accumulation_steps,  # Add gradient accumulation
)

reweight_config = Config(
    type="darts",
    precision=args.precision,
    gradient_accumulation=args.gradient_accumulation_steps,
    log_step=args.valid_step,
)

finetune = Finetune(
    name="finetune",
    module=reward_model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_data_loader=train_loader,
    config=finetune_config,
)

reweight = Reweight(
    name="reweight",
    module=meta_net,
    optimizer=meta_optimizer,
    train_data_loader=meta_loader,
    config=reweight_config,
)

# -----------------------
# Set up dependencies
# -----------------------
if args.baseline:
    problems = [finetune]
    u2l, l2u = {}, {}
else:
    problems = [reweight, finetune]
    u2l = {reweight: [finetune]}  # Upper depends on lower
    l2u = {finetune: [reweight]}  # Lower depends on upper

dependencies = {"l2u": l2u, "u2l": u2l}

# -----------------------
# Create engine and run
# -----------------------
engine = PRMEngine(
    config=engine_config,
    problems=problems,
    dependencies=dependencies,
)

print("Starting training...")
engine.run()
print("Training completed!")
if not args.baseline:
    print(f"\nFinal checkpoint saved:")
    print(f"  - Location: {args.output_dir}")
    print(f"  - Files: reward_model_step_{args.train_iters}.pt, meta_net_step_{args.train_iters}.pt")
