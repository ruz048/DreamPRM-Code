"""
PRM Evaluation Script for BLO (Bilevel Optimization) Models
This script loads a BLO-trained PRM model and evaluates it on test data.
"""

import json
import os
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import sys

from lcb_runner.benchmarks.code_generation import split_string_to_prefix_list

# -----------------------
# Parse command-line arguments
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLO-trained PRM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B",
                        help="Base model name (must match training)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to BLO checkpoint directory")
    parser.add_argument("--checkpoint_step", type=int, default=50,
                        help="Which step checkpoint to load (default: 50)")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to evaluation data JSON file")
    parser.add_argument("--output_json", type=str, default="prm_blo_results.json",
                        help="Output path for evaluation results JSON")
    parser.add_argument("--text_max_len", type=int, default=4096,
                        help="Maximum text length for tokenization")
    parser.add_argument("--use_bce", action="store_true",
                        help="Use BCE mode (must match training configuration)")
    parser.add_argument("--use_prefix_eval", action="store_true", default=False,
                        help="Split code by functions and evaluate each prefix (typical PRM)")
    parser.add_argument("--prefix_aggregation", type=str, default="last",
                        choices=["min", "mean", "last", "product", "logodds"],
                        help="How to aggregate prefix scores")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (must match training)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    # LoRA parameters (must match training configuration)
    parser.add_argument("--lora_r", type=int, default=4,
                        help="LoRA rank (must match training)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (must match training)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout (must match training)")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,v_proj,k_proj,o_proj",
                        help="Comma-separated list of target modules for LoRA")
    return parser.parse_args()

args = parse_args()

# -----------------------
# Configuration
# -----------------------
MODEL_NAME = args.model_name
CHECKPOINT_DIR = args.checkpoint_dir
CHECKPOINT_STEP = args.checkpoint_step
DATA_JSON = args.eval_data
OUTPUT_JSON = args.output_json
TEXT_MAX_LEN = args.text_max_len
USE_BCE = args.use_bce
USE_PREFIX_EVAL = args.use_prefix_eval
PREFIX_AGGREGATION = args.prefix_aggregation
DROPOUT = args.dropout

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Prompt template (must match training)
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

def build_text(ex: Dict[str, Any], code: str = "") -> str:
    """Build the input text for PRM scoring."""
    # Extract base question_id without step suffix
    question_id = ex.get("question_id", "")
    if "-" in question_id:
        question_id = question_id.rsplit("-", 1)[0]

    base = PROMPT_TMPL.format(
        platform=ex.get("platform", ""),
        question_id=question_id,
        title=ex.get("question_title", ""),
        content=ex.get("question_content", ""),
        starter=code,
    )
    return base


# -----------------------
# Model Definition (must match training)
# -----------------------
class RewardScalarModel(nn.Module):
    def __init__(self, base_name: str, use_bce: bool = True, dropout: float = 0.1,
                 lora_config: LoraConfig = None):
        super().__init__()
        # Load model in float32 (matching training)
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )

        # Apply LoRA if config is provided
        if lora_config is not None:
            self.lm = get_peft_model(self.lm, lora_config)

        hidden = self.lm.config.hidden_size
        # No dropout to match training (dropout parameter kept for compatibility)
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


def load_blo_model(checkpoint_dir: str, base_model_name: str, step: int) -> Tuple[RewardScalarModel, AutoTokenizer]:
    """Load the BLO-trained PRM model from checkpoint."""
    print(f"Loading BLO model from {checkpoint_dir}...")

    # Initialize tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create LoRA configuration (must match training)
    lora_target_modules = args.lora_target_modules.split(',')
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )

    print("LoRA Configuration:")
    print(f"  - Rank (r): {args.lora_r}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Dropout: {args.lora_dropout}")
    print(f"  - Target modules: {lora_target_modules}")

    # Create model structure with LoRA
    print("Initializing model structure with LoRA...")
    model = RewardScalarModel(base_model_name, use_bce=USE_BCE, dropout=DROPOUT,
                             lora_config=lora_config)

    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"reward_model_step_{step}.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Available checkpoints:")
        for f in os.listdir(checkpoint_dir):
            if f.startswith("reward_model_step_") and f.endswith(".pt"):
                print(f"  - {f}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load to CPU first, then move to device
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    # Check if this is a full training state (with optimizer, scheduler, etc.)
    if isinstance(checkpoint, dict) and "module" in checkpoint:
        # This is a DDP checkpoint with full training state
        print("Detected DDP checkpoint format, extracting model weights...")
        state_dict = checkpoint["module"]
    else:
        # Assume it's already a state dict
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP wrapping)
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict.keys()):
        print("Removing 'module.' prefix from keys...")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "", 1)  # Remove only the first occurrence
            new_state_dict[new_key] = value
        state_dict = new_state_dict

    # Load state dict (model is already on the correct device from device_map="auto")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint from step {step}")
    if missing:
        print(f"Warning: Missing keys: {len(missing)}")
        print(f"  First few: {missing[:5]}")
    if unexpected:
        print(f"Warning: Unexpected keys: {len(unexpected)}")
        print(f"  First few: {unexpected[:5]}")

    # Ensure reward_head is on the same device as the LM
    print("Moving reward_head to model device...")
    model_device = next(model.lm.parameters()).device
    model.reward_head = model.reward_head.to(model_device)

    model.eval()
    print(f"Model loaded successfully on device: {model_device}!")
    return model, tokenizer


def prm_aggregate_score(probs):
    """
    Compute the PRM aggregation score for a list or array of step probabilities.

    Args:
        probs (list or np.ndarray): Sequence of predicted probabilities (each in (0, 1)).

    Returns:
        float: Aggregated PRM score (sum of log-odds).
    """
    probs = np.clip(np.array(probs), 1e-8, 1 - 1e-8)  # avoid log(0)
    logits = np.log(probs / (1 - probs))
    return np.sum(logits)


def score_code_with_prm(model: RewardScalarModel, tokenizer: AutoTokenizer,
                        problem: Dict[str, Any], code: str) -> float:
    """Score a single code solution using the PRM."""
    text = build_text(problem, code)

    # Tokenize
    enc = tokenizer(
        text,
        max_length=TEXT_MAX_LEN,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # Get model device
    model_device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)

    # Get score (model outputs sigmoid, already in [0, 1])
    with torch.no_grad():
        score = model(input_ids=input_ids, attention_mask=attention_mask)
        score = score.item()  # Already in [0, 1] due to sigmoid

    return float(score)


def score_codes_batch(model: RewardScalarModel, tokenizer: AutoTokenizer,
                     texts: List[str], batch_size: int = 16) -> List[float]:
    """
    Score multiple code solutions in batches for efficiency.

    Args:
        model: The PRM model
        tokenizer: The tokenizer
        texts: List of pre-built text inputs (problem + code)
        batch_size: Batch size for inference

    Returns:
        List of scores corresponding to each text
    """
    all_scores = []
    model_device = next(model.parameters()).device

    # Process in batches with progress bar
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Batch inference"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        batch_enc = tokenizer(
            batch_texts,
            max_length=TEXT_MAX_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = batch_enc["input_ids"].to(model_device)
        attention_mask = batch_enc["attention_mask"].to(model_device)

        # Get scores (model outputs sigmoid, already in [0, 1])
        with torch.no_grad():
            scores = model(input_ids=input_ids, attention_mask=attention_mask)
            scores = scores.cpu().numpy()

        all_scores.extend(scores.tolist())

    return all_scores


def score_code_prefixes_with_prm(model: RewardScalarModel, tokenizer: AutoTokenizer,
                                 problem: Dict[str, Any], code: str,
                                 aggregation: str = "last") -> Tuple[float, List[float]]:
    """
    Score code by splitting it into function prefixes and evaluating each prefix.
    This is the typical PRM approach where we evaluate intermediate reasoning steps.

    Args:
        model: The PRM model
        tokenizer: The tokenizer
        problem: Problem dictionary
        code: The code to score
        aggregation: How to aggregate scores ("min", "mean", "last", "product")

    Returns:
        (aggregated_score, list_of_prefix_scores)
    """
    # Split code into prefixes
    prefixes = split_string_to_prefix_list(code)

    if not prefixes:
        # If no prefixes, score the whole code
        score = score_code_with_prm(model, tokenizer, problem, code)
        return score, [score]

    # Score each prefix
    prefix_scores = []
    for prefix in prefixes:
        score = score_code_with_prm(model, tokenizer, problem, prefix)
        prefix_scores.append(score)

    # Aggregate scores
    if aggregation == "min":
        final_score = min(prefix_scores)
    elif aggregation == "mean":
        final_score = np.mean(prefix_scores)
    elif aggregation == "last":
        final_score = prefix_scores[-1]
    elif aggregation == "product":
        final_score = np.prod(prefix_scores)
    elif aggregation == "logodds":
        final_score = prm_aggregate_score(prefix_scores)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return float(final_score), prefix_scores


def evaluate_prm(model: RewardScalarModel, tokenizer: AutoTokenizer,
                eval_data: List[Dict[str, Any]], batch_size: int = 16) -> Dict[str, Any]:
    """
    Evaluate the PRM model on test data using batched processing.

    For each problem with multiple code candidates:
    1. Score each candidate using PRM (in batches)
    2. Select the candidate with highest PRM score
    3. Check if it's correct (graded=True)
    4. Compare with baseline (selecting first candidate)
    5. Track results by difficulty level
    """
    # Group by base question_id
    from collections import defaultdict
    problems = defaultdict(list)
    problem_difficulty = {}  # Maps base_id to difficulty level

    for item in eval_data:
        question_id = item.get("question_id", "")
        if "-" in question_id:
            base_id = question_id.rsplit("-", 1)[0]
        else:
            base_id = question_id
        problems[base_id].append(item)

        # Extract difficulty level (if available)
        if base_id not in problem_difficulty:
            difficulty = item.get("difficulty", "Unknown")
            problem_difficulty[base_id] = difficulty

    # Count total candidates across all code_list entries
    total_candidates = sum(len(item.get("code_list", [])) for item in eval_data)
    print(f"Found {len(problems)} unique problems with {total_candidates} total candidates")
    print(f"Batch size: {batch_size}")

    # STEP 1: Prepare all samples for batched processing
    print("Preparing data for batched processing...")
    all_samples = []

    for base_id, items in problems.items():
        for item in items:
            code_list = item.get("code_list", [])
            graded_list = item.get("graded_list", [])

            # Validate that code_list and graded_list have matching lengths
            if len(code_list) != len(graded_list):
                print(f"Warning: Mismatched lengths for {item.get('question_id')}: "
                      f"code_list={len(code_list)}, graded_list={len(graded_list)}")
                min_len = min(len(code_list), len(graded_list))
                code_list = code_list[:min_len]
                graded_list = graded_list[:min_len]

            for code, is_correct in zip(code_list, graded_list):
                if USE_PREFIX_EVAL:
                    # Split into prefixes
                    prefixes = split_string_to_prefix_list(code)
                    if not prefixes:
                        prefixes = [code]
                else:
                    prefixes = [code]

                all_samples.append({
                    "base_id": base_id,
                    "item": item,
                    "code": code,
                    "is_correct": is_correct,
                    "prefixes": prefixes,
                })

    print(f"Total samples to evaluate: {len(all_samples)}")

    # STEP 2: Build all text inputs for batched inference
    print("Building text inputs...")
    all_texts = []
    text_to_sample_map = []  # Maps text index to (sample_idx, prefix_idx)

    for sample_idx, sample in enumerate(all_samples):
        for prefix_idx, prefix in enumerate(sample["prefixes"]):
            text = build_text(sample["item"], prefix)
            all_texts.append(text)
            text_to_sample_map.append((sample_idx, prefix_idx))

    print(f"Total texts to score: {len(all_texts)}")

    # STEP 3: Batch inference on all texts
    print("Running batched inference...")
    all_scores = score_codes_batch(model, tokenizer, all_texts, batch_size=batch_size)

    # STEP 4: Map scores back to samples and aggregate prefixes
    print("Aggregating scores...")
    for sample in all_samples:
        sample["prefix_scores"] = []

    for text_idx, (sample_idx, prefix_idx) in enumerate(text_to_sample_map):
        all_samples[sample_idx]["prefix_scores"].append(all_scores[text_idx])

    # Aggregate prefix scores for each sample
    for sample in all_samples:
        prefix_scores = sample["prefix_scores"]

        if USE_PREFIX_EVAL and len(prefix_scores) > 1:
            # Aggregate multiple prefix scores
            if PREFIX_AGGREGATION == "min":
                sample["final_score"] = float(np.min(prefix_scores))
            elif PREFIX_AGGREGATION == "mean":
                sample["final_score"] = float(np.mean(prefix_scores))
            elif PREFIX_AGGREGATION == "last":
                sample["final_score"] = prefix_scores[-1]
            elif PREFIX_AGGREGATION == "product":
                sample["final_score"] = float(np.prod(prefix_scores))
            elif PREFIX_AGGREGATION == "logodds":
                sample["final_score"] = prm_aggregate_score(prefix_scores)
            else:
                raise ValueError(f"Unknown aggregation: {PREFIX_AGGREGATION}")
        else:
            # Single score
            sample["final_score"] = prefix_scores[0] if prefix_scores else 0.0

    # STEP 5: Group results by problem and compute metrics
    print("Computing metrics...")
    results = []
    prm_correct = 0
    baseline_avg_correct = 0.0  # Now a float for average correctness
    pass_at_n_correct = 0
    total_problems = 0

    # Track metrics by difficulty level
    difficulty_metrics = defaultdict(lambda: {
        "total": 0,
        "prm_correct": 0,
        "baseline_avg_correct": 0.0,
        "pass_at_n_correct": 0
    })

    for base_id, items in tqdm(problems.items(), desc="Computing problem metrics"):
        if not items:
            continue

        total_problems += 1

        # Get all samples for this problem
        problem_samples = [s for s in all_samples if s["base_id"] == base_id]

        if not problem_samples:
            continue

        # PRM selection: highest score
        best_by_prm = max(problem_samples, key=lambda x: x["final_score"])

        # Baseline: average correctness across all generations
        baseline_avg_correctness = np.mean([s["is_correct"] for s in problem_samples])

        # Pass@N: check if any candidate is correct
        has_any_correct = any(s["is_correct"] for s in problem_samples)

        # Count correct selections
        if best_by_prm["is_correct"]:
            prm_correct += 1
        baseline_avg_correct += baseline_avg_correctness
        if has_any_correct:
            pass_at_n_correct += 1

        # Get difficulty level for this problem
        difficulty = problem_difficulty.get(base_id, "Unknown")

        # Update difficulty-specific metrics
        difficulty_metrics[difficulty]["total"] += 1
        if best_by_prm["is_correct"]:
            difficulty_metrics[difficulty]["prm_correct"] += 1
        difficulty_metrics[difficulty]["baseline_avg_correct"] += baseline_avg_correctness
        if has_any_correct:
            difficulty_metrics[difficulty]["pass_at_n_correct"] += 1

        # Store result
        results.append({
            "base_question_id": base_id,
            "difficulty": difficulty,
            "num_candidates": len(problem_samples),
            "prm_selected": best_by_prm["item"].get("question_id"),
            "prm_score": best_by_prm["final_score"],
            "prm_correct": best_by_prm["is_correct"],
            "baseline_avg_correctness": baseline_avg_correctness,
            "pass_at_n_correct": has_any_correct,
            "all_candidates": [
                {
                    "question_id": s["item"].get("question_id"),
                    "score": s["final_score"],
                    "prefix_scores": s["prefix_scores"],
                    "is_correct": s["is_correct"],
                    "code": s["code"]
                }
                for s in problem_samples
            ]
        })

    # Calculate overall metrics
    prm_accuracy = prm_correct / total_problems if total_problems > 0 else 0
    baseline_accuracy = baseline_avg_correct / total_problems if total_problems > 0 else 0
    pass_at_n_accuracy = pass_at_n_correct / total_problems if total_problems > 0 else 0
    improvement = prm_accuracy - baseline_accuracy

    # Calculate metrics by difficulty
    difficulty_summary = {}
    for difficulty, metrics in difficulty_metrics.items():
        total = metrics["total"]
        if total > 0:
            prm_acc = metrics["prm_correct"] / total
            baseline_acc = metrics["baseline_avg_correct"] / total
            pass_at_n_acc = metrics["pass_at_n_correct"] / total
            diff_improvement = prm_acc - baseline_acc

            difficulty_summary[difficulty] = {
                "total_problems": total,
                "prm_correct": metrics["prm_correct"],
                "baseline_avg_correct": metrics["baseline_avg_correct"],
                "pass_at_n_correct": metrics["pass_at_n_correct"],
                "prm_accuracy": prm_acc,
                "baseline_accuracy": baseline_acc,
                "pass_at_n_accuracy": pass_at_n_acc,
                "improvement": diff_improvement,
                "improvement_pct": diff_improvement / baseline_acc * 100 if baseline_acc > 0 else 0,
            }

    summary = {
        "total_problems": total_problems,
        "prm_correct": prm_correct,
        "baseline_avg_correct": baseline_avg_correct,
        "pass_at_n_correct": pass_at_n_correct,
        "prm_accuracy": prm_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "pass_at_n_accuracy": pass_at_n_accuracy,
        "improvement": improvement,
        "improvement_pct": improvement / baseline_accuracy * 100 if baseline_accuracy > 0 else 0,
        "by_difficulty": difficulty_summary,
    }

    return {
        "summary": summary,
        "results": results,
        "config": {
            "model_name": MODEL_NAME,
            "checkpoint_dir": CHECKPOINT_DIR,
            "checkpoint_step": CHECKPOINT_STEP,
            "use_prefix_eval": USE_PREFIX_EVAL,
            "prefix_aggregation": PREFIX_AGGREGATION if USE_PREFIX_EVAL else None,
        }
    }


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("=" * 80)
    print("BLO PRM Evaluation")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT_DIR} (step {CHECKPOINT_STEP})")
    print(f"Eval data: {DATA_JSON}")
    print(f"Prefix evaluation: {USE_PREFIX_EVAL}")
    if USE_PREFIX_EVAL:
        print(f"Prefix aggregation: {PREFIX_AGGREGATION}")
    print("=" * 80)

    # Load model
    model, tokenizer = load_blo_model(CHECKPOINT_DIR, MODEL_NAME, CHECKPOINT_STEP)

    # Load evaluation data
    print(f"\nLoading evaluation data from {DATA_JSON}...")
    with open(DATA_JSON, "r") as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} evaluation samples")

    # Evaluate
    print("\nStarting evaluation...")
    eval_results = evaluate_prm(model, tokenizer, eval_data, batch_size=args.batch_size)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - OVERALL")
    print("=" * 80)
    summary = eval_results["summary"]
    print(f"Total problems: {summary['total_problems']}")
    print(f"Baseline (avg over all generations) accuracy: {summary['baseline_accuracy']:.2%}")
    print(f"PRM-based selection correct: {summary['prm_correct']} ({summary['prm_accuracy']:.2%})")
    print(f"Pass@N (upper bound) correct: {summary['pass_at_n_correct']} ({summary['pass_at_n_accuracy']:.2%})")
    print(f"PRM improvement over baseline: {summary['improvement']:.2%} ({summary['improvement_pct']:.1f}% relative)")
    print("=" * 80)

    # Print results by difficulty level
    if summary.get("by_difficulty"):
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS - BY DIFFICULTY LEVEL")
        print("=" * 80)

        # Sort difficulty levels for consistent ordering (if numeric or standard names)
        difficulty_levels = sorted(summary["by_difficulty"].keys())

        for difficulty in difficulty_levels:
            diff_metrics = summary["by_difficulty"][difficulty]
            print(f"\n{difficulty}:")
            print(f"  Total problems: {diff_metrics['total_problems']}")
            print(f"  Baseline accuracy: {diff_metrics['baseline_accuracy']:.2%}")
            print(f"  PRM accuracy: {diff_metrics['prm_accuracy']:.2%}")
            print(f"  Pass@N accuracy: {diff_metrics['pass_at_n_accuracy']:.2%}")
            print(f"  PRM improvement: {diff_metrics['improvement']:.2%} ({diff_metrics['improvement_pct']:.1f}% relative)")

        print("=" * 80)

    # Save results
    print(f"\nSaving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(eval_results, f, indent=2)
    print("Done!")
