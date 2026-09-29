"""
run_mlm_pretrain.py  —  Continued MLM Pre-training (GPU Required)
==================================================================
WHY THIS IS NEEDED:
  Adding new tokens to mBERT gives them mean-initialized embeddings.
  These embeddings have NO context-based meaning yet — the model has
  never seen these tokens in any sentence. Without continued MLM
  pre-training, the token embeddings remain generic and the model
  cannot benefit from the vocabulary expansion during fine-tuning.
  This is documented in Artetxe et al. (2020) and Chau et al. (2020).

WHAT THIS DOES:
  1. Loads mbert_adapted_500 (646 new tokens, mean-initialized)
  2. Runs continued Masked Language Modeling on the full Gujlish corpus
     (21,346 sentences, no labels needed — pure self-supervised)
  3. Saves the MLM-pre-trained model to models/mbert_adapted_mlm/
  4. This model is then used in run_phase7_v2.py for fine-tuning

EXPECTED OUTCOME:
  After MLM pre-training, the new token embeddings learn actual
  contextual representations from Gujlish sentences. This should
  produce a meaningful improvement over the baseline when fine-tuned.

SETUP (on friend's GPU machine):
  git pull
  pip install -r requirements.txt
  python run_mlm_pretrain.py        # ~30-90 min on RTX 3060
  python run_phase7_v2.py           # fine-tune after MLM

REFERENCES:
  Artetxe et al. (2020). Massively Multilingual Transfer for NER.
  Chau et al. (2020). Parsing with Multilingual BERT, a Small Corpus.
  Pfeiffer et al. (2020). MAD-X: An Adapter-Based Framework.
"""

import sys, io, os, gc, time, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path(__file__).parent
os.environ["HF_HOME"]            = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache")

OUTPUT_DIR = ROOT / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import torch

print("=" * 65)
print("  MLM CONTINUED PRE-TRAINING — Gujlish Corpus")
print("=" * 65)

# ── GPU check ─────────────────────────────────────────────────────────────────
print("\n[1/5] Checking hardware...")
if not torch.cuda.is_available():
    print("  WARNING: No CUDA GPU — MLM will be very slow on CPU (10+ hours).")
    print("  Strongly recommended to run on a GPU machine.")
    DEVICE    = "cpu"
    GPU_NAME  = "CPU"
    VRAM_GB   = 0
    BATCH     = 8
    USE_FP16  = False
    USE_CPU   = True
else:
    DEVICE    = "cuda"
    GPU_NAME  = torch.cuda.get_device_name(0)
    VRAM_GB   = torch.cuda.get_device_properties(0).total_memory / 1e9
    BATCH     = 32 if VRAM_GB >= 12 else 16
    USE_FP16  = True
    USE_CPU   = False
    print(f"  GPU       : {GPU_NAME} ({VRAM_GB:.1f} GB)")
    print(f"  Batch     : {BATCH}")

# ── Config ────────────────────────────────────────────────────────────────────
MLM_PROB     = 0.15    # standard BERT masking probability
MAX_LEN      = 128     # max sequence length
MLM_EPOCHS   = 5       # continued pre-training epochs
LR           = 2e-4    # higher LR for embedding training (not fine-tuning)
WARMUP_RATIO = 0.06
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_IN  = ROOT / "models" / "mbert_adapted_500"   # input: 646-token adapted model
MODEL_OUT = ROOT / "models" / "mbert_adapted_mlm"   # output: after MLM pre-training

if not MODEL_IN.exists():
    print(f"\n  ERROR: {MODEL_IN} not found.")
    print("  Make sure friend_gpu_files_v2.zip was extracted into the project folder.")
    sys.exit(1)

print(f"  Input model  : {MODEL_IN.name}")
print(f"  Output model : {MODEL_OUT.name}")
print(f"  MLM epochs   : {MLM_EPOCHS}  |  LR: {LR}  |  Mask prob: {MLM_PROB}")

# ── Load corpus ───────────────────────────────────────────────────────────────
print("\n[2/5] Loading Gujlish corpus...")

DATA = ROOT / "data" / "processed"
frames = []
for fname in ["filtered_clean.csv", "expanded_dev_subset_full.csv", "combined_full_dataset.csv"]:
    path = DATA / fname
    if path.exists():
        df_tmp = pd.read_csv(path, usecols=["text"], low_memory=False)
        frames.append(df_tmp)
        print(f"  Loaded {fname}: {len(df_tmp):,} rows")

if not frames:
    print("  ERROR: No dataset found in data/processed/")
    sys.exit(1)

df_corpus = pd.concat(frames, ignore_index=True).drop_duplicates(subset="text")
df_corpus["text"] = df_corpus["text"].astype(str).str.strip()
df_corpus = df_corpus[df_corpus["text"].str.len() > 10].reset_index(drop=True)

corpus_texts = df_corpus["text"].tolist()
print(f"  Total unique sentences for MLM: {len(corpus_texts):,}")

# ── Tokenize corpus ───────────────────────────────────────────────────────────
print("\n[3/5] Tokenizing corpus...")
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                           DataCollatorForWholeWordMask,
                           TrainingArguments, Trainer)
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_IN), use_fast=True)
print(f"  Vocab size   : {len(tokenizer):,}")
print(f"  New tokens   : {len(tokenizer) - 119547} (over base mBERT)")

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_special_tokens_mask=True,
    )

hf_dataset = Dataset.from_dict({"text": corpus_texts})
tokenized  = hf_dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    desc="Tokenizing"
)
print(f"  Tokenized {len(tokenized):,} examples")

# ── Load model for MLM ────────────────────────────────────────────────────────
print("\n[4/5] Loading model for MLM pre-training...")
model = AutoModelForMaskedLM.from_pretrained(str(MODEL_IN), ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))
print(f"  Parameters   : {sum(p.numel() for p in model.parameters()):,}")

# Whole-word masking is better for code-mixed text (masks full words not subwords)
data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROB,
)

# ── Training ──────────────────────────────────────────────────────────────────
total_steps = (len(tokenized) // BATCH) * MLM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

print(f"\n  Training steps : {total_steps:,}  |  Warmup: {warmup_steps}")
print(f"  Estimated time (RTX 3060): ~{int(total_steps * 0.025 / 60)} min")

args = TrainingArguments(
    output_dir                  = str(ROOT / "tmp_mlm"),
    num_train_epochs            = MLM_EPOCHS,
    per_device_train_batch_size = BATCH,
    learning_rate               = LR,
    weight_decay                = 0.01,
    warmup_steps                = warmup_steps,
    fp16                        = USE_FP16,
    save_strategy               = "epoch",
    save_total_limit            = 1,
    logging_steps               = 100,
    report_to                   = "none",
    use_cpu                     = USE_CPU,
    no_cuda                     = USE_CPU,
    dataloader_num_workers      = 0,
    seed                        = SEED,
    prediction_loss_only        = True,
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = tokenized,
    data_collator   = data_collator,
    tokenizer       = tokenizer,
)

print("\n[5/5] Running MLM pre-training...")
t0 = time.time()
train_result = trainer.train()
elapsed = time.time() - t0

print(f"\n  Training complete in {elapsed/60:.1f} minutes")
print(f"  Final train loss: {train_result.training_loss:.4f}")

# ── Save MLM pre-trained model ────────────────────────────────────────────────
MODEL_OUT.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(str(MODEL_OUT))
model.save_pretrained(str(MODEL_OUT))

# Save metadata
meta = {
    "base_model":         str(MODEL_IN.name),
    "mlm_epochs":         MLM_EPOCHS,
    "mlm_probability":    MLM_PROB,
    "corpus_size":        len(corpus_texts),
    "vocab_size":         len(tokenizer),
    "new_tokens_over_mbert": len(tokenizer) - 119547,
    "final_train_loss":   round(float(train_result.training_loss), 4),
    "training_time_min":  round(elapsed / 60, 1),
    "gpu":                GPU_NAME,
    "learning_rate":      LR,
    "max_seq_len":        MAX_LEN,
    "whole_word_masking": True,
}
with open(MODEL_OUT / "mlm_training_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n  Saved MLM pre-trained model → models/mbert_adapted_mlm/")
print(f"  Saved metadata → models/mbert_adapted_mlm/mlm_training_meta.json")

# Clean up tmp
import shutil
shutil.rmtree(ROOT / "tmp_mlm", ignore_errors=True)

print("\n" + "=" * 65)
print("  MLM PRE-TRAINING COMPLETE")
print("=" * 65)
print(f"  GPU            : {GPU_NAME}")
print(f"  Time           : {elapsed/60:.1f} minutes")
print(f"  Corpus size    : {len(corpus_texts):,} sentences")
print(f"  Final MLM loss : {train_result.training_loss:.4f}")
print(f"\n  NEXT STEP: Run fine-tuning")
print(f"    python run_phase7_v2.py")
print(f"  (It will auto-detect models/mbert_adapted_mlm and use it)")
