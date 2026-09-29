"""
PHASE A — Vocabulary Expansion + MLM Pre-Training
Adds ~800 tokens to mBERT (498 Gujarati OBPE + ~300 Romanized Gujlish)
then runs Masked Language Model continued pre-training on 51k Gujlish corpus.

KAGGLE SETUP:
  Dataset  → rud12/gujarati-english-tokenization-study
  Model    → rudrakachhia/mbert-obpe-adapted (transformers / default / 1)
  GPU      → T4 x2  (or T4 x1)
  Internet → OFF after adding inputs
  Runtime  → ~8-12 hours total

HOW TO RUN:
  Just click "Run All" — all 18 cells run in sequence automatically.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install & Imports
# ═══════════════════════════════════════════════════════════════════════════════
import subprocess, sys

for pkg in ["transformers[torch]", "datasets", "accelerate", "sentencepiece", "nltk"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, gc, re, json, math, random, logging, warnings
import numpy as np
import pandas as pd
import torch
import nltk
from pathlib import Path
from datetime import datetime
from collections import Counter

from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer,
    EarlyStoppingCallback, set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

SEED = 42
set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i} : {p.name}  {p.total_memory//1024**2} MB")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Configuration  (edit these if needed)
# ═══════════════════════════════════════════════════════════════════════════════
class CFG:
    # ── Kaggle Paths ───────────────────────────────────────────────────────────
    DATA_DIR   = "data/processed"
    MODEL_DIR  = "models/mbert_adapted"
    OUT_VOCAB  = "/kaggle/working/out/vocab/gujlish_800tok_model"    # expanded vocab model
    OUT_MLM    = "/kaggle/working/out/model/gujlish_mlm_final"       # after MLM training
    CKPT_DIR   = "/kaggle/working/out/cp/mlm_checkpoints"
    LOG_DIR    = "/kaggle/working/out/log/mlm_logs"

    # ── CSV column name containing text ───────────────────────────────────────
    TEXT_COL   = "text"          # change if your CSVs use a different column

    # ── Vocabulary expansion ───────────────────────────────────────────────────
    TARGET_NEW_TOKENS  = 300     # Romanized Gujlish tokens to add
    MIN_WORD_FREQ      = 5       # ignore words appearing < 5 times
    MIN_SUBWORDS       = 2       # only consider words split into ≥ 2 pieces
    MAX_WORD_LEN       = 25      # ignore very long tokens (likely noise)
    MIN_WORD_LEN       = 3       # ignore very short tokens

    # ── MLM Training ──────────────────────────────────────────────────────────
    MAX_SEQ_LEN        = 128
    MLM_PROBABILITY    = 0.15
    NUM_EPOCHS         = 3
    BATCH_SIZE         = 32      # reduce to 16 if you get OOM errors
    GRAD_ACCUM         = 2       # effective batch = 32 × 2 = 64
    LEARNING_RATE      = 2e-5
    WEIGHT_DECAY       = 0.01
    WARMUP_RATIO       = 0.10
    LR_SCHEDULER       = "cosine"
    FP16               = True
    EVAL_RATIO         = 0.02    # 2% for validation
    EVAL_STEPS         = 500
    SAVE_STEPS         = 500
    SAVE_TOTAL_LIMIT   = 3
    LOGGING_STEPS      = 100

for d in [CFG.OUT_VOCAB, CFG.OUT_MLM, CFG.CKPT_DIR, CFG.LOG_DIR]:
    os.makedirs(d, exist_ok=True)
print("Config ready.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Load Raw Text Corpus
# ═══════════════════════════════════════════════════════════════════════════════
def find_text_column(df):
    """Auto-detect the text column from common names."""
    for col in ["text", "sentence", "comment", "review", "content"]:
        if col in df.columns:
            return col
    # fallback: first string column
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError(f"No text column found. Columns: {list(df.columns)}")

def load_corpus(data_dir, text_col):
    texts = []
    print(f"Scanning {data_dir} ...")
    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path)
            col = text_col if text_col in df.columns else find_text_column(df)
            t = df[col].dropna().astype(str).tolist()
            texts.extend(t)
            print(f"  {fname:<45} {len(t):>7,} rows  (col='{col}')")
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    # Clean
    texts = [s.strip() for s in texts]
    texts = list(dict.fromkeys(texts))           # deduplicate
    texts = [s for s in texts if 10 < len(s) < 5000]
    print(f"\nTotal unique sentences: {len(texts):,}")
    return texts

all_texts = load_corpus(CFG.DATA_DIR, CFG.TEXT_COL)

print("\nSample sentences:")
for s in random.sample(all_texts, min(5, len(all_texts))):
    print(f"  → {s[:120]}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Load OBPE-Adapted Model + Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════
def find_model_path(primary):
    if os.path.exists(primary):
        return primary
    # Auto-search under /kaggle/input/
    print(f"Not found at {primary}. Searching /kaggle/input/ ...")
    print("Available inputs:", os.listdir("/kaggle/input/"))
    for root, dirs, files in os.walk("/kaggle/input/"):
        if "tokenizer_config.json" in files or "vocab.txt" in files:
            print(f"  Found model at: {root}")
            return root
    raise FileNotFoundError(
        "OBPE-adapted model not found. "
        "Add 'rudrakachhia/mbert-obpe-adapted' as a Kaggle Model input."
    )

model_path = find_model_path(CFG.MODEL_DIR)
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForMaskedLM.from_pretrained(model_path, ignore_mismatched_sizes=False)

vocab_size_before = len(tokenizer)
emb_before = model.bert.embeddings.word_embeddings.weight.shape[0]

assert vocab_size_before == emb_before, (
    f"Embedding mismatch in source model: tokenizer={vocab_size_before}, "
    f"embedding={emb_before}. Wrong model loaded."
)

print(f"Tokenizer vocab size : {vocab_size_before:,}")
if vocab_size_before >= 120_000:
    extra = vocab_size_before - 119_547
    print(f"  ✓ OBPE-adapted confirmed (+{extra} tokens over base mBERT)")
else:
    print("  ⚠ WARNING: This looks like base mBERT, not the OBPE-adapted version!")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Extract Top Romanized Gujlish Tokens
# ═══════════════════════════════════════════════════════════════════════════════
# Strategy:
#   1. Tokenize every word in corpus with current tokenizer
#   2. Keep only Latin-script words that are fragmented (≥2 pieces)
#   3. Filter out real English words using NLTK wordlist
#   4. Rank by  score = log2(freq) × num_subwords
#   5. Take top TARGET_NEW_TOKENS

print("Downloading NLTK English word list...")
nltk.download("words", quiet=True)
from nltk.corpus import words as nltk_words
ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())
print(f"English word list: {len(ENGLISH_WORDS):,} words")

# ── Gujarati Unicode range ─────────────────────────────────────────────────────
GUJARATI_RANGE = re.compile(r'[\u0A80-\u0AFF]')
LATIN_WORD     = re.compile(r'^[a-zA-Z][a-zA-Z0-9\'-]{2,}$')  # pure Latin words

def is_romanized_gujlish(word: str) -> bool:
    """Return True if word is a Latin-script Gujarati/code-mixed word (not English)."""
    w = word.strip().lower()
    if not LATIN_WORD.match(w):              return False  # must be Latin
    if GUJARATI_RANGE.search(w):             return False  # skip Gujarati script
    if w in ENGLISH_WORDS:                   return False  # skip real English words
    if len(w) < CFG.MIN_WORD_LEN:            return False
    if len(w) > CFG.MAX_WORD_LEN:            return False
    return True

print("\nExtracting word frequencies from corpus...")
word_freq: Counter = Counter()
for sentence in all_texts:
    for word in sentence.split():
        w = word.strip(".,!?;:\"'()[]{}").lower()
        if w:
            word_freq[w] += 1

print(f"Unique words in corpus: {len(word_freq):,}")

# ── Find fragmented Romanized Gujlish words ───────────────────────────────────
print("\nAnalyzing tokenization fragmentation for Romanized words...")
candidates = {}

for word, freq in word_freq.items():
    if freq < CFG.MIN_WORD_FREQ:
        continue
    if not is_romanized_gujlish(word):
        continue

    tokens = tokenizer.tokenize(word)
    n_subwords = len(tokens)

    if n_subwords < CFG.MIN_SUBWORDS:
        continue  # already handled as single token

    # Score = log2(freq) × num_subwords  (same formula as original vocab selection)
    score = math.log2(freq) * n_subwords
    candidates[word] = {
        "freq"      : freq,
        "subwords"  : n_subwords,
        "score"     : round(score, 2),
        "tokens"    : tokens,
    }

print(f"Romanized Gujlish candidate words: {len(candidates):,}")

# ── Rank and select top N ──────────────────────────────────────────────────────
ranked = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)

# Also check that these words are not already in tokenizer vocab
existing_vocab = set(tokenizer.vocab.keys())
new_romanized  = []
for word, info in ranked:
    if word not in existing_vocab and ("##" + word) not in existing_vocab:
        new_romanized.append((word, info))
    if len(new_romanized) >= CFG.TARGET_NEW_TOKENS:
        break

print(f"\nTop {len(new_romanized)} Romanized Gujlish tokens selected:")
print(f"{'Word':<20} {'Freq':>8} {'Subwords':>9} {'Score':>8} {'mBERT splits'}")
print("-" * 75)
for word, info in new_romanized[:30]:  # show top 30
    splits = " + ".join(info["tokens"])
    print(f"{word:<20} {info['freq']:>8,} {info['subwords']:>9} {info['score']:>8.2f}  {splits}")
if len(new_romanized) > 30:
    print(f"  ... and {len(new_romanized)-30} more")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Add Romanized Tokens to Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════
romanized_words = [w for w, _ in new_romanized]

print(f"\nAdding {len(romanized_words)} Romanized Gujlish tokens to vocabulary...")
num_added = tokenizer.add_tokens(romanized_words)
print(f"  Tokens successfully added: {num_added}")
print(f"  Vocab size: {vocab_size_before:,} → {len(tokenizer):,}")
print(f"  Total new tokens over base mBERT: +{len(tokenizer) - 119_547}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Resize Embedding Matrix + Mean Initialize New Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
# Get the OLD embedding matrix BEFORE resizing
old_embeddings = model.bert.embeddings.word_embeddings.weight.data.clone()  # [old_vocab, 768]
old_vocab_size = old_embeddings.shape[0]

# Resize model embeddings to match new tokenizer vocab
model.resize_token_embeddings(len(tokenizer))

print(f"\nEmbedding matrix resized: {old_vocab_size:,} → {len(tokenizer):,}")

# ── Mean initialize ONLY the newly added Romanized tokens ────────────────────
# (The 498 OBPE tokens added earlier already have good mean-init from the source model)
new_embedding_layer = model.bert.embeddings.word_embeddings

with torch.no_grad():
    for word in romanized_words:
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id is None or token_id < old_vocab_size:
            continue  # skip if token already existed

        # Get constituent subword pieces from the ORIGINAL tokenizer state
        # We use tokenizer.tokenize() on the whole word to get the pieces
        subword_pieces = tokenizer.tokenize(word)
        # Filter out the word itself if it tokenizes to itself (already added)
        subword_pieces = [p for p in subword_pieces if p != word and p != f"##{word}"]

        if subword_pieces:
            # Get embedding IDs of constituent subwords
            piece_ids = tokenizer.convert_tokens_to_ids(subword_pieces)
            piece_ids = [pid for pid in piece_ids if pid < old_vocab_size]  # only from old vocab

            if piece_ids:
                # Mean of constituent subword embeddings
                piece_embs = old_embeddings[piece_ids]  # [k, 768]
                mean_emb   = piece_embs.mean(dim=0)      # [768]
                new_embedding_layer.weight.data[token_id] = mean_emb
                continue

        # Fallback: use global mean of all embeddings (rare case)
        new_embedding_layer.weight.data[token_id] = old_embeddings.mean(dim=0)

# Also resize the MLM head output (tied weights in BERT — must match)
model.tie_weights()

# ── Final verification ─────────────────────────────────────────────────────────
final_emb_size = model.bert.embeddings.word_embeddings.weight.shape[0]
assert final_emb_size == len(tokenizer), (
    f"Mismatch after resize: embedding={final_emb_size}, tokenizer={len(tokenizer)}"
)

total_added = len(tokenizer) - 119_547
print(f"\n✓ Vocabulary expansion complete")
print(f"  Base mBERT          : 119,547 tokens")
print(f"  + Gujarati OBPE     : {vocab_size_before - 119_547} tokens")
print(f"  + Romanized Gujlish : {num_added} tokens")
print(f"  ─────────────────────────────────────")
print(f"  TOTAL vocab size    : {len(tokenizer):,} tokens  (+{total_added} new)")
print(f"  Embedding matrix    : {final_emb_size:,} × 768  ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Save Expanded Vocabulary Model (before MLM training)
# ═══════════════════════════════════════════════════════════════════════════════
tokenizer.save_pretrained(CFG.OUT_VOCAB)
model.save_pretrained(CFG.OUT_VOCAB)

# Save token list for documentation
token_info = {
    "base_mbert_vocab"      : 119_547,
    "gujarati_obpe_tokens"  : vocab_size_before - 119_547,
    "romanized_gujlish_tokens": num_added,
    "total_vocab_size"      : len(tokenizer),
    "total_new_tokens"      : total_added,
    "romanized_tokens_added": romanized_words,
    "top30_candidates"      : [
        {"word": w, **info} for w, info in new_romanized[:30]
    ],
}
with open(os.path.join(CFG.OUT_VOCAB, "token_expansion_info.json"), "w") as f:
    json.dump(token_info, f, indent=2, ensure_ascii=False)

print(f"\nExpanded model saved to: {CFG.OUT_VOCAB}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Prepare Dataset for MLM
# ═══════════════════════════════════════════════════════════════════════════════
random.shuffle(all_texts)
n_eval  = max(200, int(len(all_texts) * CFG.EVAL_RATIO))
n_train = len(all_texts) - n_eval

train_texts = all_texts[:n_train]
eval_texts  = all_texts[n_train:]
print(f"Train: {len(train_texts):,}  |  Eval: {len(eval_texts):,}")

train_ds = Dataset.from_dict({"text": train_texts})
eval_ds  = Dataset.from_dict({"text": eval_texts})

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=CFG.MAX_SEQ_LEN,
        padding=False,
        return_special_tokens_mask=True,
    )

print("Tokenizing...")
tok_train = train_ds.map(tokenize_fn, batched=True, batch_size=1000,
                          num_proc=2, remove_columns=["text"], desc="Train")
tok_eval  = eval_ds.map(tokenize_fn,  batched=True, batch_size=1000,
                          num_proc=2, remove_columns=["text"], desc="Eval")

print(f"Tokenized train: {len(tok_train):,}  |  eval: {len(tok_eval):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Data Collator
# ═══════════════════════════════════════════════════════════════════════════════
data_collator = DataCollatorForLanguageModeling(
    tokenizer       = tokenizer,
    mlm             = True,
    mlm_probability = CFG.MLM_PROBABILITY,
    pad_to_multiple_of = 8 if CFG.FP16 else None,
)

# Quick sanity check
sample = data_collator([tok_train[i] for i in range(4)])
masked = (sample["labels"] != -100).sum().item()
total  = sample["input_ids"].numel()
print(f"Collator OK — batch shape: {sample['input_ids'].shape}")
print(f"Masked tokens: {masked}/{total} = {masked/total:.1%}  (target ~15%)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — Training Arguments
# ═══════════════════════════════════════════════════════════════════════════════
n_gpus       = max(torch.cuda.device_count(), 1)
eff_batch    = CFG.BATCH_SIZE * CFG.GRAD_ACCUM * n_gpus
total_steps  = (len(tok_train) // eff_batch) * CFG.NUM_EPOCHS
warmup_steps = int(total_steps * CFG.WARMUP_RATIO)

print(f"\nTraining schedule:")
print(f"  GPUs               : {n_gpus}")
print(f"  Effective batch    : {eff_batch}")
print(f"  Total steps        : {total_steps:,}")
print(f"  Warmup steps       : {warmup_steps:,}")
print(f"  Estimated time     : ~{total_steps * 0.35 / 3600:.1f} hours (T4 estimate)")

training_args = TrainingArguments(
    output_dir                   = CFG.CKPT_DIR,
    overwrite_output_dir         = True,
    num_train_epochs             = CFG.NUM_EPOCHS,
    per_device_train_batch_size  = CFG.BATCH_SIZE,
    per_device_eval_batch_size   = CFG.BATCH_SIZE,
    gradient_accumulation_steps  = CFG.GRAD_ACCUM,
    learning_rate                = CFG.LEARNING_RATE,
    weight_decay                 = CFG.WEIGHT_DECAY,
    adam_beta1                   = 0.9,
    adam_beta2                   = 0.999,
    adam_epsilon                 = 1e-8,
    max_grad_norm                = 1.0,
    lr_scheduler_type            = CFG.LR_SCHEDULER,
    warmup_steps                 = warmup_steps,
    fp16                         = CFG.FP16 and torch.cuda.is_available(),
    evaluation_strategy          = "steps",
    eval_steps                   = CFG.EVAL_STEPS,
    save_strategy                = "steps",
    save_steps                   = CFG.SAVE_STEPS,
    save_total_limit             = CFG.SAVE_TOTAL_LIMIT,
    load_best_model_at_end       = True,
    metric_for_best_model        = "eval_loss",
    greater_is_better            = False,
    logging_dir                  = CFG.LOG_DIR,
    logging_strategy             = "steps",
    logging_steps                = CFG.LOGGING_STEPS,
    report_to                    = "none",
    seed                         = SEED,
    data_seed                    = SEED,
    dataloader_num_workers       = 2,
    dataloader_pin_memory        = True,
    group_by_length              = True,
    prediction_loss_only         = True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Build Trainer
# ═══════════════════════════════════════════════════════════════════════════════
trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = tok_train,
    eval_dataset  = tok_eval,
    tokenizer     = tokenizer,
    data_collator = data_collator,
    callbacks     = [
        EarlyStoppingCallback(
            early_stopping_patience  = 5,
            early_stopping_threshold = 0.001,
        )
    ],
)
print("Trainer ready.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STARTING MLM PRE-TRAINING")
print(f"  Vocab size : {len(tokenizer):,}  (+{total_added} over base mBERT)")
print(f"  Dataset    : {len(tok_train):,} train  |  {len(tok_eval):,} eval")
print(f"  Epochs     : {CFG.NUM_EPOCHS}")
print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*65 + "\n")

# Auto-resume if Kaggle session restarted
last_ckpt = get_last_checkpoint(CFG.CKPT_DIR) if os.path.isdir(CFG.CKPT_DIR) else None
if last_ckpt:
    print(f"Resuming from checkpoint: {last_ckpt}")

train_result = trainer.train(resume_from_checkpoint=last_ckpt)

print(f"\nTraining finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14 — Save Final Model
# ═══════════════════════════════════════════════════════════════════════════════
trainer.save_model(CFG.OUT_MLM)
tokenizer.save_pretrained(CFG.OUT_MLM)

# Copy token expansion info to final model folder
import shutil
src = os.path.join(CFG.OUT_VOCAB, "token_expansion_info.json")
shutil.copy(src, CFG.OUT_MLM)

train_metrics = train_result.metrics
train_metrics["train_samples"] = len(tok_train)
trainer.save_metrics("train", train_metrics)

print(f"Final model saved to: {CFG.OUT_MLM}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15 — Evaluate + Perplexity
# ═══════════════════════════════════════════════════════════════════════════════
eval_metrics = trainer.evaluate()
eval_loss    = eval_metrics["eval_loss"]
perplexity   = math.exp(eval_loss)

print(f"\n{'='*50}")
print("  FINAL RESULTS")
print(f"{'='*50}")
print(f"  Train loss   : {train_metrics.get('train_loss', 'N/A'):.4f}")
print(f"  Eval loss    : {eval_loss:.4f}")
print(f"  Perplexity   : {perplexity:.2f}")
print(f"  Train time   : {train_metrics.get('train_runtime',0)/3600:.2f} hrs")
print(f"  Vocab size   : {len(tokenizer):,}  (+{total_added} new tokens)")
print(f"{'='*50}")

if   perplexity < 5:  print("  Excellent — very low perplexity.")
elif perplexity < 15: print("  Good — reasonable for code-mixed text.")
elif perplexity < 30: print("  Fair — consider more epochs or data.")
else:                 print("  Note — high perplexity; check LR and data.")

# Save all metrics
with open(os.path.join(CFG.OUT_MLM, "mlm_metrics.json"), "w") as f:
    json.dump({
        "train_loss"            : train_metrics.get("train_loss"),
        "eval_loss"             : eval_loss,
        "perplexity"            : perplexity,
        "train_samples"         : len(tok_train),
        "eval_samples"          : len(tok_eval),
        "total_vocab_size"      : len(tokenizer),
        "new_tokens_total"      : total_added,
        "gujarati_obpe_tokens"  : vocab_size_before - 119_547,
        "romanized_new_tokens"  : num_added,
        "epochs"                : CFG.NUM_EPOCHS,
        "max_seq_len"           : CFG.MAX_SEQ_LEN,
        "mlm_probability"       : CFG.MLM_PROBABILITY,
        "learning_rate"         : CFG.LEARNING_RATE,
    }, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 16 — Verify Saved Model
# ═══════════════════════════════════════════════════════════════════════════════
print("\nVerifying saved model...")
from transformers import AutoTokenizer as AT, AutoModelForMaskedLM as AM
chk_tok = AT.from_pretrained(CFG.OUT_MLM)
chk_mdl = AM.from_pretrained(CFG.OUT_MLM)

assert len(chk_tok) == len(tokenizer)
assert chk_mdl.bert.embeddings.word_embeddings.weight.shape[0] == len(tokenizer)
print(f"  ✓ Vocab size     : {len(chk_tok):,}")
print(f"  ✓ Embedding size : {chk_mdl.bert.embeddings.word_embeddings.weight.shape[0]:,}")

del chk_mdl; gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 17 — Fill-Mask Sanity Test
# ═══════════════════════════════════════════════════════════════════════════════
from transformers import pipeline

fill = pipeline(
    "fill-mask", model=CFG.OUT_MLM, tokenizer=CFG.OUT_MLM,
    device=0 if torch.cuda.is_available() else -1, top_k=5,
)

test_sents = [
    "Aaje hava khub [MASK] chhe.",
    "Tamaro [MASK] khub saras chhe.",
    "Mane aa [MASK] bilkul pasand nathi.",
    "This [MASK] is very good for everyone.",
    "Ekdum [MASK] chhe yaar, maja avi gayi.",
]

print("\n" + "="*55)
print("  FILL-MASK SANITY CHECK")
print("="*55)
for sent in test_sents:
    try:
        results = fill(sent)
        print(f"\nInput : {sent}")
        for r in results[:3]:
            print(f"  [{r['score']:.3f}] {r['sequence']}")
    except Exception as e:
        print(f"Skipped '{sent}': {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 18 — Summary and Next Steps
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  DONE — SUMMARY")
print("="*65)
print(f"  Base mBERT vocab     : 119,547 tokens")
print(f"  + Gujarati OBPE      : +{vocab_size_before - 119_547} tokens")
print(f"  + Romanized Gujlish  : +{num_added} tokens")
print(f"  Final vocab size     : {len(tokenizer):,} tokens")
print(f"  MLM Perplexity       : {perplexity:.2f}")
print(f"  Model saved at       : {CFG.OUT_MLM}")
print()
print("  NEXT STEPS:")
print("  1. Upload /kaggle/working/gujlish_mlm_final/ to Kaggle Models")
print("  2. Use it in Phase 7 classification (replace rudrakachhia/mbert-obpe-adapted)")
print("  3. Compare F1 against:")
print("     - mBERT baseline          : F1 = 0.7377")
print("     - mBERT OBPE adapted      : F1 = 0.7372  (no MLM)")
print("     - THIS MODEL (with MLM)   : F1 = ???  ← expected improvement")
print("="*65)

print("\nFiles in output:")
for f in sorted(os.listdir(CFG.OUT_MLM)):
    sz = os.path.getsize(os.path.join(CFG.OUT_MLM, f)) / 1024 / 1024
    print(f"  {f:<40} {sz:>8.2f} MB")
