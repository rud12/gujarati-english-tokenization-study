"""
PHASE A — Vocabulary Expansion + MLM Pre-Training
Adds ~800 tokens to mBERT (498 Gujarati OBPE + ~300 Romanized Gujlish)
then runs Masked Language Model continued pre-training on 51k Gujlish corpus.

KAGGLE SETUP:
  Dataset  → rud12/gujarati-english-tokenization-study
  Model    → rudrakachhia/mbert-obpe-adapted (transformers / default / 1)
  GPU      → T4 x2  (or T4 x1)
  Internet → OFF after package install / local inputs are available
  Runtime  → ~20-30 min expected for the current ~51k corpus on 2×T4; measure actual
  Warnings  → logged and allowed to continue; only genuine fatal/API/file errors stop.

HOW TO RUN:
  Just click "Run All" — all 18 cells run in sequence automatically.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install & Imports
# ═══════════════════════════════════════════════════════════════════════════════
import subprocess, sys

for pkg in [
    "transformers==4.57.6",
    "datasets>=2.15.0,<5.0.0",
    "accelerate>=1.1.0,<2.0.0",
    "sentencepiece",
    "nltk"
]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, gc, re, json, math, random, logging, warnings, inspect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
import torch
import nltk
import transformers
import datasets as hf_datasets
import accelerate
from pathlib import Path
from datetime import datetime
from collections import Counter

EXPECTED_TRANSFORMERS = "4.57.6"
if transformers.__version__ != EXPECTED_TRANSFORMERS:
    raise RuntimeError(
        f"Incompatible Transformers version: {transformers.__version__}. "
        f"Expected exactly {EXPECTED_TRANSFORMERS}. "
        "Restart the Kaggle session after changing the pip pin, then Run All."
    )

print(f"Transformers : {transformers.__version__}")
print(f"Datasets     : {hf_datasets.__version__}")
print(f"Accelerate   : {accelerate.__version__}")

from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer,
    EarlyStoppingCallback, set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from transformers.modeling_outputs import MaskedLMOutput

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def warn_and_continue(message):
    print(f"WARNING: {message}")

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
ROOT = Path(__file__).resolve().parent

class CFG:
    # ── Kaggle Paths ───────────────────────────────────────────────────────────
    DATA_DIR   = ROOT / "data" / "processed"
    MODEL_DIR  = ROOT / "models" / "mbert_adapted"
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
    BATCH_SIZE         = 32      # loss-only multi-GPU gather; reduce to 16 only if smoke test OOMs
    GRAD_ACCUM         = 2       # effective batch = 32 × 2 = 64
    LEARNING_RATE      = 2e-5
    WEIGHT_DECAY       = 0.01
    WARMUP_RATIO       = 0.10
    LR_SCHEDULER       = "cosine"
    FP16               = True
    EVAL_RATIO         = 0.02    # 2% for validation
    EVAL_STEPS         = 250
    SAVE_STEPS         = 250
    SAVE_TOTAL_LIMIT   = 3
    LOGGING_STEPS      = 100

    # ── Safety / reproducibility ──────────────────────────────────────────────
    EXPECTED_BASE_M_BERT_VOCAB = 119_547
    EXPECTED_OBPE_ADDITIONS    = 498
    # Start clean for the first experiment. Set True ONLY when an actual
    # training run was interrupted and you intentionally want to resume it.
    RESUME_FROM_CHECKPOINT     = False

for d in [CFG.OUT_VOCAB, CFG.OUT_MLM, CFG.CKPT_DIR, CFG.LOG_DIR]:
    os.makedirs(d, exist_ok=True)
# ── Zero-training API preflight ────────────────────────────────────────────────
# Fail here, before reading/tokenizing the 50k corpus, if the installed Trainer
# API does not match the script.
_required_ta = {
    "output_dir", "overwrite_output_dir", "eval_strategy", "eval_steps",
    "save_strategy", "save_steps", "load_best_model_at_end",
    "metric_for_best_model", "greater_is_better", "warmup_ratio",
    "group_by_length", "prediction_loss_only",
}
_ta_params = set(inspect.signature(TrainingArguments).parameters)
_missing_ta = sorted(_required_ta - _ta_params)
if _missing_ta:
    raise RuntimeError(
        "TrainingArguments API preflight failed. "
        f"Missing parameters: {_missing_ta}. "
        f"Transformers={transformers.__version__}"
    )

_trainer_params = set(inspect.signature(Trainer).parameters)
if "processing_class" not in _trainer_params:
    raise RuntimeError(
        "Trainer API preflight failed: `processing_class` is unavailable. "
        f"Transformers={transformers.__version__}"
    )

if tuple(int(x) for x in accelerate.__version__.split('.')[:2]) < (1, 1):
    raise RuntimeError(
        f"Accelerate {accelerate.__version__} is too old for data_seed in this script. "
        "Use accelerate>=1.1.0,<2.0.0."
    )

_collator_params = set(inspect.signature(DataCollatorForLanguageModeling).parameters)
_required_collator = {"tokenizer", "mlm", "mlm_probability", "pad_to_multiple_of"}
_missing_collator = sorted(_required_collator - _collator_params)
if _missing_collator:
    raise RuntimeError(
        "DataCollatorForLanguageModeling API preflight failed. "
        f"Missing parameters: {_missing_collator}. "
        f"Transformers={transformers.__version__}"
    )

_preflight_args = TrainingArguments(
    output_dir=str(CFG.CKPT_DIR),
    overwrite_output_dir=True,
    per_device_train_batch_size=CFG.BATCH_SIZE,
    per_device_eval_batch_size=CFG.BATCH_SIZE,
    gradient_accumulation_steps=CFG.GRAD_ACCUM,
    learning_rate=CFG.LEARNING_RATE,
    weight_decay=CFG.WEIGHT_DECAY,
    lr_scheduler_type=CFG.LR_SCHEDULER,
    warmup_ratio=CFG.WARMUP_RATIO,
    fp16=CFG.FP16 and torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=CFG.EVAL_STEPS,
    save_strategy="steps",
    save_steps=CFG.SAVE_STEPS,
    save_total_limit=CFG.SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=str(CFG.LOG_DIR),
    logging_strategy="steps",
    logging_steps=CFG.LOGGING_STEPS,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    group_by_length=True,
    prediction_loss_only=True,
)
del _preflight_args
print("TrainingArguments API preflight: OK")

# NLTK resource preflight. If internet is disabled, the corpus must already be
# available in the runtime; fail now instead of halfway through the run.
try:
    nltk.data.find("corpora/words")
except LookupError:
    try:
        nltk.download("words", quiet=True)
        nltk.data.find("corpora/words")
    except LookupError as e:
        raise RuntimeError(
            "NLTK corpus 'words' is unavailable. Enable internet once to download it "
            "or pre-bundle the NLTK words corpus before running offline."
        ) from e

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
    primary = Path(primary)
    if primary.exists():
        return str(primary)
    raise FileNotFoundError(
        f"OBPE-adapted model not found at: {primary}\n"
        "Place the intended OBPE mBERT model in models/mbert_adapted "
        "or update CFG.MODEL_DIR."
    )

model_path = find_model_path(CFG.MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)
model, loading_info = AutoModelForMaskedLM.from_pretrained(
    model_path,
    ignore_mismatched_sizes=False,
    local_files_only=True,
    output_loading_info=True,
)

missing_keys = loading_info.get("missing_keys", [])
unexpected_keys = loading_info.get("unexpected_keys", [])
error_msgs = loading_info.get("error_msgs", [])

if error_msgs:
    raise RuntimeError(
        "Model loading reported checkpoint errors:\n" + "\n".join(error_msgs[:10])
    )

if getattr(model.config, "model_type", None) != "bert" or not hasattr(model, "bert"):
    raise RuntimeError(
        f"Expected a BERT/mBERT MaskedLM model, got model_type={getattr(model.config, 'model_type', None)!r}."
    )

missing_mlm_head = [k for k in missing_keys if k.startswith("cls.predictions.")]
if missing_mlm_head:
    print("WARNING: MLM prediction-head weights are missing from the source checkpoint.")
    print("         The BERT encoder will be loaded from the checkpoint, while the")
    print("         missing MLM-head parameters will be initialized by Transformers")
    print("         and trained during this MLM run.")
    print("         This is valid training, but it is NOT strict continuation from")
    print("         a checkpoint that already contained a trained MLM prediction head.")
    for k in missing_mlm_head:
        print(f"  missing: {k}")

if unexpected_keys:
    print(f"WARNING: {len(unexpected_keys)} unexpected checkpoint keys found.")
    for k in unexpected_keys[:20]:
        print(f"  unexpected: {k}")

# IMPORTANT FOR KAGGLE T4 x2:
# Trainer uses torch.nn.DataParallel when launched normally from a notebook/script.
# BertForMaskedLM returns [batch, seq, vocab] logits (~120k vocab), and DataParallel
# gathers those logits onto GPU 0. That gather can OOM even when each T4 can finish
# its own forward/backward. We therefore return only the scalar MLM loss whenever
# labels are supplied. Trainer needs the loss, not the full logits, during training
# and evaluation. The saved checkpoint remains a normal BertForMaskedLM model and
# reloads with AutoModelForMaskedLM for downstream use.
class KaggleLossOnlyBertForMaskedLM(BertForMaskedLM):
    def forward(self, *args, labels=None, **kwargs):
        # Transformers 4.57.x may pass this Trainer-only kwarg. BERT does not consume it.
        kwargs.pop("num_items_in_batch", None)
        outputs = super().forward(*args, labels=labels, **kwargs)
        if labels is not None:
            return MaskedLMOutput(loss=outputs.loss)
        return outputs

if not isinstance(model, BertForMaskedLM):
    raise RuntimeError(
        f"Expected BertForMaskedLM for the memory-safe multi-GPU path, got {type(model).__name__}."
    )
model.__class__ = KaggleLossOnlyBertForMaskedLM
print("Multi-GPU MLM memory guard: enabled (loss-only DataParallel gather)")

vocab_size_before = len(tokenizer)
emb_before = model.bert.embeddings.word_embeddings.weight.shape[0]

assert vocab_size_before == emb_before, (
    f"Embedding mismatch in source model: tokenizer={vocab_size_before}, "
    f"embedding={emb_before}. Wrong model loaded."
)

print(f"Tokenizer vocab size : {vocab_size_before:,}")
expected_obpe_vocab = (
    CFG.EXPECTED_BASE_M_BERT_VOCAB + CFG.EXPECTED_OBPE_ADDITIONS
)
if vocab_size_before != expected_obpe_vocab:
    raise RuntimeError(
        f"Unexpected source vocabulary size: {vocab_size_before}. "
        f"Expected {expected_obpe_vocab} "
        f"({CFG.EXPECTED_BASE_M_BERT_VOCAB} base + {CFG.EXPECTED_OBPE_ADDITIONS} OBPE)."
    )
print(
    f"  ✓ OBPE-adapted confirmed "
    f"(+{CFG.EXPECTED_OBPE_ADDITIONS} tokens over base mBERT)"
)

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

if len(new_romanized) != CFG.TARGET_NEW_TOKENS:
    warn_and_continue(
        f"Only {len(new_romanized)} new Romanized tokens found; "
        f"target was {CFG.TARGET_NEW_TOKENS}. Continuing with the available tokens."
    )

# CRITICAL: capture the original WordPiece decomposition BEFORE add_tokens().
original_subword_pieces = {
    word: tokenizer.tokenize(word)
    for word, _ in new_romanized
}

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
if num_added != len(romanized_words):
    warn_and_continue(
        f"Tokenizer added {num_added} tokens but {len(romanized_words)} were requested. "
        "Continuing with successfully added tokens only."
    )
    romanized_words = [w for w in romanized_words if tokenizer.convert_tokens_to_ids(w) is not None]
print(f"  Vocab size: {vocab_size_before:,} → {len(tokenizer):,}")
print(f"  Total new tokens over base mBERT: +{len(tokenizer) - 119_547}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Resize Embedding Matrix + Mean Initialize New Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
# Get the OLD embedding matrix BEFORE resizing
old_embeddings = model.bert.embeddings.word_embeddings.weight.data.clone()  # [old_vocab, 768]
old_vocab_size = old_embeddings.shape[0]

# Resize model embeddings to match new tokenizer vocab.
# We initialize the new vectors ourselves, so disable Transformers' default
# mean/covariance initializer to avoid doing unnecessary work twice.
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

print(f"\nEmbedding matrix resized: {old_vocab_size:,} → {len(tokenizer):,}")

# ── Mean initialize ONLY the newly added Romanized tokens ────────────────────
new_embedding_layer = model.bert.embeddings.word_embeddings
global_mean = old_embeddings.mean(dim=0)

initialized_words = []
with torch.no_grad():
    for word in romanized_words:
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id is None or token_id < old_vocab_size:
            warn_and_continue(
                f"Could not initialize newly added token '{word}' (id={token_id}). "
                "Skipping that token instead of aborting the run."
            )
            continue

        # Use the ORIGINAL decomposition captured before add_tokens().
        subword_pieces = original_subword_pieces.get(word, [])
        piece_ids = tokenizer.convert_tokens_to_ids(subword_pieces)
        piece_ids = [
            pid for pid in piece_ids
            if pid is not None and pid < old_vocab_size
        ]

        if piece_ids:
            new_embedding_layer.weight.data[token_id] = old_embeddings[piece_ids].mean(dim=0)
        else:
            warn_and_continue(
                f"No valid original subword pieces found for '{word}'. "
                "Using the global embedding mean as fallback."
            )
            new_embedding_layer.weight.data[token_id] = global_mean
        initialized_words.append(word)

# BERT MLM input/output word embeddings are tied.
model.tie_weights()

# Fail fast if the 300 new vectors accidentally collapsed to the same vector.
new_ids = [tokenizer.convert_tokens_to_ids(w) for w in romanized_words]
new_matrix = new_embedding_layer.weight.data[new_ids]
pairwise_std = new_matrix.std(dim=0).mean().item()
if not math.isfinite(pairwise_std) or pairwise_std == 0.0:
    warn_and_continue(
        "New Romanized token embeddings have zero/invalid variability. "
        "Continuing anyway; this may reduce the benefit of vocabulary expansion."
    )
print(f"New-token embedding variability check: {pairwise_std:.6e}")

# ── Final verification ─────────────────────────────────────────────────────────
final_emb_size = model.bert.embeddings.word_embeddings.weight.shape[0]
assert final_emb_size == len(tokenizer), (
    f"Mismatch after resize: embedding={final_emb_size}, tokenizer={len(tokenizer)}"
)

total_added = len(tokenizer) - CFG.EXPECTED_BASE_M_BERT_VOCAB
print(f"\n✓ Vocabulary expansion complete")
print(f"  Base mBERT          : {CFG.EXPECTED_BASE_M_BERT_VOCAB:,} tokens")
print(f"  + Gujarati OBPE     : {vocab_size_before - CFG.EXPECTED_BASE_M_BERT_VOCAB} tokens")
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
    "base_mbert_vocab"      : CFG.EXPECTED_BASE_M_BERT_VOCAB,
    "gujarati_obpe_tokens"  : vocab_size_before - CFG.EXPECTED_BASE_M_BERT_VOCAB,
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

# Verify that the newly added tokens actually occur in the tokenized corpus.
# If a token never appears, its embedding cannot learn from MLM loss.
new_token_ids = {tokenizer.convert_tokens_to_ids(w): w for w in romanized_words}
new_token_counts = Counter()
for ids in tok_train["input_ids"]:
    for token_id in ids:
        if token_id in new_token_ids:
            new_token_counts[token_id] += 1
missing_in_train = [
    word for token_id, word in new_token_ids.items()
    if new_token_counts[token_id] == 0
]
if missing_in_train:
    warn_and_continue(
        f"{len(missing_in_train)} newly added tokens never occur in the training split. "
        "Their embeddings may receive little or no MLM learning. Examples: "
        + ", ".join(missing_in_train[:20])
    )
print(
    f"New-token corpus coverage: min={min(new_token_counts.values())}, "
    f"max={max(new_token_counts.values())}, "
    f"tokens_with_occurrences={sum(v > 0 for v in new_token_counts.values())}/{len(new_token_ids)}"
)

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
n_gpus = max(torch.cuda.device_count(), 1)
approx_global_batch = CFG.BATCH_SIZE * CFG.GRAD_ACCUM * n_gpus
approx_steps_per_epoch = math.ceil(len(tok_train) / approx_global_batch)
approx_total_steps = approx_steps_per_epoch * CFG.NUM_EPOCHS

print(f"\nTraining schedule (approximate):")
print(f"  GPUs                 : {n_gpus}")
print(f"  Approx global batch  : {approx_global_batch}")
print(f"  Approx total steps  : {approx_total_steps:,}")
print(f"  Warmup ratio         : {CFG.WARMUP_RATIO:.2%}")

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
    warmup_ratio                 = CFG.WARMUP_RATIO,
    fp16                         = CFG.FP16 and torch.cuda.is_available(),
    eval_strategy                = "steps",
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
    processing_class = tokenizer,
    data_collator = data_collator,
    callbacks     = [
        EarlyStoppingCallback(
            early_stopping_patience  = 2,
            early_stopping_threshold = 0.001,
        )
    ],
)
print("Trainer ready.")

# One-batch smoke test: validates model + tokenizer + collator + loss + backward
# on the actual training model before the long run begins. This is intentionally tiny.
smoke_features = [tok_train[i] for i in range(min(2, len(tok_train)))]
smoke_batch = data_collator(smoke_features)
smoke_batch = {
    k: v.to(training_args.device) if torch.is_tensor(v) else v
    for k, v in smoke_batch.items()
}
model.train()
smoke_outputs = model(**smoke_batch)
smoke_loss = smoke_outputs.loss
if smoke_loss is None or not torch.isfinite(smoke_loss):
    raise RuntimeError(f"MLM smoke test failed: loss={smoke_loss}")
smoke_loss.backward()
smoke_grad_norm_sq = 0.0
for p in model.parameters():
    if p.grad is not None:
        g = p.grad.detach()
        smoke_grad_norm_sq += float((g.float() ** 2).sum().item())
model.zero_grad(set_to_none=True)
smoke_grad_norm = math.sqrt(smoke_grad_norm_sq)
if not math.isfinite(smoke_grad_norm) or smoke_grad_norm == 0.0:
    raise RuntimeError(f"MLM smoke test failed: gradient norm={smoke_grad_norm}")
print(f"MLM one-batch smoke test: OK (loss={smoke_loss.item():.4f}, grad_norm={smoke_grad_norm:.4f})")

# Verify the exact multi-GPU path used by Trainer before committing to the full run.
# This catches the previous DataParallel logits-gather OOM without consuming a long run.
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    try:
        dp_smoke = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        dp_features = [tok_train[i] for i in range(min(4, len(tok_train)))]
        dp_batch = data_collator(dp_features)
        dp_out = dp_smoke(**{
            k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
            for k, v in dp_batch.items()
        })
        dp_loss = dp_out.loss
        if dp_loss is None or not torch.isfinite(dp_loss):
            raise RuntimeError(f"Multi-GPU smoke test produced invalid loss: {dp_loss}")
        print(f"Multi-GPU DataParallel smoke test: OK (loss={dp_loss.item():.4f})")
        del dp_smoke, dp_out, dp_batch, dp_features
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        # Do not silently start a long run that will immediately OOM.
        torch.cuda.empty_cache()
        raise RuntimeError(
            "Multi-GPU smoke test hit CUDA OOM even with loss-only gathering. "
            "Reduce BATCH_SIZE before starting the expensive run."
        ) from e

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
last_ckpt = (
    get_last_checkpoint(CFG.CKPT_DIR)
    if CFG.RESUME_FROM_CHECKPOINT and os.path.isdir(CFG.CKPT_DIR)
    else None
)
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
        "gujarati_obpe_tokens"  : vocab_size_before - CFG.EXPECTED_BASE_M_BERT_VOCAB,
        "romanized_new_tokens"  : num_added,
        "epochs"                : CFG.NUM_EPOCHS,
        "max_seq_len"           : CFG.MAX_SEQ_LEN,
        "mlm_probability"       : CFG.MLM_PROBABILITY,
        "learning_rate"         : CFG.LEARNING_RATE,
        "missing_mlm_head_keys_at_load": missing_mlm_head,
        "mlm_head_initialized_during_load": bool(missing_mlm_head),
        "new_tokens_requested": CFG.TARGET_NEW_TOKENS,
        "new_tokens_selected": len(new_romanized),
        "new_tokens_added": num_added,
        "new_tokens_initialized": len(initialized_words),
        "new_tokens_missing_from_train": len(missing_in_train),
    }, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 16 — Verify Saved Model
# ═══════════════════════════════════════════════════════════════════════════════
print("\nVerifying saved model...")
from transformers import AutoTokenizer as AT, AutoModelForMaskedLM as AM
chk_tok = AT.from_pretrained(CFG.OUT_MLM, local_files_only=True)
chk_mdl = AM.from_pretrained(CFG.OUT_MLM, local_files_only=True)

assert len(chk_tok) == len(tokenizer)
assert chk_mdl.bert.embeddings.word_embeddings.weight.shape[0] == len(tokenizer)
assert hasattr(chk_mdl, "cls") and hasattr(chk_mdl.cls, "predictions")
print(f"  ✓ Vocab size     : {len(chk_tok):,}")
print(f"  ✓ Embedding size : {chk_mdl.bert.embeddings.word_embeddings.weight.shape[0]:,}")

del chk_mdl
del trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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
print(f"  Base mBERT vocab     : {CFG.EXPECTED_BASE_M_BERT_VOCAB:,} tokens")
print(f"  + Gujarati OBPE      : +{vocab_size_before - CFG.EXPECTED_BASE_M_BERT_VOCAB} tokens")
print(f"  + Romanized Gujlish  : +{num_added} tokens")
print(f"  Final vocab size     : {len(tokenizer):,} tokens")
print(f"  MLM Perplexity       : {perplexity:.2f}")
print(f"  Model saved at       : {CFG.OUT_MLM}")
print()
print("  NEXT STEPS:")
print(f"  1. Upload {CFG.OUT_MLM} to Kaggle Models")
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
