"""
PHASE A — Vocabulary Expansion + MLM Pre-Training (NaN-safe collator)
Adds ~800 tokens to mBERT (500 Gujarati OBPE + ~300 Romanized Gujlish)
then runs Masked Language Model continued pre-training on 51k Gujlish corpus.

KAGGLE SETUP:
  Dataset  → rud12/gujarati-english-tokenization-study
  Model    → rudrakachhia/mbert-obpe-adapted (transformers / default / 1)
  GPU      → single T4 (CUDA_VISIBLE_DEVICES=0 is forced)
  Internet → needed once for BASE_MODEL + NLTK words

V3 CHANGES
  * FIX: the adapted checkpoint has 120,045 embedding rows while its config/tokenizer
    say 120,047. BertModel.from_pretrained crashed with a size mismatch. The adapted
    weights are now loaded MANUALLY: the 120,045 existing embedding rows are copied and
    the last 2 rows keep their default init (learned during MLM training).
  * Output-bias head reference is kept consistent (head.bias is decoder.bias).
  * Docstring: OBPE additions = 500 (120,047 - 119,547).

V2 CHANGES
  1. MLM model is built from the ORIGINAL pretrained bert-base-multilingual-cased
     (trained MLM head); the OBPE-adapted encoder/embeddings are copied on top.
  2. 20 epochs, LR 5e-5, warmup 6%, no gradient checkpointing.
  3. New tokens added with single_word=True, plus Capitalized variants.
  4. New-token output biases initialised; junk tokens filtered.
  5. Held-out files excluded from the MLM corpus.
  6. Step-0 eval loss printed to verify the head transplant.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install & Imports
# ═══════════════════════════════════════════════════════════════════════════════
import os, subprocess, sys

# One T4 only: avoids DataParallel gathering huge MLM logits onto GPU 0 (OOM).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for pkg in [
    "transformers==4.57.6",
    "datasets>=2.15.0,<5.0.0",
    "accelerate>=1.1.0,<2.0.0",
    "sentencepiece",
    "nltk",
    "safetensors",
]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import gc, re, json, math, random, logging, warnings, inspect
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
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer,
    EarlyStoppingCallback, set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

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
# CELL 2 — Configuration
# ═══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent


class CFG:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR   = ROOT / "data" / "processed"
    MODEL_DIR  = ROOT / "models" / "mbert_adapted"     # OBPE tokenizer + embeddings
    BASE_MODEL = "bert-base-multilingual-cased"        # source of the PRETRAINED MLM head
    EXCLUDE_FILES = {"human_audit_sample.csv"}         # held-out; never used for MLM
    OUT_VOCAB  = "/kaggle/working/out/vocab/gujlish_800tok_model"
    OUT_MLM    = "/kaggle/working/out/model/gujlish_mlm_final"
    CKPT_DIR   = "/kaggle/working/out/cp/mlm_checkpoints"
    LOG_DIR    = "/kaggle/working/out/log/mlm_logs"

    TEXT_COL   = "text"

    # ── Vocabulary expansion ───────────────────────────────────────────────────
    TARGET_NEW_TOKENS  = 300
    MIN_WORD_FREQ      = 20
    MIN_SUBWORDS       = 2
    MAX_WORD_LEN       = 25
    SINGLE_WORD_TOKENS = True
    ADD_CAPITALIZED_VARIANTS = True
    MIN_WORD_LEN       = 3

    # ── MLM Training ──────────────────────────────────────────────────────────
    MAX_SEQ_LEN        = 128
    MLM_PROBABILITY    = 0.15
    NUM_EPOCHS         = 20
    BATCH_SIZE         = 16      # auto-reduced to 8/4 if the smoke test OOMs
    GRAD_ACCUM         = 8
    TARGET_GLOBAL_BATCH = 128
    EVAL_BATCH_SIZE     = 16
    LEARNING_RATE      = 5e-5
    WEIGHT_DECAY       = 0.01
    WARMUP_RATIO       = 0.06
    LR_SCHEDULER       = "cosine"
    FP16               = True
    EVAL_RATIO         = 0.02
    EVAL_STEPS         = 400
    SAVE_STEPS         = 400
    SAVE_TOTAL_LIMIT   = 2
    LOGGING_STEPS      = 100
    EARLY_STOP_PATIENCE = 5
    CLEAN_CHECKPOINTS  = True

    # ── Safety / reproducibility ──────────────────────────────────────────────
    EXPECTED_BASE_M_BERT_VOCAB = 119_547
    EXPECTED_OBPE_ADDITIONS    = 500      # tokenizer/config: 120,047
    RESUME_FROM_CHECKPOINT     = False


for d in [CFG.OUT_VOCAB, CFG.OUT_MLM, CFG.CKPT_DIR, CFG.LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Zero-training API preflight ────────────────────────────────────────────────
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
        f"Missing parameters: {_missing_ta}. Transformers={transformers.__version__}"
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
        f"Missing parameters: {_missing_collator}. Transformers={transformers.__version__}"
    )

_mask_params = set(inspect.signature(DataCollatorForLanguageModeling.torch_mask_tokens).parameters)
if "offset_mapping" not in _mask_params:
    raise RuntimeError(
        "DataCollatorForLanguageModeling.torch_mask_tokens API preflight failed: "
        "expected `offset_mapping` for Transformers 4.57.x. "
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
    for col in ["text", "sentence", "comment", "review", "content"]:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError(f"No text column found. Columns: {list(df.columns)}")


def load_corpus(data_dir, text_col):
    texts = []
    held_out = set()
    print(f"Scanning {data_dir} ...")
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        if fname in CFG.EXCLUDE_FILES:
            try:
                _df = pd.read_csv(os.path.join(data_dir, fname))
                _col = text_col if text_col in _df.columns else find_text_column(_df)
                held_out.update(_df[_col].dropna().astype(str).str.strip())
                print(f"  {fname:<45} EXCLUDED (held out, {len(_df):,} rows)")
            except Exception as e:
                print(f"  Could not read excluded file {fname}: {e}")
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

    texts = [s.strip() for s in texts]
    texts = list(dict.fromkeys(texts))
    texts = [s for s in texts if 10 < len(s) < 5000]
    if held_out:
        n0 = len(texts)
        texts = [s for s in texts if s not in held_out]
        print(f"Removed {n0 - len(texts):,} sentences that also appear in held-out files")
    print(f"\nTotal unique sentences: {len(texts):,}")
    return texts


all_texts = load_corpus(CFG.DATA_DIR, CFG.TEXT_COL)

print("\nSample sentences:")
for s in random.sample(all_texts, min(5, len(all_texts))):
    print(f"  → {s[:120]}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Load Pretrained MLM Model + Manually Load OBPE-Adapted Weights
# ═══════════════════════════════════════════════════════════════════════════════
from safetensors.torch import load_file


def find_model_path(primary):
    primary = Path(primary)
    if primary.exists():
        return str(primary)
    raise FileNotFoundError(
        f"OBPE-adapted model not found at: {primary}\n"
        "Place the intended OBPE mBERT model in models/mbert_adapted "
        "or update CFG.MODEL_DIR."
    )


def load_adapted_state(path):
    """Read the adapted checkpoint directly (bypasses the config-vs-weights size check).
    The checkpoint was saved from a classifier: strip 'bert.' prefix, drop classifier.*"""
    p = Path(path)
    if (p / "model.safetensors").exists():
        sd = load_file(str(p / "model.safetensors"))
    elif (p / "pytorch_model.bin").exists():
        sd = torch.load(p / "pytorch_model.bin", map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors / pytorch_model.bin in {p}")
    return {(k[5:] if k.startswith("bert.") else k): v
            for k, v in sd.items() if not k.startswith("classifier.")}


model_path = find_model_path(CFG.MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

base_tok = AutoTokenizer.from_pretrained(CFG.BASE_MODEL)
base_n = CFG.EXPECTED_BASE_M_BERT_VOCAB
if len(base_tok) != base_n:
    raise RuntimeError(f"BASE_MODEL tokenizer has {len(base_tok)} tokens, expected {base_n}.")

model, loading_info = AutoModelForMaskedLM.from_pretrained(
    CFG.BASE_MODEL,
    output_loading_info=True,
)
missing_keys = loading_info.get("missing_keys", [])
unexpected_keys = loading_info.get("unexpected_keys", [])
error_msgs = loading_info.get("error_msgs", [])
if error_msgs:
    raise RuntimeError("Model loading reported checkpoint errors:\n" + "\n".join(error_msgs[:10]))
if getattr(model.config, "model_type", None) != "bert" or not hasattr(model, "bert"):
    raise RuntimeError(f"Expected a BERT MaskedLM model, got {getattr(model.config, 'model_type', None)!r}.")

missing_mlm_head = [k for k in missing_keys
                    if k.startswith("cls.predictions.") and "decoder" not in k]
if missing_mlm_head:
    raise RuntimeError(
        "Pretrained MLM head not found in BASE_MODEL; missing: " + ", ".join(missing_mlm_head)
    )
print("✓ Pretrained MLM head loaded from", CFG.BASE_MODEL)

# Grow embeddings/decoder to the OBPE vocab (120,047), then copy adapted weights in.
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
_ref_emb   = model.bert.embeddings.word_embeddings.weight.data[:base_n].clone()
_ref_layer = model.bert.encoder.layer[-1].output.dense.weight.data.clone()

adapted_sd = load_adapted_state(model_path)
emb_key = "embeddings.word_embeddings.weight"
ckpt_rows = adapted_sd[emb_key].shape[0]
model_rows = model.bert.embeddings.word_embeddings.weight.shape[0]
print(f"Adapted checkpoint rows: {ckpt_rows:,} | model rows: {model_rows:,} "
      f"(last {model_rows - ckpt_rows} rows left at default init)")
if ckpt_rows > model_rows:
    raise RuntimeError(f"Checkpoint has MORE rows ({ckpt_rows}) than the tokenizer ({model_rows}).")

target_sd = model.bert.state_dict()   # tensors share storage with the real params
missing = [k for k in target_sd if k not in adapted_sd and "position_ids" not in k]
if missing:
    raise RuntimeError(f"Adapted encoder is missing keys: {missing[:10]}")

with torch.no_grad():
    for k, v in adapted_sd.items():
        if k not in target_sd:
            continue
        if k == emb_key:
            target_sd[k][:ckpt_rows] = v.to(target_sd[k].dtype)
        elif target_sd[k].shape == v.shape:
            target_sd[k].copy_(v)
        else:
            raise RuntimeError(f"Shape mismatch for {k}: {tuple(v.shape)} vs {tuple(target_sd[k].shape)}")

model.tie_weights()

_emb_drift   = (model.bert.embeddings.word_embeddings.weight.data[:base_n] - _ref_emb).abs().max().item()
_layer_drift = (model.bert.encoder.layer[-1].output.dense.weight.data - _ref_layer).abs().max().item()
print(f"Adapted-vs-original drift: base embeddings {_emb_drift:.3e} | last layer {_layer_drift:.3e}")
if _emb_drift > 1e-6 or _layer_drift > 1e-6:
    warn_and_continue("mbert_adapted differs from the original mBERT; the pretrained MLM head was "
                      "trained against the original encoder, so expect a higher step-0 loss.")
del adapted_sd, target_sd, _ref_emb, _ref_layer
gc.collect()


def set_new_output_bias(token_ids, piece_ids_list):
    """MLM output bias of each new token = mean bias of its original subword pieces."""
    head = model.cls.predictions
    bias = head.decoder.bias
    with torch.no_grad():
        fallback = bias.data[:base_n].mean()
        for tid, pids in zip(token_ids, piece_ids_list):
            pids = [p for p in pids if p is not None and p < tid and p != tokenizer.unk_token_id]
            bias.data[tid] = bias.data[pids].mean() if pids else fallback
    if head.bias is not bias:
        head.bias = bias      # keep both references pointing at the same parameter


# OBPE tokens (ids base_n .. len(tokenizer)-1), including the 2 rows without trained embeddings.
_obpe_ids = list(range(base_n, len(tokenizer)))
set_new_output_bias(
    _obpe_ids,
    [base_tok.convert_tokens_to_ids(base_tok.tokenize(tokenizer.convert_ids_to_tokens(i)))
     for i in _obpe_ids],
)
print(f"Initialised output bias for {len(_obpe_ids)} OBPE tokens")

if unexpected_keys:
    print(f"Note: {len(unexpected_keys)} unused BASE_MODEL keys (e.g. NSP head): {unexpected_keys[:3]}")

print("Multi-GPU DataParallel disabled: using CUDA_VISIBLE_DEVICES=0 (single T4)")

vocab_size_before = len(tokenizer)
emb_before = model.bert.embeddings.word_embeddings.weight.shape[0]

assert vocab_size_before == emb_before, (
    f"Embedding mismatch in source model: tokenizer={vocab_size_before}, "
    f"embedding={emb_before}. Wrong model loaded."
)

print(f"Tokenizer vocab size : {vocab_size_before:,}")
expected_obpe_vocab = CFG.EXPECTED_BASE_M_BERT_VOCAB + CFG.EXPECTED_OBPE_ADDITIONS
if vocab_size_before != expected_obpe_vocab:
    raise RuntimeError(
        f"Unexpected source vocabulary size: {vocab_size_before}. "
        f"Expected {expected_obpe_vocab} "
        f"({CFG.EXPECTED_BASE_M_BERT_VOCAB} base + {CFG.EXPECTED_OBPE_ADDITIONS} OBPE)."
    )
print(f"  ✓ OBPE-adapted confirmed (+{CFG.EXPECTED_OBPE_ADDITIONS} tokens over base mBERT)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Extract Top Romanized Gujlish Tokens
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading NLTK English word list...")
nltk.download("words", quiet=True)
from nltk.corpus import words as nltk_words
ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())
print(f"English word list: {len(ENGLISH_WORDS):,} words")

GUJARATI_RANGE = re.compile(r'[\u0A80-\u0AFF]')
LATIN_WORD     = re.compile(r'^[a-zA-Z][a-zA-Z0-9\'-]{2,}$')


def is_romanized_gujlish(word: str) -> bool:
    """True if word is a Latin-script Gujarati/code-mixed word (not English)."""
    w = word.strip().lower()
    if not LATIN_WORD.match(w):              return False
    if GUJARATI_RANGE.search(w):             return False
    if w in ENGLISH_WORDS:                   return False
    if len(w) < CFG.MIN_WORD_LEN:            return False
    if len(w) > CFG.MAX_WORD_LEN:            return False
    if re.search(r"(.)\1{3,}", w):           return False
    if w.endswith("'s"):                     return False
    return True


print("\nExtracting word frequencies from corpus...")
word_freq: Counter = Counter()
raw_freq: Counter = Counter()
for sentence in all_texts:
    for word in sentence.split():
        w_raw = word.strip(".,!?;:\"'()[]{}")
        if w_raw:
            raw_freq[w_raw] += 1
            word_freq[w_raw.lower()] += 1

print(f"Unique words in corpus: {len(word_freq):,}")

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
        continue
    score = math.log2(freq) * n_subwords
    candidates[word] = {
        "freq"      : freq,
        "subwords"  : n_subwords,
        "score"     : round(score, 2),
        "tokens"    : tokens,
    }

print(f"Romanized Gujlish candidate words: {len(candidates):,}")

ranked = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)

existing_vocab = set(tokenizer.vocab.keys())
new_romanized = []
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
variant_words = []
if CFG.ADD_CAPITALIZED_VARIANTS:
    _base_set = {w for w, _ in new_romanized}
    for w, _ in new_romanized:
        cap = w[:1].upper() + w[1:]
        if (cap != w and cap not in _base_set and cap not in existing_vocab
                and raw_freq.get(cap, 0) >= CFG.MIN_WORD_FREQ):
            variant_words.append(cap)
    print(f"Capitalized variants to add: {len(variant_words)}")

original_subword_pieces = {
    word: tokenizer.tokenize(word)
    for word in [w for w, _ in new_romanized] + variant_words
}

print(f"\nTop {len(new_romanized)} Romanized Gujlish tokens selected:")
print(f"{'Word':<20} {'Freq':>8} {'Subwords':>9} {'Score':>8} {'mBERT splits'}")
print("-" * 75)
for word, info in new_romanized[:30]:
    splits = " + ".join(info["tokens"])
    print(f"{word:<20} {info['freq']:>8,} {info['subwords']:>9} {info['score']:>8.2f}  {splits}")
if len(new_romanized) > 30:
    print(f"  ... and {len(new_romanized)-30} more")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Add Romanized Tokens to Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════
romanized_words = [w for w, _ in new_romanized] + variant_words

print(f"\nAdding {len(romanized_words)} Romanized Gujlish tokens to vocabulary...")
from tokenizers import AddedToken
_to_add = (
    [AddedToken(w, single_word=True) for w in romanized_words]
    if CFG.SINGLE_WORD_TOKENS else romanized_words
)
num_added = tokenizer.add_tokens(_to_add)
print(f"  Tokens successfully added: {num_added}")
if num_added != len(romanized_words):
    warn_and_continue(
        f"Tokenizer added {num_added} tokens but {len(romanized_words)} were requested. "
        "Continuing with successfully added tokens only."
    )
    romanized_words = [w for w in romanized_words if tokenizer.convert_tokens_to_ids(w) is not None]
print(f"  Vocab size: {vocab_size_before:,} → {len(tokenizer):,}")
print(f"  Total new tokens over base mBERT: +{len(tokenizer) - CFG.EXPECTED_BASE_M_BERT_VOCAB}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Resize Embedding Matrix + Mean Initialize New Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
old_embeddings = model.bert.embeddings.word_embeddings.weight.data.clone()
old_vocab_size = old_embeddings.shape[0]

model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
print(f"\nEmbedding matrix resized: {old_vocab_size:,} → {len(tokenizer):,}")

new_embedding_layer = model.bert.embeddings.word_embeddings
global_mean = old_embeddings.mean(dim=0)

initialized_words = []
with torch.no_grad():
    for word in romanized_words:
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id is None or token_id < old_vocab_size:
            warn_and_continue(
                f"Could not initialize newly added token '{word}' (id={token_id}). Skipping."
            )
            continue

        subword_pieces = original_subword_pieces.get(word, [])
        piece_ids = tokenizer.convert_tokens_to_ids(subword_pieces)
        piece_ids = [pid for pid in piece_ids if pid is not None and pid < old_vocab_size]

        if piece_ids:
            new_embedding_layer.weight.data[token_id] = old_embeddings[piece_ids].mean(dim=0)
        else:
            warn_and_continue(
                f"No valid original subword pieces found for '{word}'. "
                "Using the global embedding mean as fallback."
            )
            new_embedding_layer.weight.data[token_id] = global_mean
        initialized_words.append(word)

model.tie_weights()

set_new_output_bias(
    [tokenizer.convert_tokens_to_ids(w) for w in initialized_words],
    [tokenizer.convert_tokens_to_ids(original_subword_pieces.get(w, [])) for w in initialized_words],
)

new_ids = [tokenizer.convert_tokens_to_ids(w) for w in romanized_words]
new_matrix = new_embedding_layer.weight.data[new_ids]
pairwise_std = new_matrix.std(dim=0).mean().item()
if not math.isfinite(pairwise_std) or pairwise_std == 0.0:
    warn_and_continue(
        "New Romanized token embeddings have zero/invalid variability. "
        "Continuing anyway; this may reduce the benefit of vocabulary expansion."
    )
print(f"New-token embedding variability check: {pairwise_std:.6e}")

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

token_info = {
    "base_mbert_vocab"        : CFG.EXPECTED_BASE_M_BERT_VOCAB,
    "gujarati_obpe_tokens"    : vocab_size_before - CFG.EXPECTED_BASE_M_BERT_VOCAB,
    "romanized_gujlish_tokens": num_added,
    "total_vocab_size"        : len(tokenizer),
    "total_new_tokens"        : total_added,
    "romanized_tokens_added"  : romanized_words,
    "top30_candidates"        : [{"word": w, **info} for w, info in new_romanized[:30]],
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
tok_eval  = eval_ds.map(tokenize_fn, batched=True, batch_size=1000,
                        num_proc=2, remove_columns=["text"], desc="Eval")

print(f"Tokenized train: {len(tok_train):,}  |  eval: {len(tok_eval):,}")

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
    f"New-token corpus coverage: min={min(new_token_counts.values(), default=0)}, "
    f"max={max(new_token_counts.values(), default=0)}, "
    f"tokens_with_occurrences={sum(v > 0 for v in new_token_counts.values())}/{len(new_token_ids)}"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Data Collator
# ═══════════════════════════════════════════════════════════════════════════════
class SafeMLMCollator(DataCollatorForLanguageModeling):
    """MLM collator that guarantees at least one valid target per example, so an
    eval batch can never have 0 supervised tokens (0/0 -> NaN loss)."""

    def torch_mask_tokens(self, inputs, special_tokens_mask=None, offset_mapping=None):
        inputs, labels = super().torch_mask_tokens(
            inputs,
            special_tokens_mask=special_tokens_mask,
            offset_mapping=offset_mapping,
        )

        for row in range(labels.size(0)):
            if (labels[row] != -100).any():
                continue

            valid = torch.ones(labels.size(1), dtype=torch.bool, device=labels.device)
            if special_tokens_mask is not None:
                row_special = special_tokens_mask[row]
                if not torch.is_tensor(row_special):
                    row_special = torch.as_tensor(row_special)
                valid &= ~row_special.to(labels.device).bool()

            if not valid.any():
                print("WARNING: no non-special token available for forced MLM masking; "
                      "using position 0 and continuing.")
                valid[:] = True

            pos = int(torch.nonzero(valid, as_tuple=False)[0].item())
            labels[row, pos] = inputs[row, pos].clone()
            if self.tokenizer.mask_token_id is not None:
                inputs[row, pos] = self.tokenizer.mask_token_id
            else:
                print("WARNING: tokenizer has no mask_token_id; forcing an MLM target "
                      "without replacing the input token.")

        return inputs, labels


data_collator = SafeMLMCollator(
    tokenizer          = tokenizer,
    mlm                = True,
    mlm_probability    = CFG.MLM_PROBABILITY,
    pad_to_multiple_of = 8 if CFG.FP16 else None,
)

sample = data_collator([tok_train[i] for i in range(4)])
masked = int((sample["labels"] != -100).sum().item())
total  = sample["input_ids"].numel()
print(f"Collator OK — batch shape: {sample['input_ids'].shape}")
print(f"Masked tokens: {masked}/{total} = {masked/total:.1%}  (target ~15%)")
if masked == 0:
    print("WARNING: collator produced zero MLM targets; continuing, but this is unexpected.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — Training Arguments (+ batch-size auto-tune)
# ═══════════════════════════════════════════════════════════════════════════════
n_gpus = max(torch.cuda.device_count(), 1)
if n_gpus != 1:
    raise RuntimeError(f"Expected exactly 1 visible GPU after CUDA_VISIBLE_DEVICES=0, got {n_gpus}")

training_args_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_real_batch_smoke(batch_size):
    indices = sorted(
        range(len(tok_train)),
        key=lambda i: len(tok_train[i]["input_ids"]),
        reverse=True,
    )[:min(batch_size, len(tok_train))]
    features = [tok_train[i] for i in indices]
    batch = data_collator(features)
    batch = {k: v.to(training_args_device) if torch.is_tensor(v) else v for k, v in batch.items()}
    model.train()
    out = model(**batch)
    loss = out.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"Real-batch smoke test produced invalid loss: {loss}")
    loss.backward()
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            grad_sq += float((g.float() ** 2).sum().item())
    model.zero_grad(set_to_none=True)
    if not math.isfinite(grad_sq) or grad_sq <= 0:
        raise RuntimeError("Real-batch smoke test produced invalid/zero gradients")
    return loss.item(), math.sqrt(grad_sq), batch["input_ids"].shape


model.to(training_args_device)
chosen_batch = None
for candidate in [CFG.BATCH_SIZE, 8, 4]:
    try:
        smoke_loss, smoke_grad_norm, smoke_shape = run_real_batch_smoke(candidate)
        chosen_batch = candidate
        print(
            f"Real-batch smoke test: OK | batch={candidate} | shape={tuple(smoke_shape)} "
            f"| loss={smoke_loss:.4f} | grad_norm={smoke_grad_norm:.4f}"
        )
        break
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"WARNING: batch={candidate} hit CUDA OOM during preflight; trying a smaller batch.")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if chosen_batch is None:
    raise RuntimeError("No safe per-device batch size (16, 8, or 4) fit the single T4 during preflight.")

CFG.BATCH_SIZE = chosen_batch
if CFG.TARGET_GLOBAL_BATCH % CFG.BATCH_SIZE != 0:
    raise RuntimeError(
        f"TARGET_GLOBAL_BATCH={CFG.TARGET_GLOBAL_BATCH} is not divisible by chosen batch {CFG.BATCH_SIZE}."
    )
CFG.GRAD_ACCUM = CFG.TARGET_GLOBAL_BATCH // CFG.BATCH_SIZE

approx_global_batch = CFG.BATCH_SIZE * CFG.GRAD_ACCUM
approx_steps_per_epoch = math.ceil(len(tok_train) / approx_global_batch)
approx_total_steps = approx_steps_per_epoch * CFG.NUM_EPOCHS


def verify_eval_batch_has_targets():
    checks = [0, min(4, len(tok_eval) - 1), min(8, len(tok_eval) - 1)]
    for idx in checks:
        batch = data_collator([tok_eval[idx]])
        target_count = int((batch["labels"] != -100).sum().item())
        if target_count == 0:
            print("WARNING: evaluation collator produced zero MLM targets; continuing.")
        else:
            print(f"Evaluation MLM-target check: OK (sample {idx}, targets={target_count})")


verify_eval_batch_has_targets()

print(f"\nTraining schedule (single-T4 configuration):")
print(f"  Visible GPUs         : {n_gpus}")
print(f"  Per-device batch     : {CFG.BATCH_SIZE}")
print(f"  Gradient accumulation: {CFG.GRAD_ACCUM}")
print(f"  Effective batch      : {approx_global_batch}")
print(f"  Approx total steps   : {approx_total_steps:,}")
print(f"  Warmup ratio         : {CFG.WARMUP_RATIO:.2%}")

training_args = TrainingArguments(
    output_dir                   = CFG.CKPT_DIR,
    overwrite_output_dir         = True,
    num_train_epochs             = CFG.NUM_EPOCHS,
    per_device_train_batch_size  = CFG.BATCH_SIZE,
    per_device_eval_batch_size   = CFG.EVAL_BATCH_SIZE,
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
    fp16_full_eval               = False,
    gradient_checkpointing       = False,
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

# Finite-evaluation smoke test (catches NaN eval loss before the expensive run).
model.eval()
_eval_features = [tok_eval[i] for i in range(min(CFG.EVAL_BATCH_SIZE, len(tok_eval)))]
_eval_batch = data_collator(_eval_features)
_eval_batch = {
    k: v.to(training_args_device) if torch.is_tensor(v) else v
    for k, v in _eval_batch.items()
}
with torch.no_grad():
    _eval_out = model(**_eval_batch)
_eval_loss_smoke = _eval_out.loss
if _eval_loss_smoke is None or not torch.isfinite(_eval_loss_smoke):
    raise RuntimeError(
        f"Preflight evaluation loss is not finite: {_eval_loss_smoke}. "
        "The run was stopped before expensive MLM training."
    )
print(f"Evaluation loss smoke test: OK (loss={_eval_loss_smoke.item():.4f})")
del _eval_out, _eval_batch, _eval_features
model.train()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Build Trainer
# ═══════════════════════════════════════════════════════════════════════════════
trainer = Trainer(
    model            = model,
    args             = training_args,
    train_dataset    = tok_train,
    eval_dataset     = tok_eval,
    processing_class = tokenizer,
    data_collator    = data_collator,
    callbacks        = [
        EarlyStoppingCallback(
            early_stopping_patience  = CFG.EARLY_STOP_PATIENCE,
            early_stopping_threshold = 0.001,
        )
    ],
)
print("Trainer ready.")

# Step-0 evaluation: with the pretrained head this should be far below ~12.9 (random head).
_init_eval = trainer.evaluate()
initial_eval_loss = _init_eval["eval_loss"]
print(f"Step-0 eval loss: {initial_eval_loss:.4f}  (ppl {math.exp(initial_eval_loss):.1f})")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STARTING MLM PRE-TRAINING")
print(f"  Vocab size : {len(tokenizer):,}  (+{total_added} over base mBERT)")
print(f"  Dataset    : {len(tok_train):,} train  |  {len(tok_eval):,} eval")
print(f"  Epochs     : {CFG.NUM_EPOCHS}")
print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65 + "\n")

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

import shutil
src = os.path.join(CFG.OUT_VOCAB, "token_expansion_info.json")
shutil.copy(src, CFG.OUT_MLM)

train_metrics = train_result.metrics
train_metrics["train_samples"] = len(tok_train)
trainer.save_metrics("train", train_metrics)

print(f"Final model saved to: {CFG.OUT_MLM}")

if CFG.CLEAN_CHECKPOINTS:
    shutil.rmtree(CFG.CKPT_DIR, ignore_errors=True)
    print("Removed intermediate checkpoints.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15 — Evaluate + Perplexity
# ═══════════════════════════════════════════════════════════════════════════════
eval_metrics = trainer.evaluate()
eval_loss    = eval_metrics["eval_loss"]
perplexity   = math.exp(eval_loss)

print(f"\n{'=' * 50}")
print("  FINAL RESULTS")
print(f"{'=' * 50}")
print(f"  Train loss   : {train_metrics.get('train_loss', float('nan')):.4f}")
print(f"  Eval loss    : {eval_loss:.4f}")
print(f"  Perplexity   : {perplexity:.2f}")
print(f"  Train time   : {train_metrics.get('train_runtime', 0)/3600:.2f} hrs")
print(f"  Vocab size   : {len(tokenizer):,}  (+{total_added} new tokens)")
print(f"{'=' * 50}")

if   perplexity < 5:  print("  Excellent — very low perplexity.")
elif perplexity < 15: print("  Good — reasonable for code-mixed text.")
elif perplexity < 30: print("  Fair — consider more epochs or data.")
else:                 print("  Note — high perplexity; check LR and data.")

with open(os.path.join(CFG.OUT_MLM, "mlm_metrics.json"), "w") as f:
    json.dump({
        "train_loss"            : train_metrics.get("train_loss"),
        "eval_loss"             : eval_loss,
        "initial_eval_loss"     : initial_eval_loss,
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
        "adapted_ckpt_embedding_rows": ckpt_rows,
        "rows_left_at_default_init"  : model_rows - ckpt_rows,
        "missing_mlm_head_keys_at_load": missing_mlm_head,
        "mlm_head_initialized_during_load": bool(missing_mlm_head),
        "new_tokens_requested"  : CFG.TARGET_NEW_TOKENS,
        "new_tokens_selected"   : len(new_romanized),
        "new_tokens_added"      : num_added,
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
    "Hu tane bahu [MASK] karu chhu.",
    "Bhai, aa video bahu [MASK] hato.",
]

print("\n" + "=" * 55)
print("  FILL-MASK SANITY CHECK")
print("=" * 55)
for sent in test_sents:
    try:
        results = fill(sent)
        print(f"\nInput : {sent}")
        for r in results[:5]:
            print(f"  [{r['score']:.3f}] {r['sequence']}")
    except Exception as e:
        print(f"Skipped '{sent}': {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 18 — Summary and Next Steps
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DONE — SUMMARY")
print("=" * 65)
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
print("=" * 65)

print("\nFiles in output:")
for f in sorted(os.listdir(CFG.OUT_MLM)):
    sz = os.path.getsize(os.path.join(CFG.OUT_MLM, f)) / 1024 / 1024
    print(f"  {f:<40} {sz:>8.2f} MB")