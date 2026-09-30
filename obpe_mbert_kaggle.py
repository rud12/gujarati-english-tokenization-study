"""
obpe_mbert_kaggle.py — Recreate the OBPE-adapted mBERT model
=============================================================
Produces `out/mbert_adapted/` that is 100% compatible with
old2mlm_800tok_kaggle.py (expects mbert_adapted with 120,045 vocab).

What this script does
---------------------
1. Loads `bert-base-multilingual-cased` (needs Kaggle internet ON, or point
   BASE_MODEL to an existing local copy).
2. Scans the same corpus CSVs as the MLM script.
3. Finds Gujarati-script words that mBERT fragments most severely
   (score = log2(freq) × n_subwords) — the OBPE fragmentation metric.
4. Adds exactly 498 of them to the vocabulary.
5. Resizes the embedding matrix so config, tokenizer, and weights all agree
   on vocab_size = 119,547 + 498 = 120,045.

KAGGLE SETUP
------------
  Internet : ON  (only needed once to download bert-base-multilingual-cased)
              OR set BASE_MODEL to a local Kaggle model path
  GPU      : not required (no training)
  Dataset  : rud12/gujarati-english-tokenization-study (same as MLM script)

Run BEFORE old2mlm_800tok_kaggle.py in the same notebook, then:
  - Download out/mbert_adapted/ from Kaggle output
  - Upload as a new version of your mbert_adapted Kaggle dataset
  - Run old2mlm_800tok_kaggle.py pointing at models/mbert_adapted/
"""

import os, re, gc, json, math, random, logging, subprocess, sys
import warnings
warnings.filterwarnings("ignore")

for pkg in ["transformers==4.57.6", "sentencepiece"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, BertForMaskedLM, set_seed

logging.basicConfig(level=logging.WARNING)
set_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT = Path("/kaggle/working/gujarati-english-tokenization-study")

DATA_DIR   = ROOT / "data" / "processed"    # same as old2mlm_800tok_kaggle.py
TEXT_COL   = "text"

# Source mBERT — HuggingFace id (internet ON) or local path
BASE_MODEL = "google-bert/bert-base-multilingual-cased"
# To use a local copy, uncomment:
# BASE_MODEL = str(ROOT / "models" / "bert_base_multilingual_cased")

OUT_DIR    = Path("/kaggle/working/gujarati-english-tokenization-study/models/mbert_adapted")   # <- upload this to Kaggle

# Must match EXPECTED_OBPE_ADDITIONS in old2mlm_800tok_kaggle.py
TARGET_OBPE_TOKENS = 498

MIN_GUJARATI_FREQ  = 3    # ignore Gujarati words seen fewer times
MIN_SUBWORDS       = 2    # only add words currently split into >= 2 pieces
MAX_WORD_LEN       = 20
MIN_WORD_LEN       = 2

EXPECTED_BASE_VOCAB = 119_547   # bert-base-multilingual-cased

# Gujarati script Unicode block U+0A80-U+0AFF
GUJARATI_WORD = re.compile(r'[\u0A80-\u0AFF][\u0A80-\u0AFF\u200C\u200D]*')

# ── Step 1: Load corpus ────────────────────────────────────────────────────────
print("=" * 60)
print("  OBPE mBERT Recreation")
print("=" * 60)

print(f"\n[1/5] Scanning corpus at {DATA_DIR} ...")
texts = []
for csv_path in sorted(DATA_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        if TEXT_COL not in df.columns:
            continue
        rows = df[TEXT_COL].dropna().astype(str).tolist()
        texts.extend(rows)
        print(f"  {csv_path.name:<45} {len(rows):,} rows")
    except Exception as e:
        print(f"  SKIP {csv_path.name}: {e}")

texts = list(dict.fromkeys(texts))
texts = [s for s in texts if 2 < len(s) < 5000]
print(f"  Total unique sentences: {len(texts):,}")

# ── Step 2: Load base mBERT ───────────────────────────────────────────────────
print(f"\n[2/5] Loading base mBERT from: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model     = BertForMaskedLM.from_pretrained(BASE_MODEL)
model.eval()

actual_base = len(tokenizer)
print(f"  Base vocab size : {actual_base:,}")
if actual_base != EXPECTED_BASE_VOCAB:
    print(f"  NOTE: expected {EXPECTED_BASE_VOCAB:,}, got {actual_base:,}.")
    EXPECTED_BASE_VOCAB = actual_base

# ── Step 3: Extract Gujarati word frequencies ─────────────────────────────────
print(f"\n[3/5] Extracting Gujarati-script word frequencies ...")
guj_freq: Counter = Counter()
for text in texts:
    for match in GUJARATI_WORD.finditer(text):
        word = match.group()
        if MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN:
            guj_freq[word] += 1

print(f"  Unique Gujarati words: {len(guj_freq):,}")

# ── Step 4: Score by fragmentation x frequency (OBPE metric) ─────────────────
print(f"\n[4/5] Scoring by OBPE fragmentation metric ...")
existing_vocab = set(tokenizer.get_vocab().keys())

candidates = []
for word, freq in guj_freq.items():
    if freq < MIN_GUJARATI_FREQ:
        continue
    if word in existing_vocab:
        continue
    pieces = tokenizer.tokenize(word)
    n_sub = len(pieces)
    if n_sub < MIN_SUBWORDS:
        continue
    score = math.log2(max(freq, 1)) * n_sub
    candidates.append((word, freq, n_sub, score, pieces))

candidates.sort(key=lambda x: x[3], reverse=True)
print(f"  OBPE candidates: {len(candidates):,}")

selected = candidates[:TARGET_OBPE_TOKENS]

print(f"\n  Top 20 selected Gujarati tokens:")
print(f"  {'Word':<25} {'Freq':>6} {'Sub':>4} {'Score':>8}  mBERT fragments")
print("  " + "-" * 72)
for word, freq, n_sub, score, pieces in selected[:20]:
    print(f"  {word:<25} {freq:>6,} {n_sub:>4} {score:>8.2f}  {' + '.join(pieces)}")
if len(selected) > 20:
    print(f"  ... and {len(selected)-20} more")

if len(selected) < TARGET_OBPE_TOKENS:
    shortfall = TARGET_OBPE_TOKENS - len(selected)
    print(f"\n  WARNING: only {len(selected)} candidates found "
          f"({shortfall} short of target={TARGET_OBPE_TOKENS}).")
    print(f"  The corpus may not contain enough Gujarati-script text.")
    print(f"  Proceeding with {len(selected)} tokens.")
    print(f"  -> Update EXPECTED_OBPE_ADDITIONS={len(selected)} in "
          f"old2mlm_800tok_kaggle.py if needed.")

# ── Step 5: Add tokens, resize, save ─────────────────────────────────────────
print(f"\n[5/5] Adding {len(selected)} tokens and saving ...")
new_tokens = [w for w, *_ in selected]
num_added  = tokenizer.add_tokens(new_tokens)
print(f"  Tokens added: {num_added}")

model.resize_token_embeddings(len(tokenizer))
print(f"  Embedding resized: {EXPECTED_BASE_VOCAB:,} -> {len(tokenizer):,}")

# CRITICAL consistency check
emb_size = model.bert.embeddings.word_embeddings.weight.shape[0]
tok_size  = len(tokenizer)
assert emb_size == tok_size, (
    f"BUG: embedding {emb_size} != tokenizer {tok_size}"
)

total_new = tok_size - EXPECTED_BASE_VOCAB
print(f"\n  Base mBERT          : {EXPECTED_BASE_VOCAB:,}")
print(f"  + Gujarati OBPE     : +{total_new}")
print(f"  Final vocab size    : {tok_size:,}  ✓")
print(f"  Embedding matrix    : {tok_size:,} x {model.config.hidden_size}  ✓")

# Check MLM script compatibility
if tok_size in (EXPECTED_BASE_VOCAB + 498, EXPECTED_BASE_VOCAB + 500):
    print(f"  ✓ Compatible with old2mlm_800tok_kaggle.py")
else:
    print(f"  ACTION REQUIRED: update EXPECTED_OBPE_ADDITIONS={total_new} "
          f"in old2mlm_800tok_kaggle.py")

OUT_DIR.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(str(OUT_DIR))
model.save_pretrained(str(OUT_DIR))

summary = {
    "base_model"           : BASE_MODEL,
    "base_vocab_size"      : EXPECTED_BASE_VOCAB,
    "gujarati_obpe_added"  : num_added,
    "final_vocab_size"     : tok_size,
    "corpus_sentences"     : len(texts),
    "gujarati_unique_words": len(guj_freq),
    "candidates_scored"    : len(candidates),
    "top_tokens"           : new_tokens[:50],
}
with open(OUT_DIR / "obpe_info.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n  Saved to: {OUT_DIR}")

# List output files
import subprocess as _sp
result = _sp.run(["ls", "-lh", str(OUT_DIR)], capture_output=True, text=True)
print(result.stdout)

print("NEXT STEPS:")
print(f"  1. Download {OUT_DIR}/ from Kaggle output")
print(f"  2. Upload as a new version of your mbert_adapted Kaggle dataset")
print(f"  3. Run old2mlm_800tok_kaggle.py  (reads from models/mbert_adapted/)")
print(f"     OR set CFG.MODEL_DIR = Path('{OUT_DIR}') to skip the upload step")

# Tokenization sanity check on common Gujarati words
print(f"\n  TOKENIZATION SANITY CHECK:")
test_words = ["ગુજરાત", "ભારત", "ગ્રામ", "સૌંદર્ય", "જ્ઞાન", "Jaydeepbhai"]
for word in test_words:
    pieces = tokenizer.tokenize(word)
    in_vocab = tokenizer.convert_tokens_to_ids(word) != tokenizer.unk_token_id
    added = word in new_tokens
    tag = "[OBPE]" if added else ("[exist]" if in_vocab else "")
    print(f"  {word:<20} {tag:<8} -> {' + '.join(pieces)}")

print(f"\nDone. vocab_size={tok_size:,}")
