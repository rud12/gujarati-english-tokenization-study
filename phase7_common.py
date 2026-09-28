"""
phase7_common.py
=================
Shared utilities for the Phase 7 model-comparison pipeline. Imported by
run_phase7_model1.py (mBERT baseline), run_phase7_model2.py (MuRIL baseline),
and run_phase7_model3.py (mBERT adapted, OBPE + mean-init).

WHY A SHARED MODULE:
All three per-model scripts MUST see the exact same row order and the exact
same StratifiedKFold split (same seed, same n_splits) — otherwise fold 3 for
model A and fold 3 for model B won't contain the same validation sentences,
which silently invalidates every paired significance test (t-test, Wilcoxon,
Cohen's d) and the cross-model fragmentation-stratified comparison in
evaluate_models.py. Import load_data() and get_fold_split() from here in
every script — never re-implement them locally.

MEASURES THIS MODULE CAPTURES (so you never have to re-run training just to
get a number you forgot):
  - Per-fold aggregate accuracy / macro-F1 / macro-precision / macro-recall
  - Per-example prediction, correctness, and softmax confidence
  - Per-example fragmentation ratio (avg subwords/word, via THIS model's own
    tokenizer) — needed for the fragmentation-confidence correlation
  - Per-example is_code_mixed flag (Gujarati script + Latin script both
    present) — needed for the Welch's t-test (code-mixed vs monolingual
    fragmentation)
  - Per-fold training time, for your methods section

Kaggle T4 notes:
  - Batch size auto-scales to VRAM (T4 16GB -> batch 32)
  - fp16 enabled automatically on CUDA
  - Fully resumable: if a Kaggle session disconnects mid-run, just restart
    the same script on a fresh session with the same CHECKPOINT/ folder
    present — completed folds are detected and skipped automatically.
  - Kaggle sessions do NOT persist your working directory after the session
    ends. Before your session times out (or when a model finishes), download
    or "Save Version" the CHECKPOINT/modelN/ folder, or it will be lost.
"""

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "CHECKPOINT"
CHECKPOINT_DIR.mkdir(exist_ok=True)

os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "hf_cache"))

# ── Global config — identical across all three model scripts, do not change
#    between runs or the fold splits will no longer line up ─────────────────
RANDOM_SEED = 42
N_FOLDS = 3
N_EPOCHS = 5
N_EPOCHS_FREEZE = 1   # epochs to train ONLY embeddings+head when staged_freeze=True
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 96

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

GUJARATI_UNICODE_RANGE = (0x0A80, 0x0AFF)  # Unicode block for Gujarati script


def detect_hardware():
    """Auto-detect GPU/CPU and return device settings. Tuned for Kaggle T4 (16GB)."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = 32 if vram_gb >= 10 else 16
        use_fp16 = True
        use_cpu = False
    elif torch.backends.mps.is_available():
        device, gpu_name, vram_gb = "mps", "Apple MPS", 0.0
        batch_size, use_fp16, use_cpu = 8, False, False
    else:
        device, gpu_name, vram_gb = "cpu", "CPU", 0.0
        batch_size, use_fp16, use_cpu = 16, False, True
    return dict(device=device, gpu_name=gpu_name, vram_gb=vram_gb,
                batch_size=batch_size, use_fp16=use_fp16, use_cpu=use_cpu)


def is_code_mixed(text: str) -> bool:
    """
    True if the sentence contains BOTH Gujarati-script characters AND Latin
    letters — i.e. is actually code-mixed rather than purely one script.
    Used for the Welch's t-test comparing fragmentation of code-mixed vs
    monolingual sentences (mirrors the mBERT/Hindi-Urdu-English study).
    """
    has_gujarati = any(GUJARATI_UNICODE_RANGE[0] <= ord(ch) <= GUJARATI_UNICODE_RANGE[1] for ch in text)
    has_latin = bool(re.search(r"[A-Za-z]", text))
    return bool(has_gujarati and has_latin)


def compute_fragmentation_ratio(text: str, tokenizer) -> float:
    """
    Avg subwords-per-word for this sentence under `tokenizer`, matching the
    fragmentation metric used throughout the project. Splits on whitespace,
    tokenizes each word separately, and averages piece counts.
    """
    words = text.split()
    if not words:
        return 0.0
    piece_counts = [max(1, len(tokenizer.tokenize(w))) for w in words]
    return float(np.mean(piece_counts))


def load_data():
    """
    Load and standardize the full dataset EXACTLY the same way in every
    model script. Returns (df, label2id, id2label, num_labels, dataset_name).
    """
    dataset_1 = DATA_PROCESSED / "clean_sample_3000.csv"
    # dataset_2 = DATA_PROCESSED / "expanded_dev_subset_full.csv"
    dev_subset = DATA_PROCESSED / "dev_subset.csv"

    frames = []
    if dataset_1.exists():
        frames.append(pd.read_csv(dataset_1, encoding="utf-8-sig", low_memory=False))
    # if dataset_2.exists():
    #     frames.append(pd.read_csv(dataset_2, encoding="utf-8-sig", low_memory=False))

    if frames:
        df_raw = pd.concat(frames, ignore_index=True)
        col = "sentiment_label" if "sentiment_label" in df_raw.columns else "label"
        df = df_raw[df_raw[col].notna()].copy()
        dataset_name = "merged_50k" if len(frames) > 1 else dataset_1.name
    # elif dev_subset.exists():
    #     df = pd.read_csv(dev_subset, encoding="utf-8", low_memory=False)
    #     dataset_name = "dev_subset_3000"
    else:
        from datasets import load_dataset
        ds = load_dataset("ShrutiPatel3011/gujarati-english-codemixed-sentiment")
        frames = [split.to_pandas() for split in ds.values()]
        df = pd.concat(frames, ignore_index=True)
        dataset_name = "hf_download"

    if "text" not in df.columns:
        text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[0])
        df = df.rename(columns={text_col: "text"})
    if "sentiment_label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "sentiment_label"})

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].notna() & (df["text"] != "") & df["sentiment_label"].notna()].copy()

    # De-duplicate BEFORE the split. filtered_clean.csv and
    # expanded_dev_subset_full.csv may overlap; duplicate sentences would let
    # the same sentence land in both train and val within a fold (leakage)
    # and would silently inflate every downstream metric.
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    n_dropped = before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} duplicate sentence(s) across source files.")

    label2id = {lbl: i for i, lbl in enumerate(sorted(df["sentiment_label"].unique()))}
    id2label = {v: k for k, v in label2id.items()}
    df["label"] = df["sentiment_label"].map(label2id)
    num_labels = len(label2id)

    return df, label2id, id2label, num_labels, dataset_name


def get_fold_split(texts_all, labels_all):
    """Returns the SAME 5-fold StratifiedKFold split for every model script."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return list(skf.split(texts_all, labels_all))


def save_label_map(model_dir: Path, model_name: str, label2id: dict, dataset_name: str, n_rows: int):
    with open(model_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "label2id": label2id,
            "dataset_name": dataset_name,
            "n_rows": n_rows,
            "n_folds": N_FOLDS,
            "n_epochs": N_EPOCHS,
            "random_seed": RANDOM_SEED,
        }, f, indent=2, ensure_ascii=False)


def run_training_pipeline(model_name: str, model_path: str, checkpoint_dirname: str,
                          staged_freeze: bool = False):
    """
    Full 5-fold loop for ONE model: trains, evaluates, and saves per-fold
    aggregate metrics + full per-example predictions.

    staged_freeze=True (use for the vocab-adapted model only):
      Epoch 1   — freeze all BERT encoder layers; only the embedding table
                  (including the 500 new rows) and the classifier head are updated.
                  This lets new token embeddings settle before attention layers adapt.
      Epochs 2+ — all layers unfrozen, standard fine-tuning continues.
    Call from run_phase7_model{1,2,3}.py.
    """
    import time
    import datetime
    import shutil
    import gc
    import csv
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer, DataCollatorWithPadding)
    from datasets import Dataset as HFDataset

    hw = detect_hardware()
    print(f"[{model_name}] Hardware: {hw['gpu_name']}  (batch={hw['batch_size']}, fp16={hw['use_fp16']})")

    model_dir = CHECKPOINT_DIR / checkpoint_dirname
    model_dir.mkdir(parents=True, exist_ok=True)

    df, label2id, id2label, num_labels, dataset_name = load_data()
    save_label_map(model_dir, model_name, label2id, dataset_name, len(df))
    print(f"[{model_name}] Dataset: {dataset_name} ({len(df):,} rows, {num_labels} classes)")

    texts_all = df["text"].values
    labels_all = df["label"].values
    splits = get_fold_split(texts_all, labels_all)

    class WeightedTrainer(Trainer):
        """Class-weighted CrossEntropyLoss — fixes the minority-class imbalance."""
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights.to(self.args.device)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def tokenize_fn(texts, labels, tokenizer):
        enc = tokenizer(list(texts), truncation=True, padding=False,
                         max_length=MAX_SEQ_LEN, return_tensors=None)
        enc["labels"] = list(labels)
        return HFDataset.from_dict(enc)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
            "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        }

    fold_metrics_path = model_dir / "fold_metrics.csv"
    fold_fields = ["fold", "accuracy", "macro_precision", "macro_recall", "macro_f1", "train_time_min", "timestamp"]

    def fold_done(fold_num):
        pred_path = model_dir / f"fold{fold_num}_predictions.csv"
        if not pred_path.exists() or not fold_metrics_path.exists():
            return False
        try:
            fm = pd.read_csv(fold_metrics_path)
            return fold_num in fm["fold"].values
        except Exception:
            return False

    def save_fold_metrics(fold_num, acc, prec, rec, f1, train_time):
        write_hdr = not fold_metrics_path.exists()
        with open(fold_metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fold_fields)
            if write_hdr:
                w.writeheader()
            w.writerow({"fold": fold_num, "accuracy": round(acc, 6),
                        "macro_precision": round(prec, 6), "macro_recall": round(rec, 6),
                        "macro_f1": round(f1, 6), "train_time_min": round(train_time, 2),
                        "timestamp": datetime.datetime.now().isoformat()})

    total_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_num = fold_idx + 1

        if fold_done(fold_num):
            print(f"[{model_name}] Fold {fold_num}/{N_FOLDS} — already done, skipping.")
            continue

        train_texts, val_texts = texts_all[train_idx], texts_all[val_idx]
        train_labels, val_labels = labels_all[train_idx], labels_all[val_idx]
        print(f"\n[{model_name}] ==== Fold {fold_num}/{N_FOLDS} "
              f"(train={len(train_texts):,} val={len(val_texts):,}) ====")

        weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        weights_tensor = torch.FloatTensor(weights)

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels, ignore_mismatched_sizes=True,
            id2label=id2label, label2id=label2id,
        )

        train_ds = tokenize_fn(train_texts, train_labels, tokenizer)
        val_ds = tokenize_fn(val_texts, val_labels, tokenizer)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        tmp_dir = str(ROOT / f"tmp_{checkpoint_dirname}_fold{fold_num}")

        # ── Staged freeze: epoch 1 = only embeddings + classifier head ─────
        # Applies only for vocab-adapted model (staged_freeze=True).
        # Freezes all BERT encoder layers so the 500 new token embeddings
        # learn to occupy good positions BEFORE the attention layers adapt.
        if staged_freeze:
            print(f"[{model_name}] Staged freeze: training embeddings+head only for "
                  f"{N_EPOCHS_FREEZE} epoch(s), then unfreezing...")
            # Freeze everything except embeddings and classifier
            for name, param in model.named_parameters():
                is_embedding = "embeddings" in name
                is_classifier = "classifier" in name
                param.requires_grad = bool(is_embedding or is_classifier)

            # Phase 1: frozen body
            freeze_args = TrainingArguments(
                output_dir=tmp_dir + "_freeze",
                num_train_epochs=N_EPOCHS_FREEZE,
                per_device_train_batch_size=hw["batch_size"],
                per_device_eval_batch_size=hw["batch_size"],
                learning_rate=LEARNING_RATE * 5,   # higher LR for embeddings
                weight_decay=0.01,
                eval_strategy="no", save_strategy="no", load_best_model_at_end=False,
                seed=RANDOM_SEED, logging_steps=50, report_to="none",
                use_cpu=hw["use_cpu"], fp16=hw["use_fp16"],
                dataloader_num_workers=0,
                warmup_steps=0.1,
            )
            freeze_trainer = WeightedTrainer(
                class_weights=weights_tensor, model=model, args=freeze_args,
                train_dataset=train_ds, eval_dataset=val_ds,
                processing_class=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
            )
            freeze_trainer.train()
            del freeze_trainer
            shutil.rmtree(tmp_dir + "_freeze", ignore_errors=True)

            # Unfreeze all parameters for the remaining epochs
            for param in model.parameters():
                param.requires_grad = True
            print(f"[{model_name}] Unfroze all layers — continuing for {N_EPOCHS - N_EPOCHS_FREEZE} epochs.")

        # ── Main training (all layers) ───────────────────────────────────────
        remaining_epochs = N_EPOCHS - N_EPOCHS_FREEZE if staged_freeze else N_EPOCHS
        args = TrainingArguments(
            output_dir=tmp_dir, num_train_epochs=remaining_epochs,
            per_device_train_batch_size=hw["batch_size"], per_device_eval_batch_size=hw["batch_size"],
            learning_rate=LEARNING_RATE, weight_decay=0.01,
            # FIX Issue-1: eval per epoch + save best checkpoint so we score the
            # best epoch, not necessarily the last one.
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1", greater_is_better=True,
            seed=RANDOM_SEED, logging_steps=50, report_to="none",
            use_cpu=hw["use_cpu"], fp16=hw["use_fp16"],
            dataloader_num_workers=0, gradient_accumulation_steps=max(1, 32 // hw["batch_size"]),
            warmup_steps=0.1,
        )

        trainer = WeightedTrainer(
            class_weights=weights_tensor, model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
            processing_class=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
        )

        print(f"[{model_name}] Training fold {fold_num}...", flush=True)
        trainer.train()

        # FIX Issue-2: single predict() call — metrics AND logits come from the
        # same (best-checkpoint) forward pass, so they are guaranteed consistent.
        # Previously evaluate() + predict() were two separate passes which would
        # diverge if load_best_model_at_end swaps checkpoints between calls.
        print(f"[{model_name}] Evaluating + predicting fold {fold_num} (single pass)...", flush=True)
        pred_output = trainer.predict(val_ds)
        logits      = pred_output.predictions                         # (N, num_labels)
        true_labels = pred_output.label_ids                           # (N,)
        probs       = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_labels_idx = probs.argmax(axis=-1)
        confidences     = probs.max(axis=-1)

        acc  = accuracy_score(true_labels, pred_labels_idx)
        prec = precision_score(true_labels, pred_labels_idx, average="macro", zero_division=0)
        rec  = recall_score(true_labels, pred_labels_idx, average="macro", zero_division=0)
        f1   = f1_score(true_labels, pred_labels_idx, average="macro", zero_division=0)

        print(f"[{model_name}] Computing fragmentation ratios for fold {fold_num}...")
        frag_ratios = [compute_fragmentation_ratio(t, tokenizer) for t in val_texts]
        code_mixed_flags = [is_code_mixed(t) for t in val_texts]

        pred_rows = []
        for text, true_l, pred_l, conf, frag, cm in zip(
            val_texts, val_labels, pred_labels_idx, confidences, frag_ratios, code_mixed_flags
        ):
            pred_rows.append({
                "text": text,
                "true_label": id2label[int(true_l)],
                "pred_label": id2label[int(pred_l)],
                "correct": bool(true_l == pred_l),
                "confidence": round(float(conf), 6),
                "fragmentation_ratio": round(float(frag), 6),
                "is_code_mixed": cm,
                "fold": fold_num,
            })
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(model_dir / f"fold{fold_num}_predictions.csv", index=False, encoding="utf-8-sig")

        train_time = (time.time() - t0) / 60
        save_fold_metrics(fold_num, acc, prec, rec, f1, train_time)
        print(f"[{model_name}] Fold {fold_num}: F1={f1:.4f}  Acc={acc:.4f}  ({train_time:.1f} min)")

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_time = (time.time() - total_start) / 60
    print(f"\n[{model_name}] ALL FOLDS COMPLETE. Total time: {total_time:.1f} min")
    print(f"[{model_name}] Results saved to {model_dir}/")
    print(f"[{model_name}] -> download/persist CHECKPOINT/{checkpoint_dirname}/ now, "
          f"then combine with the other two models' checkpoints for evaluate_models.py")
