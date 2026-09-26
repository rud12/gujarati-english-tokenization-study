"""
run_phase7_model1.py — mBERT baseline (bert-base-multilingual-cased)
================================================================================
Run this on its own machine / Kaggle T4 session, independently of model2 and
model3. Requires phase7_common.py and the data/ folder in the same directory.

Fully resumable — if your Kaggle session disconnects mid-run, just restart
this exact script in a fresh session with the same CHECKPOINT/ folder
present; completed folds are detected automatically and skipped.

KAGGLE SETUP:
  1. Upload/clone the repo so this file, phase7_common.py, and data/processed/
     are all present in /kaggle/working/
  2. pip install -q transformers datasets scikit-learn scipy pandas numpy
  3. Runtime -> Accelerator -> GPU T4 x2 (or T4 x1)
  4. python run_phase7_model1.py
  5. IMPORTANT: before your session ends, download or "Save Version" the
     CHECKPOINT/model1/ folder — Kaggle does not persist it otherwise.

OUTPUT (in CHECKPOINT/model1/):
  label_map.json          — label2id / dataset info for this run
  fold_metrics.csv         — per-fold accuracy / macro-F1 / precision / recall
  fold1_predictions.csv    — per-example text, true/pred label, correctness,
  ...fold5_predictions.csv   confidence, fragmentation_ratio, is_code_mixed

Once all three model scripts have finished, zip CHECKPOINT/ (all three
model folders together) and run evaluate_models.py to combine everything.
"""

from phase7_common import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline(
        model_name="mBERT_baseline",
        model_path="bert-base-multilingual-cased",
        checkpoint_dirname="model1",
    )
