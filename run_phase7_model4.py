"""
run_phase7_model4.py — XLM-RoBERTa base (xlm-roberta-base)
================================================================================
Run this on its own Kaggle T4 session, independently of model1/2/3.
Requires phase7_common.py and the data/ folder in the same directory.

WHY XLM-R:
  mBERT and MuRIL use WordPiece tokenization, which fragments romanized Gujlish
  (e.g., "jaydeepbhai" → [jay, ##dee, ##pb, ##hai]). XLM-RoBERTa uses SentencePiece
  (BPE) trained on 100-language Common Crawl data that includes romanized social-media
  text, giving it better native coverage of romanized Indic tokens.

  Literature on romanized Hindi-English code-mixing (most comparable task) shows
  XLM-R outperforms mBERT by +2–4% macro-F1.

Fully resumable — see run_phase7_model1.py header for Kaggle setup notes.
Completed folds are detected automatically and skipped.

KAGGLE SETUP:
  1. Upload this file + phase7_common.py + data/ folder to /kaggle/working/
  2. pip install -q transformers datasets scikit-learn scipy pandas numpy
  3. Runtime -> Accelerator -> GPU T4 x2 (or T4 x1)
  4. python run_phase7_model4.py
  5. IMPORTANT: download CHECKPOINT/model4/ before your session ends.

OUTPUT (in CHECKPOINT/model4/):
  label_map.json, fold_metrics.csv, fold1..5_predictions.csv
  (same schema as model1/2/3 — see phase7_common.py docstring)

Once all model scripts have finished, put all CHECKPOINT/model{1..4}/ folders
together and run evaluate_models.py to combine results.
"""

from phase7_common import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline(
        model_name="XLM-R_base",
        model_path="xlm-roberta-base",
        checkpoint_dirname="model4",
        staged_freeze=False,   # XLM-R has no new tokens — standard fine-tuning
    )
