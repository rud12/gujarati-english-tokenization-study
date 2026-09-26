"""
run_phase7_model2.py — MuRIL baseline (google/muril-base-cased)
================================================================================
Run this on its own machine / Kaggle T4 session, independently of model1 and
model3. Requires phase7_common.py and the data/ folder in the same directory.

Fully resumable — see run_phase7_model1.py header for Kaggle setup notes and
resume behavior; identical here except for the model.

OUTPUT (in CHECKPOINT/model2/):
  label_map.json, fold_metrics.csv, fold1..5_predictions.csv
  (same schema as model1 — see phase7_common.py docstring for column meanings)

Once all three model scripts have finished, zip CHECKPOINT/ (all three
model folders together) and run evaluate_models.py to combine everything.
"""

from phase7_common import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline(
        model_name="MuRIL_baseline",
        model_path="google/muril-base-cased",
        checkpoint_dirname="model2",
    )
