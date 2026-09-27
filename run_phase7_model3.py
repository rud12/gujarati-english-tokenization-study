"""
run_phase7_model3.py — mBERT adapted (OBPE vocabulary + mean-init embeddings)
================================================================================
Run this on its own machine / Kaggle T4 session, independently of model1 and
model2. Requires phase7_common.py and the data/ folder in the same directory,
PLUS the adapted tokenizer/model folder (models/mbert_obpe_adapted/) uploaded
alongside it.

Fully resumable — see run_phase7_model1.py header for Kaggle setup notes and
resume behavior; identical here except for the model.

MODEL PATH: looks for models/mbert_obpe_adapted/ first (the OBPE + mean-init
version from Steps 2-3). Falls back to models/mbert_adapted/ with a warning
if the new folder isn't present — double check which one you actually
uploaded before trusting the results, since these are two different models.

OUTPUT (in CHECKPOINT/model3/):
  label_map.json, fold_metrics.csv, fold1..5_predictions.csv
  (same schema as model1/model2 — see phase7_common.py docstring)

Once all three model scripts have finished, zip CHECKPOINT/ (all three
model folders together) and run evaluate_models.py to combine everything.
"""

from pathlib import Path
from phase7_common import run_training_pipeline

ROOT = Path(__file__).parent

if __name__ == "__main__":
    obpe_path = ROOT / "models" / "mbert_obpe_adapted"
    fallback_path = ROOT / "models" / "mbert_adapted"

    if obpe_path.exists():
        model_path = str(obpe_path)
    elif fallback_path.exists():
        model_path = str(fallback_path)
        print(f"WARNING: models/mbert_obpe_adapted/ not found — "
              f"falling back to {fallback_path}. Confirm this is the "
              f"model you intend to evaluate (OBPE + mean-init vs the "
              f"older frequency-only + random-init version) before "
              f"trusting the results.")
    else:
        raise FileNotFoundError(
            "Neither models/mbert_obpe_adapted/ nor models/mbert_adapted/ "
            "was found. Upload the adapted model folder before running this script."
        )

    run_training_pipeline(
        model_name="mBERT_adapted",
        model_path=model_path,
        checkpoint_dirname="model3",
        staged_freeze=True,   # freeze BERT body for epoch 1, unfreeze for epochs 2-5
    )
