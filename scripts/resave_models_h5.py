"""
Re-save LSTM models from .keras format to .h5 (legacy HDF5) format.

The .keras format saved by Keras 3 cannot be loaded by tensorflow-cpu==2.15.0
on Render (which bundles Keras 2).  The .h5 format is stable across TF 2.x.

Run this once locally (where the .keras files load fine), then commit the
new model.h5 files.

Usage:
    python scripts/resave_models_h5.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

MODELS_DIR   = PROJECT_ROOT / "models" / "lstm"
DIRECTION_KEYS = ["bul_en", "en_bul", "bul_tl", "tl_bul", "en_tl", "tl_en"]

print(f"TensorFlow version: {tf.__version__}")
print(f"Models directory:   {MODELS_DIR}\n")

converted = 0
for key in DIRECTION_KEYS:
    keras_path = MODELS_DIR / key / "model.keras"
    h5_path    = MODELS_DIR / key / "model.h5"

    if not keras_path.exists():
        print(f"[{key}] SKIP — model.keras not found")
        continue

    try:
        print(f"[{key}] Loading {keras_path} ...", end=" ", flush=True)
        model = keras.models.load_model(str(keras_path))
        print("OK")

        print(f"[{key}] Saving  {h5_path}   ...", end=" ", flush=True)
        model.save(str(h5_path), save_format="h5")
        print("OK")
        converted += 1

    except Exception as e:
        print(f"FAILED: {e}")

print(f"\nDone — {converted}/{len(DIRECTION_KEYS)} models converted to .h5")
