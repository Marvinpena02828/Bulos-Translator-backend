"""
Convert Helsinki-NLP OPUS-MT HuggingFace MarianMT models to CTranslate2 int8.

Uses ct2-transformers-converter which handles HuggingFace checkpoints directly.

Downloads:
  Helsinki-NLP/opus-mt-en-tl
  Helsinki-NLP/opus-mt-tl-en

Saves converted models + tokenizer files to:
  models/opus/en_tl/
  models/opus/tl_en/

Usage:
    python scripts/convert_opus_to_ct2.py
"""
import os, sys, shutil, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR   = PROJECT_ROOT / "models" / "opus"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CT2_CONV     = str(PROJECT_ROOT / ".venv" / "Scripts" / "ct2-transformers-converter.exe")

PAIRS = [
    ("Helsinki-NLP/opus-mt-en-tl", "en_tl"),
    ("Helsinki-NLP/opus-mt-tl-en", "tl_en"),
]

from transformers import MarianTokenizer

for hf_id, key in PAIRS:
    out_dir = MODELS_DIR / key

    if out_dir.exists() and (out_dir / "model.bin").exists():
        print(f"[{key}] Already converted — skipping")
        continue

    print(f"\n[{key}] Converting {hf_id} → {out_dir} (int8) ...")

    # ct2-transformers-converter downloads and converts in one step
    result = subprocess.run(
        [
            CT2_CONV,
            "--model",        hf_id,
            "--output_dir",   str(out_dir),
            "--quantization", "int8",
            "--force",
        ],
        capture_output=False,   # print to console in real time
    )

    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed for {key} (exit {result.returncode})")

    print(f"[{key}] CT2 model written.")

    # Also download and copy tokenizer vocab files so inference can find them
    print(f"[{key}] Saving tokenizer files ...")
    tokenizer = MarianTokenizer.from_pretrained(hf_id)
    tokenizer.save_pretrained(str(out_dir))

    print(f"[{key}] Done → {out_dir}")

print("\nAll conversions complete.")
print("Contents of models/opus/:")
for p in sorted(MODELS_DIR.rglob("*")):
    if p.is_file():
        size_kb = p.stat().st_size // 1024
        print(f"  {p.relative_to(PROJECT_ROOT)}  ({size_kb} KB)")
