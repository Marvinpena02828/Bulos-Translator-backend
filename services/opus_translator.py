"""
OPUS-MT CTranslate2 Translation Service
========================================
Serves Helsinki-NLP opus-mt-en-tl and opus-mt-tl-en models that have been
converted to CTranslate2 int8 format via scripts/convert_opus_to_ct2.py.

Used as Step 4 of the translation pipeline for en ↔ tl when Google Translate
is unavailable (e.g. Render free tier).  Has no Bulos language support.

Model directory layout (created by convert script):
  models/opus/en_tl/
    model.bin          ← CTranslate2 weights (int8, ~75 MB)
    source.spm         ← SentencePiece source tokenizer
    target.spm         ← SentencePiece target tokenizer
    vocab.json         ← vocabulary
    tokenizer_config.json
  models/opus/tl_en/
    (same structure)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)

MODELS_DIR    = Path(__file__).parent.parent / "models" / "opus"
DIRECTION_KEYS = ["en_tl", "tl_en"]


class OpusTranslator:
    """
    Lazy-loads CTranslate2 Marian models for en↔tl.
    Thread-safe singleton via a lock on first use.
    """

    def __init__(self):
        self._models:    dict = {}   # key → {"translator": CT2Translator, "tokenizer": MarianTokenizer}
        self._available: set  = set()
        self._lock = threading.Lock()
        self.per_key_errors: dict = {}

    def load_models(self) -> None:
        """Load all available direction models from disk."""
        try:
            import ctranslate2
            from transformers import MarianTokenizer
        except ImportError as e:
            logger.warning(f"CTranslate2 / transformers not installed: {e}")
            return

        loaded = 0
        for key in DIRECTION_KEYS:
            model_dir = MODELS_DIR / key
            model_bin = model_dir / "model.bin"

            if not model_bin.exists():
                logger.debug(f"[OPUS] Model not found for {key} — skipping")
                continue

            try:
                logger.info(f"[OPUS] Loading {key} from {model_dir} ...")
                translator = ctranslate2.Translator(
                    str(model_dir),
                    device="cpu",
                    inter_threads=1,
                    intra_threads=2,
                    compute_type="int8",
                )
                tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
                self._models[key] = {
                    "translator": translator,
                    "tokenizer":  tokenizer,
                }
                self._available.add(key)
                loaded += 1
                logger.info(f"[OPUS] Loaded: {key}")
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                self.per_key_errors[key] = err
                logger.warning(f"[OPUS] Failed to load {key}: {e}\n{err}")

        logger.info(
            f"[OPUS] Ready — {loaded}/{len(DIRECTION_KEYS)} models loaded. "
            f"Available: {sorted(self._available)}"
        )

    def is_available(self, src_lang: str, tgt_lang: str) -> bool:
        return f"{src_lang}_{tgt_lang}" in self._available

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """
        Translate text using the CTranslate2 Marian model.
        Returns the translated string, or None on failure.
        """
        key = f"{src_lang}_{tgt_lang}"
        if key not in self._available:
            return None

        try:
            entry      = self._models[key]
            tokenizer  = entry["tokenizer"]
            translator = entry["translator"]

            # Tokenize: MarianTokenizer returns input_ids; CT2 needs token strings
            tokens = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(text, add_special_tokens=True)
            )

            results = translator.translate_batch(
                [tokens],
                beam_size=4,
                max_decoding_length=512,
            )

            # Decode the output token ids back to a string
            output_tokens = results[0].hypotheses[0]
            translated = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(output_tokens),
                skip_special_tokens=True,
            )
            return translated.strip() if translated.strip() else None

        except Exception as e:
            logger.error(f"[OPUS] Translation failed ({key}): {e}", exc_info=True)
            return None

    @property
    def available_directions(self) -> list:
        return sorted(self._available)


# ── module-level singleton ────────────────────────────────────────────────────
_instance: Optional[OpusTranslator] = None


def get_opus_translator() -> OpusTranslator:
    """Return the module-level OpusTranslator singleton (lazy-initialised)."""
    global _instance
    if _instance is None:
        _instance = OpusTranslator()
        _instance.load_models()
    return _instance
