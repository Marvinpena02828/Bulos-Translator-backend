"""
LSTM Seq2Seq Translation Service
=================================
Loads pre-trained character-level encoder-decoder LSTM models from
models/lstm/<src>_to_<tgt>/ and provides inference for all 6 direction pairs.

Models must be trained first via:
    python scripts/train_lstm.py
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models" / "lstm"

START_TOKEN = "\t"
END_TOKEN   = "\n"

# All supported direction keys
DIRECTION_KEYS = ["bul_en", "en_bul", "bul_tl", "tl_bul", "en_tl", "tl_en"]


class LSTMTranslator:
    """
    Loads and serves all 6 LSTM direction models.
    Falls back gracefully if a model file is missing.
    """

    def __init__(self):
        self._models: dict = {}   # key → {"enc": model, "dec": model, "tok": dict}
        self._available: set = set()
        self.last_load_error: str = ""        # top-level import error
        self.per_key_errors: dict = {}        # key → error string for each failed model

    def load_models(self) -> None:
        """Load all available direction models from disk."""
        try:
            import tensorflow as tf  # noqa: F401 — ensure TF is importable
            from tensorflow import keras
        except ImportError as e:
            msg = f"TensorFlow not installed — LSTM translator unavailable: {e}"
            logger.warning(msg)
            self.last_load_error = msg
            return

        loaded = 0
        for key in DIRECTION_KEYS:
            model_dir = MODELS_DIR / key
            model_path = model_dir / "model.keras"
            tok_path   = model_dir / "tokenizer.json"

            if not model_path.exists() or not tok_path.exists():
                logger.debug(f"LSTM model not found for {key}, skipping")
                continue

            try:
                training_model = keras.models.load_model(str(model_path))
                with open(tok_path, "r", encoding="utf-8") as f:
                    tok = json.load(f)

                # Rebuild inference sub-models
                latent_dim = training_model.get_layer("encoder").units
                enc_model, dec_model = _build_inference_models(
                    training_model, latent_dim
                )

                self._models[key] = {
                    "enc":         enc_model,
                    "dec":         dec_model,
                    "tok":         tok,
                    "latent_dim":  latent_dim,
                }
                self._available.add(key)
                loaded += 1
                logger.info(f"LSTM model loaded: {key}")

            except Exception as e:
                import traceback
                err_str = traceback.format_exc()
                self.per_key_errors[key] = err_str
                logger.warning(f"Failed to load LSTM model for {key}: {e}\n{err_str}")

        logger.info(f"LSTM translator ready — {loaded}/{len(DIRECTION_KEYS)} models loaded")
        if loaded == 0:
            logger.error(
                "LSTM: 0 models loaded. Check that models/lstm/<dir>/model.keras "
                "and tokenizer.json exist and that TensorFlow can read them."
            )

    def is_available(self, src_lang: str, tgt_lang: str) -> bool:
        key = f"{src_lang}_{tgt_lang}"
        return key in self._available

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """
        Translate text using the LSTM model for the given direction.

        Returns:
            Translated string, or None if the model is unavailable or fails.
        """
        key = f"{src_lang}_{tgt_lang}"
        if key not in self._available:
            return None

        try:
            entry      = self._models[key]
            tok        = entry["tok"]
            enc_model  = entry["enc"]
            dec_model  = entry["dec"]

            src_c2i    = tok["src_char2idx"]
            tgt_c2i    = tok["tgt_char2idx"]
            tgt_i2c    = {int(k): v for k, v in tok["tgt_idx2char"].items()}
            max_src    = tok["max_src_len"]
            max_tgt    = tok["max_tgt_len"]

            enc_seq    = _encode_sequences([text], src_c2i, max_src)
            result     = _decode_sequence(
                enc_seq, enc_model, dec_model, tgt_c2i, tgt_i2c, max_tgt
            )
            return result if result else None

        except Exception as e:
            logger.error(f"LSTM translation failed ({key}): {e}", exc_info=True)
            return None

    @property
    def available_directions(self) -> list:
        return sorted(self._available)


# ── module-level singleton ────────────────────────────────────────────────────
_instance: Optional[LSTMTranslator] = None


def get_lstm_translator() -> LSTMTranslator:
    """Return the module-level LSTMTranslator singleton (lazy-initialised)."""
    global _instance
    if _instance is None:
        _instance = LSTMTranslator()
        _instance.load_models()
    return _instance


# ── helpers (mirror of training script, kept local to avoid circular imports) ─

def _encode_sequences(texts: list, char2idx: dict, max_len: int) -> np.ndarray:
    seqs = []
    for t in texts:
        seq = [char2idx.get(c, 0) for c in t][:max_len]
        seq += [0] * (max_len - len(seq))
        seqs.append(seq)
    return np.array(seqs, dtype=np.int32)


def _decode_sequence(input_seq: np.ndarray, enc_model, dec_model,
                     tgt_c2i: dict, tgt_i2c: dict,
                     max_decode_len: int) -> str:
    states     = enc_model.predict(input_seq, verbose=0)
    target_seq = np.array([[tgt_c2i.get(START_TOKEN, 1)]])
    result     = []

    for _ in range(max_decode_len):
        output, h, c = dec_model.predict([target_seq] + states, verbose=0)
        token_idx    = int(np.argmax(output[0, -1, :]))
        if token_idx == 0:
            break
        char = tgt_i2c.get(token_idx, "")
        if char == END_TOKEN:
            break
        result.append(char)
        target_seq = np.array([[token_idx]])
        states     = [h, c]

    return "".join(result)


def _build_inference_models(training_model, latent_dim: int):
    """Rebuild encoder and decoder inference sub-models from a loaded training model."""
    from tensorflow import keras
    from tensorflow.keras import layers

    # Encoder
    enc_input = training_model.get_layer("enc_input").input
    enc_emb   = training_model.get_layer("enc_emb")(enc_input)
    _, h, c   = training_model.get_layer("encoder")(enc_emb)
    enc_model = keras.Model(enc_input, [h, c])

    # Decoder
    dec_input = training_model.get_layer("dec_input").input
    dec_emb   = training_model.get_layer("dec_emb")(dec_input)
    dec_h_in  = layers.Input(shape=(latent_dim,), name="dec_h_in")
    dec_c_in  = layers.Input(shape=(latent_dim,), name="dec_c_in")
    dec_lstm  = training_model.get_layer("decoder")
    dec_out, dec_h, dec_c = dec_lstm(dec_emb,
                                     initial_state=[dec_h_in, dec_c_in])
    dec_dense  = training_model.get_layer("dec_dense")
    dec_logits = dec_dense(dec_out)
    dec_model  = keras.Model(
        [dec_input, dec_h_in, dec_c_in],
        [dec_logits, dec_h, dec_c]
    )
    return enc_model, dec_model
