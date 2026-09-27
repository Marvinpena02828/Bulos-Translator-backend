"""
Translation service — internal pipeline only.

All translation is performed exclusively through internal tools in this order:

  Step 1 — Phrase matching   : exact full-text lookup in phrase_index
  Step 2 — Word matching     : greedy longest-match token-by-token in phrase_index
  Step 3 — Fuzzy matching    : Levenshtein similarity for unmatched tokens
  Step 4 — LSTM model        : seq2seq neural translation on the original input

After each of steps 1-3, if all tokens are matched (confidence == 1.0) the
result is returned immediately.  Step 4 always executes if reached and its
output is always returned.

No external APIs (Google Translate, MyMemory, etc.) are used anywhere.
"""

import asyncio
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from config import settings
from utils.logging_config import get_logger
from utils.fuzzy_match import levenshtein_similarity

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# LSTM singleton — loaded lazily on first use
# ---------------------------------------------------------------------------

_lstm_translator = None


def _get_lstm():
    """Return the singleton LSTMTranslator, loading it once on first call."""
    global _lstm_translator
    if _lstm_translator is None:
        try:
            from services.lstm_translator import get_lstm_translator
            _lstm_translator = get_lstm_translator()
        except Exception as e:
            logger.warning(f"LSTM translator unavailable: {e}")
            _lstm_translator = False  # sentinel — don't retry
    return _lstm_translator if _lstm_translator is not False else None


# ---------------------------------------------------------------------------
# Fuzzy matching threshold
# ---------------------------------------------------------------------------

FUZZY_THRESHOLD = 0.75   # minimum Levenshtein similarity to accept a match
FUZZY_MIN_LEN   = 3      # don't fuzzy-match tokens shorter than this


class TranslationService:
    """
    Translates text using a four-step internal pipeline.
    No external API calls are made at any point.
    """

    SUPPORTED_PAIRS = [
        ("bul", "en"),
        ("en", "bul"),
        ("bul", "tl"),
        ("tl", "bul"),
        ("en", "tl"),
        ("tl", "en"),
    ]

    def __init__(self):
        # phrase_index: normalized_source_text → [(translation, lang_pair_key, word_count), ...]
        self.phrase_index: Dict[str, List[Tuple[str, str, int]]] = {}
        self.timeout = settings.translation_timeout_seconds
        logger.info(f"TranslationService initialised (timeout={self.timeout}s, "
                    f"pipeline=phrase→word→fuzzy→LSTM)")

    # -----------------------------------------------------------------------
    # Startup
    # -----------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load all translation data from JSON files and build phrase index."""
        logger.info("Initialising translation service...")
        try:
            dictionary_data = await self._load_dictionary_data()
            logger.info(f"Loaded {len(dictionary_data)} word entries from dictionary")

            alphabet_data = await self._load_alphabet_data()
            logger.info(f"Loaded {len(alphabet_data)} word entries from alphabet")

            sentence_data = await self._load_sentence_data()
            logger.info(f"Loaded {len(sentence_data)} sentence entries")

            all_vocabulary = dictionary_data + alphabet_data
            logger.info(f"Total vocabulary entries: {len(all_vocabulary)}")

            self._build_translation_indexes(all_vocabulary, sentence_data)

            logger.info("Translation service initialised successfully")
            logger.info(
                f"Phrase index size: {len(self.phrase_index)} unique keys, "
                f"{sum(len(v) for v in self.phrase_index.values())} total entries"
            )
        except Exception as e:
            logger.error(f"Failed to initialise translation service: {e}", exc_info=True)
            raise

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    async def _load_dictionary_data(self) -> List[Dict[str, str]]:
        dictionary_path = Path(__file__).parent.parent / "dictionary.json"
        if not dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")

        with open(dictionary_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)

        entries = []
        for category in dictionary.get("categories", []):
            for entry in category.get("entries", []):
                bulos   = entry.get("BULOS",    "").strip()
                filipino = entry.get("FILIPINO", "").strip()
                english = entry.get("ENGLISH",  "").strip()
                if bulos and filipino and english and bulos != "—" and filipino != "—":
                    entries.append({"bul": bulos, "tl": filipino, "en": english})

        logger.info(f"Extracted {len(entries)} valid dictionary entries")
        return entries

    async def _load_alphabet_data(self) -> List[Dict[str, str]]:
        alphabet_path = Path(__file__).parent.parent / "alphabet.json"
        if not alphabet_path.exists():
            logger.warning(f"Alphabet file not found at {alphabet_path}")
            return []

        with open(alphabet_path, "r", encoding="utf-8") as f:
            alphabet = json.load(f)

        entries = []
        seen: set = set()
        for letter_data in alphabet.get("letters", []):
            examples = letter_data.get("examples", {})
            for position in ["initial", "medial", "final"]:
                for entry in examples.get(position, []):
                    bulos    = (entry.get("BULOS")    or "").strip()
                    filipino = (entry.get("FILIPINO") or "").strip()
                    english  = (entry.get("ENGLISH")  or "").strip()
                    key = f"{bulos.lower()}|{filipino.lower()}|{english.lower()}"
                    if key in seen or not (bulos and filipino and english):
                        continue
                    if bulos == "—" or filipino == "—":
                        continue
                    entries.append({"bul": bulos, "tl": filipino, "en": english})
                    seen.add(key)

        logger.info(f"Extracted {len(entries)} unique vocabulary entries from alphabet")
        return entries

    async def _load_sentence_data(self) -> List[Dict[str, str]]:
        sentence_path = Path(__file__).parent.parent / "sentence.json"
        if not sentence_path.exists():
            logger.warning(f"Sentence file not found at {sentence_path}")
            return []

        with open(sentence_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = []
        for entry in data.get("entries", []):
            bulos    = (entry.get("BULOS")    or "").strip()
            filipino = (entry.get("FILIPINO") or "").strip()
            english  = (entry.get("ENGLISH")  or "").strip()
            if bulos and filipino and english:
                entries.append({"bul": bulos, "tl": filipino, "en": english})

        logger.info(f"Extracted {len(entries)} sentence entries")
        return entries

    # -----------------------------------------------------------------------
    # Index building
    # -----------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """Accent-strip + lowercase + whitespace collapse + trailing punctuation removal."""
        normalized = unicodedata.normalize("NFD", text)
        normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        normalized = normalized.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.rstrip(".!?,;:")
        return normalized

    def _build_translation_indexes(
        self,
        dictionary_data: List[Dict[str, str]],
        sentence_data:   List[Dict[str, str]],
    ) -> None:
        """Build phrase_index keyed by normalised source text."""
        self.phrase_index = {}

        def _add(norm_key: str, translation: str, pair: str, wc: int) -> None:
            if norm_key not in self.phrase_index:
                self.phrase_index[norm_key] = []
            self.phrase_index[norm_key].append((translation, pair, wc))

        def _add_if_missing(norm_key: str, translation: str, pair: str, wc: int) -> None:
            if norm_key not in self.phrase_index:
                self.phrase_index[norm_key] = []
            if not any(p == pair for _, p, _ in self.phrase_index[norm_key]):
                self.phrase_index[norm_key].append((translation, pair, wc))

        # Sentences first (longer phrases take priority)
        for entry in sentence_data:
            bul, tl, en = entry["bul"], entry["tl"], entry["en"]
            bn   = self._normalize_text(bul)
            tn   = self._normalize_text(tl)
            en_n = self._normalize_text(en)
            bw, tw, ew = len(bn.split()), len(tn.split()), len(en_n.split())
            _add(tn,   bul, "tl_to_bul", tw)
            _add(bn,   tl,  "bul_to_tl", bw)
            _add(en_n, bul, "en_to_bul", ew)
            _add(bn,   en,  "bul_to_en", bw)
            _add(en_n, tl,  "en_to_tl",  ew)
            _add(tn,   en,  "tl_to_en",  tw)

        # Dictionary words (don't overwrite sentences)
        for entry in dictionary_data:
            bul, tl, en = entry["bul"], entry["tl"], entry["en"]
            bn   = self._normalize_text(bul)
            tn   = self._normalize_text(tl)
            en_n = self._normalize_text(en)
            bw, tw, ew = len(bn.split()), len(tn.split()), len(en_n.split())
            _add_if_missing(tn,   bul, "tl_to_bul", tw)
            _add_if_missing(bn,   tl,  "bul_to_tl", bw)
            _add_if_missing(en_n, bul, "en_to_bul", ew)
            _add_if_missing(bn,   en,  "bul_to_en", bw)
            _add_if_missing(en_n, tl,  "en_to_tl",  ew)
            _add_if_missing(tn,   en,  "tl_to_en",  tw)

        logger.info(f"Built phrase index with {len(self.phrase_index)} unique phrases")

    # -----------------------------------------------------------------------
    # ═══════════════════════════════════════════════════════════════════════
    #  4-STEP INTERNAL TRANSLATION PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    # -----------------------------------------------------------------------

    def _lookup_phrase(self, norm_phrase: str, lang_pair: str) -> Optional[str]:
        """Return the translation for norm_phrase in lang_pair, or None."""
        if norm_phrase in self.phrase_index:
            for translation, pair, _ in self.phrase_index[norm_phrase]:
                if pair == lang_pair:
                    return translation
        return None

    # ── Step 1: Phrase-based matching ──────────────────────────────────────

    def _step1_phrase(
        self, text: str, lang_pair: str
    ) -> Optional[Dict[str, Any]]:
        """
        Exact full-text match against the phrase index.
        Returns a result dict with confidence=1.0, or None if no match.
        """
        norm = self._normalize_text(text)
        translation = self._lookup_phrase(norm, lang_pair)
        if translation is not None:
            logger.debug(f"[Step 1] Phrase match: '{text}' → '{translation}'")
            return {
                "translated_text":   translation,
                "confidence":        1.0,
                "translation_method": "phrase_match",
            }
        return None

    # ── Step 2: Word-based matching ────────────────────────────────────────

    def _step2_word(
        self, text: str, lang_pair: str
    ) -> Dict[str, Any]:
        """
        Greedy longest-match token-by-token against the phrase index.
        Returns a result dict.  confidence == 1.0 means all tokens matched.
        The 'unmatched_positions' list carries the indices of tokens that
        were NOT found (needed by step 3).
        """
        words_raw  = text.split()
        words_norm = [self._normalize_text(w) for w in words_raw]

        if not words_norm:
            return {
                "translated_text":    text,
                "confidence":         0.0,
                "translation_method": "word_match",
                "parts":              [],
                "unmatched_positions": [],
            }

        parts:      List[str]  = []
        matched:    int        = 0
        unmatched_positions: List[int] = []
        i = 0

        while i < len(words_norm):
            found_translation = None
            found_length      = 0

            # Try longest phrase first, then shorter ones
            for length in range(len(words_norm) - i, 0, -1):
                phrase = " ".join(words_norm[i : i + length])
                t = self._lookup_phrase(phrase, lang_pair)
                if t is not None:
                    found_translation = t
                    found_length      = length
                    break

            if found_translation is not None:
                parts.append(found_translation)
                matched += found_length
                logger.debug(
                    f"[Step 2] Matched '{' '.join(words_raw[i:i+found_length])}'"
                    f" → '{found_translation}'"
                )
                i += found_length
            else:
                parts.append(words_raw[i])      # preserve original token
                unmatched_positions.append(i)
                logger.debug(f"[Step 2] No match for '{words_raw[i]}' at pos {i}")
                i += 1

        total      = len(words_norm)
        confidence = matched / total if total > 0 else 0.0

        logger.info(
            f"[Step 2] '{text}' → '{' '.join(parts)}' "
            f"(matched {matched}/{total}, confidence={confidence:.2f})"
        )

        method = "word_match" if confidence == 1.0 else (
            "word_match_partial" if confidence > 0 else "no_match"
        )

        return {
            "translated_text":    " ".join(parts),
            "confidence":         confidence,
            "translation_method": method,
            "parts":              parts,               # mutable list for step 3
            "words_raw":          words_raw,           # original tokens
            "words_norm":         words_norm,          # normalised tokens
            "unmatched_positions": unmatched_positions,
        }

    # ── Step 3: Fuzzy matching (Levenshtein) ───────────────────────────────

    def _step3_fuzzy(
        self,
        step2_result: Dict[str, Any],
        lang_pair: str,
    ) -> Dict[str, Any]:
        """
        For each token that Step 2 left unmatched, search every key in
        phrase_index for the closest Levenshtein match.  If the best match
        meets FUZZY_THRESHOLD and the entry covers the target lang_pair,
        replace that token with its translation.

        Mutates and returns an updated result dict.
        """
        unmatched = step2_result.get("unmatched_positions", [])
        if not unmatched:
            # Nothing left to resolve — confidence already 1.0
            step2_result["translation_method"] = step2_result.get("translation_method", "fuzzy")
            return step2_result

        parts      = list(step2_result["parts"])
        words_norm = step2_result["words_norm"]
        words_raw  = step2_result["words_raw"]
        newly_matched = 0

        # Build lookup: all phrase_index keys that have at least one entry
        # for this lang_pair (avoids checking irrelevant entries).
        pair_keys = [k for k, entries in self.phrase_index.items()
                     if any(p == lang_pair for _, p, _ in entries)]

        for pos in unmatched:
            token = words_norm[pos]

            # Skip very short tokens — too many false positives
            if len(token) < FUZZY_MIN_LEN:
                logger.debug(f"[Step 3] Skipping short token '{token}' at pos {pos}")
                continue

            best_key   = None
            best_score = 0.0

            for candidate in pair_keys:
                # Only compare single-word candidates to single source tokens
                if " " in candidate:
                    continue
                score = levenshtein_similarity(token, candidate)
                if score > best_score:
                    best_score = score
                    best_key   = candidate

            if best_key is not None and best_score >= FUZZY_THRESHOLD:
                translation = self._lookup_phrase(best_key, lang_pair)
                if translation is not None:
                    logger.debug(
                        f"[Step 3] Fuzzy '{words_raw[pos]}' ≈ '{best_key}' "
                        f"(score={best_score:.2f}) → '{translation}'"
                    )
                    parts[pos] = translation
                    newly_matched += 1
            else:
                logger.debug(
                    f"[Step 3] No fuzzy match for '{token}' "
                    f"(best={best_score:.2f} < threshold={FUZZY_THRESHOLD})"
                )

        total         = len(words_norm)
        prev_matched  = int(round(step2_result["confidence"] * total))
        total_matched = prev_matched + newly_matched
        confidence    = total_matched / total if total > 0 else 0.0

        method = "fuzzy_match" if confidence == 1.0 else (
            "fuzzy_match_partial" if confidence > 0 else "no_match"
        )

        logger.info(
            f"[Step 3] Fuzzy resolved {newly_matched}/{len(unmatched)} unmatched tokens. "
            f"confidence={confidence:.2f}"
        )

        return {
            "translated_text":    " ".join(parts),
            "confidence":         confidence,
            "translation_method": method,
        }

    # ── Step 4: LSTM ───────────────────────────────────────────────────────

    async def _step4_lstm(
        self, text: str, source_language: str, target_language: str
    ) -> Dict[str, Any]:
        """
        Run the LSTM seq2seq model on the ORIGINAL input text.
        Always returns a result dict; falls back to the original text if the
        model is unavailable or produces nothing.
        """
        lstm = _get_lstm()
        if lstm is None or not lstm.is_available(source_language, target_language):
            reason = "LSTM singleton failed to load" if lstm is None else \
                     f"model for {source_language}_{target_language} not in available set {lstm.available_directions}"
            logger.warning(f"[Step 4] LSTM not available for {source_language}→{target_language}: {reason}. Returning original text.")
            return {
                "translated_text":   text,
                "confidence":        0.0,
                "translation_method": "lstm_unavailable",
            }

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: lstm.translate(text, source_language, target_language),
            )
            if result and result.strip():
                logger.info(
                    f"[Step 4] LSTM: '{text}' → '{result}' "
                    f"({source_language}→{target_language})"
                )
                return {
                    "translated_text":   result,
                    "confidence":        1.0,
                    "translation_method": "lstm",
                }
            # Model returned empty — fall back to original text
            logger.warning(f"[Step 4] LSTM returned empty output for '{text}'")
            return {
                "translated_text":   text,
                "confidence":        0.0,
                "translation_method": "lstm_empty",
            }
        except Exception as e:
            logger.error(f"[Step 4] LSTM failed: {e}", exc_info=True)
            return {
                "translated_text":   text,
                "confidence":        0.0,
                "translation_method": "lstm_error",
            }

    # ── Pipeline orchestrator ──────────────────────────────────────────────

    async def _translate_pipeline(
        self, text: str, source_language: str, target_language: str
    ) -> Dict[str, Any]:
        """
        Run the 4-step internal pipeline.
        Exits early (returns immediately) after any step that achieves
        confidence == 1.0.  If step 4 is reached it is always executed and
        its output is always returned.
        """
        lang_pair = f"{source_language}_to_{target_language}"

        # ── Step 1: Phrase-based matching ──────────────────────────────────
        result = self._step1_phrase(text, lang_pair)
        if result is not None:
            logger.info(f"Pipeline exit after Step 1 (phrase match)")
            return result

        # ── Step 2: Word-based matching ────────────────────────────────────
        step2 = self._step2_word(text, lang_pair)
        if step2["confidence"] == 1.0:
            logger.info("Pipeline exit after Step 2 (word match, all tokens matched)")
            # Clean up internal keys before returning
            return {
                "translated_text":   step2["translated_text"],
                "confidence":        step2["confidence"],
                "translation_method": step2["translation_method"],
            }

        # ── Step 3: Fuzzy matching (Levenshtein) ───────────────────────────
        step3 = self._step3_fuzzy(step2, lang_pair)
        if step3["confidence"] == 1.0:
            logger.info("Pipeline exit after Step 3 (fuzzy match, all tokens resolved)")
            return step3

        # ── Step 4: LSTM — runs on best partial output so far ─────────────
        # Use step 3's partially-translated text so matched tokens are preserved
        # and LSTM only has to handle what's left unresolved.
        best_so_far = step3["translated_text"]
        logger.info(
            f"Pipeline reached Step 4 — running LSTM on partial output: '{best_so_far}'"
        )
        return await self._step4_lstm(best_so_far, source_language, target_language)

    # -----------------------------------------------------------------------
    # Public translate()
    # -----------------------------------------------------------------------

    async def translate(
        self,
        text:            str,
        source_language: str,
        target_language: str,
        user_id:         Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translate text from source to target language using the 4-step
        internal pipeline.  No external API calls are made.
        """
        logger.info(
            f"Translation request: '{text}' "
            f"({source_language}→{target_language})"
        )

        if source_language not in ("en", "tl", "bul"):
            raise ValueError(
                f"Invalid source_language: {source_language}. "
                "Must be one of: en, tl, bul"
            )
        if (source_language, target_language) not in self.SUPPORTED_PAIRS:
            raise ValueError(
                f"Unsupported language pair: {source_language} → {target_language}. "
                f"Supported: {self.SUPPORTED_PAIRS}"
            )
        if source_language == target_language:
            return {
                "original_text":      text,
                "translated_text":    text,
                "source_language":    source_language,
                "target_language":    target_language,
                "confidence":         1.0,
                "translation_method": "passthrough",
            }

        text = text.strip()
        if not text:
            return {
                "original_text":      "",
                "translated_text":    "",
                "source_language":    source_language,
                "target_language":    target_language,
                "confidence":         0.0,
                "translation_method": "empty_input",
            }

        try:
            result = await asyncio.wait_for(
                self._translate_pipeline(text, source_language, target_language),
                timeout=self.timeout,
            )
            return {
                "original_text":      text,
                "translated_text":    result["translated_text"],
                "source_language":    source_language,
                "target_language":    target_language,
                "confidence":         result["confidence"],
                "translation_method": result.get("translation_method", "unknown"),
            }

        except asyncio.TimeoutError:
            logger.error(
                f"Translation timeout ({self.timeout}s) for: '{text}'"
            )
            raise asyncio.TimeoutError(
                f"Translation exceeded timeout of {self.timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            raise
