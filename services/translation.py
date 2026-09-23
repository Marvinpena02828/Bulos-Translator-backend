"""Translation service with comprehensive dictionary-based translation"""
import asyncio
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from deep_translator import GoogleTranslator
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# LSTM translator — optional, loaded lazily on first use
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


class TranslationService:
    """Service for translating text using comprehensive dictionary lookup"""

    SUPPORTED_PAIRS = [
        ("bul", "en"),
        ("en", "bul"),
        ("bul", "tl"),
        ("tl", "bul"),
        ("en", "tl"),
        ("tl", "en"),
    ]

    def __init__(self):
        """Initialize translation service."""
        self.phrase_index: Dict[str, List[Tuple[str, str, int]]] = {}
        self.google_translator = None
        self.timeout = settings.translation_timeout_seconds

        logger.info(f"TranslationService initialized with timeout: {self.timeout}s")
        logger.info("Google Translate enabled for EN↔TL and Tagalog fallback")

    # ──────────────────────────────────────────────────────────────────────────
    # Startup
    # ──────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load all translation data from JSON files and build phrase index."""
        logger.info("Initializing translation service...")

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

            logger.info("Translation service initialized successfully")
            logger.info(
                f"Total indexed phrases: {sum(len(v) for v in self.phrase_index.values())}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize translation service: {str(e)}", exc_info=True
            )
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    async def _load_dictionary_data(self) -> List[Dict[str, str]]:
        dictionary_path = Path(__file__).parent.parent / "dictionary.json"
        if not dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")

        with open(dictionary_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)

        entries = []
        for category in dictionary.get("categories", []):
            for entry in category.get("entries", []):
                bulos = entry.get("BULOS", "").strip()
                filipino = entry.get("FILIPINO", "").strip()
                english = entry.get("ENGLISH", "").strip()
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
        seen = set()
        for letter_data in alphabet.get("letters", []):
            examples = letter_data.get("examples", {})
            for position in ["initial", "medial", "final"]:
                for entry in examples.get(position, []):
                    bulos = (entry.get("BULOS") or "").strip()
                    filipino = (entry.get("FILIPINO") or "").strip()
                    english = (entry.get("ENGLISH") or "").strip()
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
            bulos = (entry.get("BULOS") or "").strip()
            filipino = (entry.get("FILIPINO") or "").strip()
            english = (entry.get("ENGLISH") or "").strip()
            if bulos and filipino and english:
                entries.append({"bul": bulos, "tl": filipino, "en": english})

        logger.info(f"Extracted {len(entries)} sentence entries")
        return entries

    # ──────────────────────────────────────────────────────────────────────────
    # Index building
    # ──────────────────────────────────────────────────────────────────────────

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
        sentence_data: List[Dict[str, str]],
    ) -> None:
        """Build phrase index keyed by normalized source text."""
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
            bn = self._normalize_text(bul)
            tn = self._normalize_text(tl)
            en_n = self._normalize_text(en)
            bw, tw, ew = len(bn.split()), len(tn.split()), len(en_n.split())
            _add(tn, bul, "tl_to_bul", tw)
            _add(bn, tl, "bul_to_tl", bw)
            _add(en_n, bul, "en_to_bul", ew)
            _add(bn, en, "bul_to_en", bw)
            _add(en_n, tl, "en_to_tl", ew)
            _add(tn, en, "tl_to_en", tw)

        # Dictionary words (don't overwrite sentences)
        for entry in dictionary_data:
            bul, tl, en = entry["bul"], entry["tl"], entry["en"]
            bn = self._normalize_text(bul)
            tn = self._normalize_text(tl)
            en_n = self._normalize_text(en)
            bw, tw, ew = len(bn.split()), len(tn.split()), len(en_n.split())
            _add_if_missing(tn, bul, "tl_to_bul", tw)
            _add_if_missing(bn, tl, "bul_to_tl", bw)
            _add_if_missing(en_n, bul, "en_to_bul", ew)
            _add_if_missing(bn, en, "bul_to_en", bw)
            _add_if_missing(en_n, tl, "en_to_tl", ew)
            _add_if_missing(tn, en, "tl_to_en", tw)

        logger.info(f"Built phrase index with {len(self.phrase_index)} unique phrases")

    # ──────────────────────────────────────────────────────────────────────────
    # Google Translate helper
    # ──────────────────────────────────────────────────────────────────────────

    async def _translate_with_google(
        self, text: str, source_lang: str, target_lang: str
    ) -> Optional[str]:
        try:
            lang_map = {"en": "en", "tl": "tl", "bul": "tl"}
            src = lang_map.get(source_lang, source_lang)
            dest = lang_map.get(target_lang, target_lang)
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                None,
                lambda: GoogleTranslator(source=src, target=dest).translate(text),
            )
            if translated:
                logger.debug(
                    f"Google Translate: '{text}' ({source_lang}→{target_lang}) = '{translated}'"
                )
                return translated
            logger.warning(f"Google Translate returned empty result for: '{text}'")
            return None
        except Exception as e:
            logger.warning(
                f"Google Translate failed for '{text}' ({source_lang}→{target_lang}): {e}"
            )
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Greedy phrase-matching helper
    # ──────────────────────────────────────────────────────────────────────────

    def _greedy_translate(
        self, text: str, source_language: str, target_language: str
    ) -> Dict[str, Any]:
        """
        Greedy longest-match translation using the phrase index.
        Returns translated_text, confidence, and translation_method.
        """
        lang_pair = f"{source_language}_to_{target_language}"
        text_norm = self._normalize_text(text)

        # Try exact full-text match first
        if text_norm in self.phrase_index:
            for translation, pair, _ in self.phrase_index[text_norm]:
                if pair == lang_pair:
                    return {
                        "translated_text": translation,
                        "confidence": 1.0,
                        "translation_method": "dictionary",
                    }

        words = text.split()
        words_norm = [w.strip(".!?,;:") for w in text_norm.split()]

        if not words_norm:
            return {
                "translated_text": text,
                "confidence": 0.0,
                "translation_method": "tagalog_only",
            }

        translated_parts = []
        matched_word_count = 0
        i = 0

        while i < len(words_norm):
            match_found = False
            best_translation = None
            best_length = 0

            for length in range(len(words_norm) - i, 0, -1):
                phrase = " ".join(words_norm[i : i + length])
                if phrase in self.phrase_index:
                    for translation, pair, _ in self.phrase_index[phrase]:
                        if pair == lang_pair:
                            best_translation = translation
                            best_length = length
                            match_found = True
                            break
                    if match_found:
                        break

            if match_found and best_translation is not None:
                translated_parts.append(best_translation)
                matched_word_count += best_length
                logger.debug(
                    f"Matched at {i}: '{' '.join(words[i:i+best_length])}' → '{best_translation}'"
                )
                i += best_length
            else:
                translated_parts.append(words[i])
                logger.debug(f"No match at {i}: '{words[i]}' (preserved)")
                i += 1

        total = len(words_norm)
        confidence = matched_word_count / total if total > 0 else 0.0
        translated_text = " ".join(translated_parts)

        if confidence == 1.0:
            method = "dictionary"
        elif confidence > 0:
            method = "dictionary_with_tagalog_preservation"
        else:
            method = "tagalog_only"

        logger.info(
            f"Greedy result: '{text}' → '{translated_text}' "
            f"(confidence={confidence:.2f}, matched={matched_word_count}/{total}, method={method})"
        )
        return {
            "translated_text": translated_text,
            "confidence": confidence,
            "translation_method": method,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Core translation routing
    # ──────────────────────────────────────────────────────────────────────────

    async def _translate_text(
        self, text: str, source_language: str, target_language: str
    ) -> Dict[str, Any]:
        """
        Route translation through the appropriate strategy:
        1. EN ↔ TL  — Google Translate directly
        2. EN → BUL — dictionary first, then 2-step (EN→TL→BUL) fallback
        3. BUL → EN — 2-step (BUL→TL→EN)
        4. TL ↔ BUL — greedy dictionary matching
        """
        text = text.strip()
        if not text:
            return {"translated_text": "", "confidence": 0.0}

        # ROUTE 1: EN ↔ TL
        if source_language in ("en", "tl") and target_language in ("en", "tl"):
            translated = await self._translate_with_google(
                text, source_language, target_language
            )
            if translated:
                return {
                    "translated_text": translated,
                    "confidence": 1.0,
                    "translation_method": "google_translate",
                }
            return {
                "translated_text": text,
                "confidence": 0.0,
                "translation_method": "fallback_original",
            }

        # ROUTE 2: EN → BUL
        if source_language == "en" and target_language == "bul":
            # Try direct EN→BUL dictionary match first
            direct = self._greedy_translate(text, "en", "bul")
            if direct["confidence"] > 0:
                return direct

            # Fallback: EN→TL via Google, then TL→BUL via dictionary
            tagalog_text = await self._translate_with_google(text, "en", "tl")
            if not tagalog_text:
                return {
                    "translated_text": text,
                    "confidence": 0.0,
                    "translation_method": "fallback_original",
                }
            logger.info(f"EN→BUL step 1 (EN→TL): '{text}' → '{tagalog_text}'")
            result = self._greedy_translate(tagalog_text, "tl", "bul")
            result["translation_method"] = "google_then_dictionary"
            result["intermediate_language"] = "tl"
            logger.info(
                f"EN→BUL step 2 (TL→BUL): '{tagalog_text}' → '{result['translated_text']}'"
            )
            return result

        # ROUTE 3: BUL → EN
        if source_language == "bul" and target_language == "en":
            result = self._greedy_translate(text, "bul", "tl")
            tagalog_text = result["translated_text"]
            bul_tl_confidence = result["confidence"]
            logger.info(
                f"BUL→EN step 1 (BUL→TL): '{text}' → '{tagalog_text}' "
                f"(confidence={bul_tl_confidence:.2f})"
            )

            if bul_tl_confidence < 1.0:
                # Partial match — don't pass mixed text to Google
                logger.warning(
                    f"BUL→TL partial match ({bul_tl_confidence:.2f}). "
                    "Skipping Google to avoid misinterpreting unmatched Bulos words."
                )
                return {
                    "translated_text": tagalog_text,
                    "confidence": bul_tl_confidence,
                    "translation_method": "dictionary_then_google",
                    "intermediate_language": "tl",
                }

            english_text = await self._translate_with_google(tagalog_text, "tl", "en")
            if not english_text:
                return {
                    "translated_text": tagalog_text,
                    "confidence": bul_tl_confidence,
                    "translation_method": "dictionary_then_google",
                    "intermediate_language": "tl",
                }
            logger.info(f"BUL→EN step 2 (TL→EN): '{tagalog_text}' → '{english_text}'")
            return {
                "translated_text": english_text,
                "confidence": bul_tl_confidence,
                "translation_method": "dictionary_then_google",
                "intermediate_language": None,
            }

        # ROUTE 4: TL ↔ BUL
        return self._greedy_translate(text, source_language, target_language)

    # ──────────────────────────────────────────────────────────────────────────
    # Public translate()
    # ──────────────────────────────────────────────────────────────────────────

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translate text from source to target language.
        LSTM is used as a fallback when dictionary/Google returns confidence=0.
        """
        logger.info(f"Translation request: '{text}' ({source_language}→{target_language})")

        if source_language not in ("en", "tl", "bul"):
            raise ValueError(
                f"Invalid source_language: {source_language}. Must be one of: en, tl, bul"
            )
        if (source_language, target_language) not in self.SUPPORTED_PAIRS:
            raise ValueError(
                f"Unsupported language pair: {source_language} → {target_language}. "
                f"Supported: {self.SUPPORTED_PAIRS}"
            )
        if source_language == target_language:
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 1.0,
            }

        try:
            result = await asyncio.wait_for(
                self._translate_text(text, source_language, target_language),
                timeout=self.timeout,
            )

            # LSTM fallback — only when dictionary/Google found nothing
            if self._should_attempt_lstm_fallback(result, source_language, target_language):
                logger.info("Dictionary/Google returned nothing — trying LSTM fallback...")
                lstm_result = await self._try_lstm_translation(
                    text, source_language, target_language
                )
                if lstm_result:
                    logger.info(f"LSTM fallback succeeded: '{lstm_result['translated_text']}'")
                    result = lstm_result
                else:
                    logger.info("LSTM fallback unavailable — keeping original result")

            return {
                "original_text": text,
                "translated_text": result["translated_text"],
                "source_language": source_language,
                "target_language": target_language,
                "confidence": result["confidence"],
                "translation_method": result.get("translation_method", "unknown"),
                "intermediate_language": result.get("intermediate_language"),
            }

        except asyncio.TimeoutError:
            logger.error(f"Translation timeout ({self.timeout}s) for: '{text}'")
            raise asyncio.TimeoutError(
                f"Translation exceeded timeout of {self.timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # LSTM fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _should_attempt_lstm_fallback(
        self,
        hybrid_result: Dict[str, Any],
        source_language: str,
        target_language: str,
    ) -> bool:
        """Return True only when dictionary/Google produced nothing useful."""
        lstm = _get_lstm()
        if lstm is None:
            return False
        if not lstm.is_available(source_language, target_language):
            return False
        confidence = hybrid_result.get("confidence", 0.0)
        method = hybrid_result.get("translation_method", "unknown")
        if confidence == 0.0 and method in ("tagalog_only", "fallback_original"):
            logger.debug(
                f"LSTM fallback triggered: confidence={confidence:.2f}, method={method}"
            )
            return True
        return False

    async def _try_lstm_translation(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> Optional[Dict[str, Any]]:
        """Attempt LSTM translation. Never raises — returns None on any failure."""
        lstm = _get_lstm()
        if lstm is None:
            return None
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: lstm.translate(text, source_language, target_language),
            )
            if result:
                logger.info(
                    f"LSTM: '{text}' → '{result}' ({source_language}→{target_language})"
                )
                return {
                    "translated_text": result,
                    "confidence": 0.6,
                    "translation_method": "lstm",
                    "intermediate_language": None,
                }
            return None
        except Exception as e:
            logger.warning(
                f"LSTM translation failed ({source_language}→{target_language}): {e}"
            )
            return None
