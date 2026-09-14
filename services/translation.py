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
from utils.fuzzy_match import levenshtein_similarity

logger = get_logger(__name__)


class TranslationService:
    """Service for translating text using comprehensive dictionary lookup"""
    
    # Supported language pairs
    SUPPORTED_PAIRS = [
        ("bul", "en"),  # Bulos <-> English
        ("en", "bul"),
        ("bul", "tl"),  # Bulos <-> Tagalog/Filipino
        ("tl", "bul"),
        ("en", "tl"),   # English <-> Tagalog
        ("tl", "en"),
    ]
    
    def __init__(self):
        """Initialize translation service."""
        self.phrase_index: Dict[str, List[Tuple[str, str, int]]] = {}
        self.google_translator = None
        self.timeout = settings.translation_timeout_seconds

        logger.info(f"TranslationService initialized with timeout: {self.timeout}s")
        logger.info("Google Translate enabled for EN<->TL and Tagalog fallback")
    
    async def initialize(self) -> None:
        """
        Load all translation data from JSON files
        
        Loads:
        - dictionary.json: Word-level translations (organized by categories)
        - sentence.json: Sentence-level translations
        - alphabet.json: Additional vocabulary from alphabet examples
        """
        logger.info("Initializing translation service...")
        
        try:
            # Load dictionary data (words organized by categories)
            dictionary_data = await self._load_dictionary_data()
            logger.info(f"Loaded {len(dictionary_data)} word entries from dictionary")
            
            # Load alphabet data (additional vocabulary from examples)
            alphabet_data = await self._load_alphabet_data()
            logger.info(f"Loaded {len(alphabet_data)} word entries from alphabet")
            
            # Load sentence data
            sentence_data = await self._load_sentence_data()
            logger.info(f"Loaded {len(sentence_data)} sentence entries")
            
            # Combine all vocabulary sources
            all_vocabulary = dictionary_data + alphabet_data
            logger.info(f"Total vocabulary entries: {len(all_vocabulary)}")
            
            # Build comprehensive lookup indexes
            self._build_translation_indexes(all_vocabulary, sentence_data)
            
            logger.info("Translation service initialized successfully")
            logger.info(f"Total indexed phrases: {sum(len(v) for v in self.phrase_index.values())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {str(e)}", exc_info=True)
            raise
    
    async def _load_dictionary_data(self) -> List[Dict[str, str]]:
        """
        Load and parse dictionary.json to extract word-level translations
        
        Returns:
            List of dictionaries with 'bul', 'tl', and 'en' translations
        """
        dictionary_path = Path(__file__).parent.parent / "dictionary.json"
        
        if not dictionary_path.exists():
            logger.error(f"Dictionary file not found at {dictionary_path}")
            raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")
        
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
        
        # Extract entries from all categories
        entries = []
        for category in dictionary.get("categories", []):
            for entry in category.get("entries", []):
                bulos = entry.get("BULOS", "").strip()
                filipino = entry.get("FILIPINO", "").strip()
                english = entry.get("ENGLISH", "").strip()
                
                # Skip entries with missing translations or placeholder values
                if bulos and filipino and english and bulos != "—" and filipino != "—":
                    entries.append({
                        "bul": bulos,
                        "tl": filipino,
                        "en": english
                    })
        
        logger.info(f"Extracted {len(entries)} valid dictionary entries")
        return entries
    
    async def _load_alphabet_data(self) -> List[Dict[str, str]]:
        """
        Load and parse alphabet.json to extract additional vocabulary from examples
        
        Returns:
            List of dictionaries with 'bul', 'tl', and 'en' translations
        """
        alphabet_path = Path(__file__).parent.parent / "alphabet.json"
        
        if not alphabet_path.exists():
            logger.warning(f"Alphabet file not found at {alphabet_path}")
            return []
        
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            alphabet = json.load(f)
        
        # Extract entries from all letters and positions (initial, medial, final)
        entries = []
        seen = set()  # Track unique entries to avoid duplicates
        
        for letter_data in alphabet.get("letters", []):
            examples = letter_data.get("examples", {})
            
            # Process all position types: initial, medial, final
            for position in ["initial", "medial", "final"]:
                position_examples = examples.get(position, [])
                
                for entry in position_examples:
                    bulos = entry.get("BULOS") if entry.get("BULOS") is not None else ""
                    filipino = entry.get("FILIPINO") if entry.get("FILIPINO") is not None else ""
                    english = entry.get("ENGLISH") if entry.get("ENGLISH") is not None else ""
                    
                    bulos = bulos.strip() if bulos else ""
                    filipino = filipino.strip() if filipino else ""
                    english = english.strip() if english else ""
                    
                    # Create unique key to avoid duplicates
                    unique_key = f"{bulos.lower()}|{filipino.lower()}|{english.lower()}"
                    
                    # Skip if already seen, missing data, or placeholder
                    if unique_key in seen:
                        continue
                    if not (bulos and filipino and english):
                        continue
                    if bulos == "—" or filipino == "—":
                        continue
                    
                    entries.append({
                        "bul": bulos,
                        "tl": filipino,
                        "en": english
                    })
                    seen.add(unique_key)
        
        logger.info(f"Extracted {len(entries)} unique vocabulary entries from alphabet")
        return entries
    
    async def _load_sentence_data(self) -> List[Dict[str, str]]:
        """
        Load and parse sentence.json to extract sentence-level translations
        
        Returns:
            List of dictionaries with 'bul', 'tl', and 'en' translations
        """
        sentence_path = Path(__file__).parent.parent / "sentence.json"
        
        if not sentence_path.exists():
            logger.warning(f"Sentence file not found at {sentence_path}")
            return []
        
        with open(sentence_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract entries
        entries = []
        for entry in data.get("entries", []):
            bulos = entry.get("BULOS") if entry.get("BULOS") is not None else ""
            filipino = entry.get("FILIPINO") if entry.get("FILIPINO") is not None else ""
            english = entry.get("ENGLISH") if entry.get("ENGLISH") is not None else ""
            
            bulos = bulos.strip() if bulos else ""
            filipino = filipino.strip() if filipino else ""
            english = english.strip() if english else ""
            
            # Include all entries (sentences can be long)
            if bulos and filipino and english:
                entries.append({
                    "bul": bulos,
                    "tl": filipino,
                    "en": english
                })
        
        logger.info(f"Extracted {len(entries)} sentence entries")
        return entries
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for lookup.
        
        Handles:
        - Diacritic/accent stripping (NFD decomposition + combining-mark removal)
          so that "ulù", "ulu", and "ULÙ" all produce the same key.
          This is essential for Bulos words which carry grave/acute accents in
          the source data but are often typed without them by users.
        - Lowercase conversion
        - Whitespace normalization
        - Trailing punctuation removal
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text (lowercase, accent-free, trimmed, no duplicate spaces)
        """
        # Decompose to NFD so each accented character becomes base + combining mark,
        # then discard all combining marks (Unicode category "Mn").
        # e.g. "ulù" (U+00F9) → "u" "l" U+0075 U+0300 → "ulu"
        normalized = unicodedata.normalize('NFD', text)
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Trim surrounding whitespace
        normalized = normalized.strip()
        
        # Collapse internal whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove trailing punctuation for lookup (but preserve original for response)
        normalized = normalized.rstrip('.!?,;:')
        
        return normalized
    
    def _build_translation_indexes(self, dictionary_data: List[Dict[str, str]], sentence_data: List[Dict[str, str]]) -> None:
        """
        Build comprehensive translation indexes from all data sources
        
        Creates phrase indexes sorted by length (longest first) for greedy matching
        
        Args:
            dictionary_data: Word-level translations
            sentence_data: Sentence-level translations
        """
        # Initialize phrase index: normalized_phrase -> [(translation, lang_pair, word_count)]
        self.phrase_index = {}
        
        # Process sentence data first (longer phrases take priority)
        for entry in sentence_data:
            bul = entry['bul']
            tl = entry['tl']
            en = entry['en']
            
            # Normalize all variants
            bul_norm = self._normalize_text(bul)
            tl_norm = self._normalize_text(tl)
            en_norm = self._normalize_text(en)
            
            # Count words (for priority sorting)
            bul_words = len(bul_norm.split())
            tl_words = len(tl_norm.split())
            en_words = len(en_norm.split())
            
            # Index all direction pairs
            # tl -> bul
            if tl_norm not in self.phrase_index:
                self.phrase_index[tl_norm] = []
            self.phrase_index[tl_norm].append((bul, "tl_to_bul", tl_words))
            
            # bul -> tl
            if bul_norm not in self.phrase_index:
                self.phrase_index[bul_norm] = []
            self.phrase_index[bul_norm].append((tl, "bul_to_tl", bul_words))
            
            # en -> bul
            if en_norm not in self.phrase_index:
                self.phrase_index[en_norm] = []
            self.phrase_index[en_norm].append((bul, "en_to_bul", en_words))
            
            # bul -> en
            if bul_norm not in self.phrase_index:
                self.phrase_index[bul_norm] = []
            self.phrase_index[bul_norm].append((en, "bul_to_en", bul_words))
            
            # en -> tl
            if en_norm not in self.phrase_index:
                self.phrase_index[en_norm] = []
            self.phrase_index[en_norm].append((tl, "en_to_tl", en_words))
            
            # tl -> en
            if tl_norm not in self.phrase_index:
                self.phrase_index[tl_norm] = []
            self.phrase_index[tl_norm].append((en, "tl_to_en", tl_words))
        
        # Process dictionary data (words and short phrases)
        for entry in dictionary_data:
            bul = entry['bul']
            tl = entry['tl']
            en = entry['en']
            
            # Normalize
            bul_norm = self._normalize_text(bul)
            tl_norm = self._normalize_text(tl)
            en_norm = self._normalize_text(en)
            
            # Count words
            bul_words = len(bul_norm.split())
            tl_words = len(tl_norm.split())
            en_words = len(en_norm.split())
            
            # Index all direction pairs (only if not already added from sentences)
            # tl -> bul
            if tl_norm not in self.phrase_index:
                self.phrase_index[tl_norm] = []
            if not any(lang_pair == "tl_to_bul" for _, lang_pair, _ in self.phrase_index[tl_norm]):
                self.phrase_index[tl_norm].append((bul, "tl_to_bul", tl_words))
            
            # bul -> tl
            if bul_norm not in self.phrase_index:
                self.phrase_index[bul_norm] = []
            if not any(lang_pair == "bul_to_tl" for _, lang_pair, _ in self.phrase_index[bul_norm]):
                self.phrase_index[bul_norm].append((tl, "bul_to_tl", bul_words))
            
            # en -> bul
            if en_norm not in self.phrase_index:
                self.phrase_index[en_norm] = []
            if not any(lang_pair == "en_to_bul" for _, lang_pair, _ in self.phrase_index[en_norm]):
                self.phrase_index[en_norm].append((bul, "en_to_bul", en_words))
            
            # bul -> en
            if bul_norm not in self.phrase_index:
                self.phrase_index[bul_norm] = []
            if not any(lang_pair == "bul_to_en" for _, lang_pair, _ in self.phrase_index[bul_norm]):
                self.phrase_index[bul_norm].append((en, "bul_to_en", bul_words))
            
            # en -> tl
            if en_norm not in self.phrase_index:
                self.phrase_index[en_norm] = []
            if not any(lang_pair == "en_to_tl" for _, lang_pair, _ in self.phrase_index[en_norm]):
                self.phrase_index[en_norm].append((tl, "en_to_tl", en_words))
            
            # tl -> en
            if tl_norm not in self.phrase_index:
                self.phrase_index[tl_norm] = []
            if not any(lang_pair == "tl_to_en" for _, lang_pair, _ in self.phrase_index[tl_norm]):
                self.phrase_index[tl_norm].append((en, "tl_to_en", tl_words))
        
        logger.info(f"Built phrase index with {len(self.phrase_index)} unique phrases")
    
    async def _translate_with_google(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text using Google Translate API
        
        Args:
            text: Text to translate
            source_lang: Source language code (en, tl)
            target_lang: Target language code (en, tl)
            
        Returns:
            Translated text or None if translation fails
        """
        try:
            # Map our language codes to Google Translate codes
            lang_map = {
                'en': 'en',
                'tl': 'tl',
                'bul': 'tl'  # Fallback: treat Bulos as Tagalog for Google
            }
            
            src = lang_map.get(source_lang, source_lang)
            dest = lang_map.get(target_lang, target_lang)
            
            # deep-translator.GoogleTranslator is synchronous; run in executor
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                None,
                lambda: GoogleTranslator(source=src, target=dest).translate(text)
            )
            
            if translated:
                logger.debug(f"Google Translate: '{text}' ({source_lang}→{target_lang}) = '{translated}'")
                return translated
            else:
                logger.warning(f"Google Translate returned empty result for: '{text}'")
                return None
                
        except Exception as e:
            logger.warning(f"Google Translate failed for '{text}' ({source_lang}→{target_lang}): {str(e)}")
            return None
    
    def _fuzzy_match_dictionary(
        self,
        word: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[Tuple[str, float]]:
        """
        Find closest dictionary match using fuzzy string matching (Levenshtein distance).
        
        This is a SUPPORTING LOOKUP TECHNIQUE within the Hybrid Translation Algorithm.
        It helps identify the intended dictionary entry when the user makes a minor typo.
        
        Rules:
        1. Only runs AFTER exact dictionary match fails
        2. Only matches single words (no phrases)
        3. Uses validated dictionary entries as candidates
        4. Returns the validated translation of the matched entry
        5. Never creates or modifies translations
        6. Rejects matches below configured threshold
        
        Args:
            word: Input word (normalized) to find closest match for
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Tuple of (validated_translation, similarity_score) or None if no acceptable match
            
        Examples:
            Input: "olu" (typo) → Matches: "ulo" (0.67) → Returns: ("ulù", 0.67)
            Input: "maata" (typo) → Matches: "mata" (0.80) → Returns: ("mala", 0.80)
            Input: "random" (no match) → Returns: None
        """
        # Check if fuzzy matching is enabled
        if not settings.fuzzy_match_enabled:
            return None
        
        # Minimum word length to prevent false positives on short words
        if len(word) < settings.fuzzy_match_min_length:
            logger.debug(f"Fuzzy match skipped: word too short ('{word}', len={len(word)})")
            return None
        
        word_norm = self._normalize_text(word)
        lang_pair = f"{source_lang}_to_{target_lang}"
        
        best_match_key = None
        best_match_translation = None
        best_score = 0.0
        
        # Search all dictionary entries for closest match
        for phrase_norm, translations in self.phrase_index.items():
            # Only match single words (skip multi-word phrases)
            if ' ' in phrase_norm:
                continue
            
            # Calculate Levenshtein similarity (0.0 to 1.0)
            score = levenshtein_similarity(word_norm, phrase_norm)
            
            if score > best_score:
                # Check if this phrase has translation for our language pair
                for translation, pair, _ in translations:
                    if pair == lang_pair:
                        best_match_key = phrase_norm
                        best_match_translation = translation
                        best_score = score
                        break
        
        # Reject if below threshold
        if best_score < settings.fuzzy_match_threshold:
            logger.debug(
                f"Fuzzy match rejected: '{word}' → '{best_match_key}' "
                f"(similarity={best_score:.3f} < threshold={settings.fuzzy_match_threshold})"
            )
            return None
        
        # Reject exact matches (should have been caught by exact lookup)
        if best_score >= 0.999:
            logger.warning(
                f"Fuzzy match found exact match: '{word}' → '{best_match_key}' "
                f"(this should have been caught by exact dictionary lookup)"
            )
            return None
        
        logger.info(
            f"Fuzzy match found: '{word}' → dictionary['{best_match_key}'] → '{best_match_translation}' "
            f"(similarity={best_score:.3f}, method=Levenshtein)"
        )
        
        return (best_match_translation, best_score)
    
    async def _translate_word_with_fallback(self, word: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a single word with Google Translate fallback for missing Bulos words
        
        Strategy for EN→BUL or TL→BUL:
        1. If target is Bulos and word not in dictionary, use Tagalog fallback
        2. For EN→BUL: translate EN→TL using Google
        3. For TL→BUL: keep original Tagalog word
        
        Args:
            word: Single word to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated word (Bulos if in dictionary, else Tagalog fallback)
        """
        # This fallback only applies when the target is Bulos, since Bulos words
        # may be missing from the dictionary — for other targets the greedy
        # matcher already handles everything and this path should not be reached.
        if target_lang != 'bul':
            logger.debug(f"_translate_word_with_fallback called for non-Bulos target '{target_lang}', returning original")
            return word
        
        # For EN→BUL: translate to Tagalog as fallback
        if source_lang == 'en':
            translated = await self._translate_with_google(word, 'en', 'tl')
            if translated:
                logger.info(f"EN→BUL fallback: '{word}' → '{translated}' (Tagalog)")
                return translated
            else:
                logger.info(f"EN→BUL fallback failed for '{word}', keeping original")
                return word
        
        # For TL→BUL: keep original Tagalog word
        elif source_lang == 'tl':
            logger.info(f"TL→BUL no match: '{word}' (kept as Tagalog)")
            return word
        
        # Default: keep original
        return word
    
    def _find_best_match_at_position(self, words: List[str], start_pos: int, source_lang: str, target_lang: str) -> Optional[Tuple[str, int]]:
        """
        Find the best (longest) matching phrase starting at a specific position
        
        Uses greedy algorithm: tries to match the longest possible phrase first
        
        Args:
            words: List of normalized words
            start_pos: Starting position in the words list
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Tuple of (translation, match_length_in_words) or None if no match
        """
        lang_pair = f"{source_lang}_to_{target_lang}"
        
        # Try matching from longest to shortest, starting at start_pos
        max_length = len(words) - start_pos
        for length in range(max_length, 0, -1):
            phrase = ' '.join(words[start_pos:start_pos+length])
            
            if phrase in self.phrase_index:
                # Find translation for this language pair
                for translation, pair, _ in self.phrase_index[phrase]:
                    if pair == lang_pair:
                        return (translation, length)
        
        return None
    
    async def _translate_text(self, text: str, source_language: str, target_language: str) -> Dict[str, Any]:
        """
        Translate text using hybrid approach: Google Translate for EN↔TL, 2-step for Bulos
        
        Strategy:
        1. EN ↔ TL: Use Google Translate directly (full translation)
        2. EN → BUL: 2-step translation (EN→TL via Google, then TL→BUL via dictionary)
        3. BUL → EN: 2-step translation (BUL→TL via dictionary, then TL→EN via Google)
        4. TL ↔ BUL: Use dictionary with Tagalog preservation for missing words
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Dictionary with translation result and confidence
        """
        text = text.strip()
        if not text:
            return {
                "translated_text": "",
                "confidence": 0.0
            }
        
        # ═══════════════════════════════════════════════════════════════
        # ROUTE 1: EN ↔ TL - Use Google Translate directly
        # ═══════════════════════════════════════════════════════════════
        if source_language in ['en', 'tl'] and target_language in ['en', 'tl']:
            logger.info(f"Using Google Translate for EN↔TL: '{text}'")
            translated = await self._translate_with_google(text, source_language, target_language)
            
            if translated:
                logger.info(f"Google Translate result: '{text}' → '{translated}'")
                return {
                    "translated_text": translated,
                    "confidence": 1.0,  # High confidence for Google Translate
                    "translation_method": "google_translate"
                }
            else:
                logger.warning(f"Google Translate failed, returning original text")
                return {
                    "translated_text": text,
                    "confidence": 0.0,
                    "translation_method": "fallback_original"
                }
        
        # ═══════════════════════════════════════════════════════════════
        # ROUTE 2: EN → BUL - 2-step translation (EN→TL→BUL)
        # ═══════════════════════════════════════════════════════════════
        if source_language == 'en' and target_language == 'bul':
            logger.info(f"2-step translation EN→TL→BUL: '{text}'")
            
            # Step 1: Translate EN → TL using Google
            tagalog_text = await self._translate_with_google(text, 'en', 'tl')
            
            if not tagalog_text:
                logger.warning(f"Google Translate EN→TL failed, using original text")
                tagalog_text = text
                intermediate_language = None
            else:
                logger.info(f"Step 1 (EN→TL): '{text}' → '{tagalog_text}'")
                intermediate_language = 'tl'
            
            # Step 2: Translate TL → BUL using dictionary (with Tagalog preservation)
            # Recursively call with TL→BUL
            result = await self._translate_text(tagalog_text, 'tl', 'bul')
            
            # Update method to indicate 2-step process
            result['translation_method'] = 'google_then_dictionary'
            result['intermediate_language'] = intermediate_language
            logger.info(f"Step 2 (TL→BUL): '{tagalog_text}' → '{result['translated_text']}'")
            
            return result
        
        # ═══════════════════════════════════════════════════════════════
        # ROUTE 3: BUL → EN - 2-step translation (BUL→TL→EN)
        # ═══════════════════════════════════════════════════════════════
        if source_language == 'bul' and target_language == 'en':
            logger.info(f"2-step translation BUL→TL→EN: '{text}'")
            
            # Step 1: Translate BUL → TL using dictionary
            result = await self._translate_text(text, 'bul', 'tl')
            tagalog_text = result['translated_text']
            
            logger.info(f"Step 1 (BUL→TL): '{text}' → '{tagalog_text}'")
            
            # Step 2: Translate TL → EN using Google
            english_text = await self._translate_with_google(tagalog_text, 'tl', 'en')
            
            if not english_text:
                logger.warning(f"Google Translate TL→EN failed, using intermediate Tagalog")
                english_text = tagalog_text
                intermediate_language = 'tl'  # Stuck at Tagalog — signal partial result
            else:
                logger.info(f"Step 2 (TL→EN): '{tagalog_text}' → '{english_text}'")
                intermediate_language = None
            
            return {
                "translated_text": english_text,
                "confidence": result['confidence'],  # Use dictionary confidence
                "translation_method": "dictionary_then_google",
                "intermediate_language": intermediate_language
            }
        
        # ═══════════════════════════════════════════════════════════════
        # ROUTE 4: TL ↔ BUL - Dictionary with Tagalog preservation
        # ═══════════════════════════════════════════════════════════════
        text_norm = self._normalize_text(text)
        
        # Try exact full-text match first
        lang_pair = f"{source_language}_to_{target_language}"
        if text_norm in self.phrase_index:
            for translation, pair, _ in self.phrase_index[text_norm]:
                if pair == lang_pair:
                    logger.info(f"Exact match found: '{text}' -> '{translation}'")
                    return {
                        "translated_text": translation,
                        "confidence": 1.0,
                        "translation_method": "dictionary"
                    }
        
        # No exact match - use greedy multi-word matching
        words = text.split()  # Original words (preserve case/punctuation for output)
        # Strip punctuation from each token individually so that e.g. "mata," matches "mata"
        words_norm = [w.strip('.!?,;:') for w in text_norm.split()]
        
        if len(words_norm) == 0:
            return {
                "translated_text": text,
                "confidence": 0.0,
                "translation_method": "tagalog_only"
            }
        
        translated_parts = []
        matched_word_count = 0
        i = 0
        
        while i < len(words_norm):
            # Try to match the longest possible phrase starting at position i
            match_found = False
            best_match_length = 0
            best_translation = None
            
            # Try from longest to shortest
            for length in range(len(words_norm) - i, 0, -1):
                phrase = ' '.join(words_norm[i:i+length])
                
                if phrase in self.phrase_index:
                    # Find translation for this language pair
                    for translation, pair, _ in self.phrase_index[phrase]:
                        if pair == lang_pair:
                            best_translation = translation
                            best_match_length = length
                            match_found = True
                            break
                    
                    if match_found:
                        break
            
            if match_found and best_translation is not None:
                # Use the matched translation
                translated_parts.append(best_translation)
                matched_word_count += best_match_length
                original_phrase = ' '.join(words[i:i+best_match_length])
                logger.debug(f"Matched at position {i}: '{original_phrase}' -> '{best_translation}' ({best_match_length} words)")
                i += best_match_length
            else:
                # No exact match found for this word/phrase
                original_word = words[i]
                word_norm = words_norm[i]
                
                # Try fuzzy matching ONLY for single words (supporting lookup technique)
                # This helps handle typos: "olu" → "ulo" → "ulù"
                fuzzy_result = self._fuzzy_match_dictionary(word_norm, source_language, target_language)
                
                if fuzzy_result:
                    # Fuzzy match found - use the validated dictionary translation
                    fuzzy_translation, similarity = fuzzy_result
                    translated_parts.append(fuzzy_translation)
                    matched_word_count += 1  # Count as matched (with confidence recorded)
                    logger.debug(
                        f"Fuzzy matched at position {i}: '{original_word}' → '{fuzzy_translation}' "
                        f"(similarity={similarity:.3f})"
                    )
                else:
                    # No fuzzy match either - preserve original word (Tagalog fallback)
                    # This ensures 100% accuracy - only use authentic Dumagat data
                    translated_parts.append(original_word)
                    logger.debug(f"No match at position {i}: '{original_word}' (preserved as-is)")
                
                i += 1
        
        # Calculate confidence based on coverage
        total_words = len(words_norm)
        confidence = matched_word_count / total_words if total_words > 0 else 0.0
        
        translated_text = ' '.join(translated_parts)
        
        # Determine translation method for TL↔BUL
        # Check if any fuzzy matches were used by checking logs or result
        if confidence == 1.0:
            method = "dictionary"
        elif confidence > 0:
            method = "dictionary_with_tagalog_preservation"
        else:
            method = "tagalog_only"
        
        logger.info(
            f"Translation result: '{text}' -> '{translated_text}' "
            f"(confidence: {confidence:.2f}, matched: {matched_word_count}/{total_words} words, method: {method})"
        )
        
        return {
            "translated_text": translated_text,
            "confidence": confidence,
            "translation_method": method
        }
    
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate
            source_language: Source language code (en, tl, or bul)
            target_language: Target language code (en, tl, or bul)
            user_id: Ignored - retained for API compatibility only

        Returns:
            Dictionary with original_text, translated_text, source_language,
            target_language, confidence, translation_method, intermediate_language.

        Raises:
            ValueError: If language pair is not supported
            asyncio.TimeoutError: If translation exceeds timeout
        """
        logger.info(f"Translation request: '{text}' ({source_language}->{target_language})")

        if source_language not in ['en', 'tl', 'bul']:
            raise ValueError(f"Invalid source_language: {source_language}. Must be one of: en, tl, bul")

        if (source_language, target_language) not in self.SUPPORTED_PAIRS:
            raise ValueError(
                f"Unsupported language pair: {source_language} -> {target_language}. "
                f"Supported pairs: {self.SUPPORTED_PAIRS}"
            )

        if source_language == target_language:
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 1.0
            }

        try:
            result = await asyncio.wait_for(
                self._translate_text(text, source_language, target_language),
                timeout=self.timeout
            )

            logger.info(f"Translation completed: '{result['translated_text']}'")

            return {
                "original_text": text,
                "translated_text": result["translated_text"],
                "source_language": source_language,
                "target_language": target_language,
                "confidence": result["confidence"],
                "translation_method": result.get("translation_method", "unknown"),
                "intermediate_language": result.get("intermediate_language")
            }

        except asyncio.TimeoutError:
            logger.error(f"Translation timeout ({self.timeout}s) exceeded for: {text}")
            raise asyncio.TimeoutError(f"Translation exceeded timeout of {self.timeout} seconds")

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}", exc_info=True)
            raise
