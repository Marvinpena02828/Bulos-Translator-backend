"""Translation service with comprehensive dictionary-based translation"""
import asyncio
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from googletrans import Translator
from services.database import DatabaseManager
from models.schemas import TranslationRequest, TranslationResponse
from config import settings
from utils.logging_config import get_logger

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
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize translation service
        
        Args:
            db_manager: Database manager instance for MongoDB operations
        """
        self.db = db_manager
        self.history_collection = "history"
        
        # Dictionary storage - will hold all translation data
        self.phrase_index: Dict[str, List[Tuple[str, str, int]]] = {}  # phrase -> [(translation, lang_pair, word_count)]
        
        # Google Translator instance (for EN ↔ TL and fallback translations)
        self.google_translator = Translator()
        
        # Translation timeout from config
        self.timeout = settings.translation_timeout_seconds
        
        logger.info(f"TranslationService initialized with timeout: {self.timeout}s")
        logger.info("Google Translate enabled for EN↔TL and Tagalog fallback")
    
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
        Normalize text for lookup with Unicode normalization
        
        Handles:
        - Unicode normalization (Ã, Ã±, accents)
        - Lowercase conversion
        - Whitespace normalization
        - Punctuation removal
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text (lowercase, trimmed, no duplicate spaces, normalized unicode)
        """
        # Normalize Unicode characters (NFC = composed form)
        # This handles Ã → o, Ã± → ñ, etc.
        normalized = unicodedata.normalize('NFC', text)
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Trim spaces
        normalized = normalized.strip()
        
        # Remove duplicate spaces
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
            
            # googletrans 4.x translate() is async, call it directly
            result = await self.google_translator.translate(text, src=src, dest=dest)
            
            if result and hasattr(result, 'text') and result.text:
                logger.debug(f"Google Translate: '{text}' ({source_lang}→{target_lang}) = '{result.text}'")
                return result.text
            else:
                logger.warning(f"Google Translate returned empty result for: '{text}'")
                return None
                
        except Exception as e:
            logger.warning(f"Google Translate failed for '{text}' ({source_lang}→{target_lang}): {str(e)}")
            return None
    
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
        # For non-Bulos targets, return original (shouldn't reach here)
        if target_lang != 'bul':
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
            else:
                logger.info(f"Step 1 (EN→TL): '{text}' → '{tagalog_text}'")
            
            # Step 2: Translate TL → BUL using dictionary (with Tagalog preservation)
            # Recursively call with TL→BUL
            result = await self._translate_text(tagalog_text, 'tl', 'bul')
            
            # Update method to indicate 2-step process
            result['translation_method'] = 'google_then_dictionary'
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
            else:
                logger.info(f"Step 2 (TL→EN): '{tagalog_text}' → '{english_text}'")
            
            return {
                "translated_text": english_text,
                "confidence": result['confidence'],  # Use dictionary confidence
                "translation_method": "dictionary_then_google"
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
                        "confidence": 1.0
                    }
        
        # No exact match - use greedy multi-word matching
        words = text.split()  # Original words (preserve case/punctuation)
        words_norm = text_norm.split()  # Normalized for lookup
        
        if len(words_norm) == 0:
            return {
                "translated_text": text,
                "confidence": 0.0
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
                # No exact match found - preserve original word
                # This ensures 100% accuracy - only use authentic Dumagat data
                original_word = words[i]
                translated_parts.append(original_word)
                logger.debug(f"No match at position {i}: '{original_word}' (preserved as-is)")
                i += 1
        
        # Calculate confidence based on coverage
        total_words = len(words_norm)
        confidence = matched_word_count / total_words if total_words > 0 else 0.0
        
        translated_text = ' '.join(translated_parts)
        
        # Determine translation method for TL↔BUL
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
        Translate text from source to target language
        
        Uses comprehensive dictionary-based approach with verified JSON data
        
        Args:
            text: Text to translate
            source_language: Source language code (en, tl, or bul)
            target_language: Target language code (en, tl, or bul)
            user_id: Optional user ID for history tracking
            
        Returns:
            Dictionary with translation result including:
            - original_text
            - translated_text
            - source_language
            - target_language
            - confidence (0.0-1.0 based on actual dictionary coverage)
            
        Raises:
            ValueError: If language pair is not supported
            asyncio.TimeoutError: If translation exceeds timeout
        """
        logger.info(f"Translation request: '{text}' ({source_language}->{target_language})")
        
        # Validate that source_language is one of our supported languages
        if source_language not in ['en', 'tl', 'bul']:
            logger.error(f"Invalid source_language: {source_language}")
            raise ValueError(f"Invalid source_language: {source_language}. Must be one of: en, tl, bul")
        
        # Validate language pair
        if (source_language, target_language) not in self.SUPPORTED_PAIRS:
            logger.warning(f"Unsupported language pair: {source_language}->{target_language}")
            raise ValueError(
                f"Unsupported language pair: {source_language} -> {target_language}. "
                f"Supported pairs: {self.SUPPORTED_PAIRS}"
            )
        
        # Check if source and target are the same
        if source_language == target_language:
            logger.info(f"No translation needed: source={source_language}, target={target_language}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 1.0
            }
        
        try:
            # Run translation with timeout
            result = await asyncio.wait_for(
                self._translate_text(text, source_language, target_language),
                timeout=self.timeout
            )
            
            # Record in history if user_id provided
            if user_id:
                await self._record_history(
                    user_id=user_id,
                    action_type="translation",
                    outcome="success",
                    details={
                        "source_language": source_language,
                        "target_language": target_language,
                        "original_text": text,
                        "translated_text": result["translated_text"],
                        "confidence": result["confidence"]
                    }
                )
            
            logger.info(f"Translation completed: '{result['translated_text']}'")
            
            return {
                "original_text": text,
                "translated_text": result["translated_text"],
                "source_language": source_language,
                "target_language": target_language,
                "confidence": result["confidence"],
                "translation_method": result.get("translation_method", "unknown")
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Translation timeout ({self.timeout}s) exceeded for: {text}")
            
            # Record failure in history
            if user_id:
                await self._record_history(
                    user_id=user_id,
                    action_type="translation",
                    outcome="timeout",
                    details={
                        "source_language": source_language,
                        "target_language": target_language,
                        "original_text": text,
                        "error": "Translation timeout"
                    }
                )
            
            raise asyncio.TimeoutError(f"Translation exceeded timeout of {self.timeout} seconds")
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}", exc_info=True)
            
            # Record failure in history
            if user_id:
                await self._record_history(
                    user_id=user_id,
                    action_type="translation",
                    outcome="error",
                    details={
                        "source_language": source_language,
                        "target_language": target_language,
                        "original_text": text,
                        "error": str(e)
                    }
                )
            
            raise
    
    async def _record_history(
        self,
        user_id: str,
        action_type: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record translation action in history collection
        
        Args:
            user_id: User ID performing the action
            action_type: Type of action (translation)
            outcome: Outcome of the action (success, timeout, error)
            details: Additional details about the action
        """
        try:
            history_record = {
                "user_id": user_id,
                "action_type": action_type,
                "resource_type": "translation",
                "outcome": outcome,
                "timestamp": datetime.utcnow(),
                "details": details or {}
            }
            
            await self.db.insert_one(self.history_collection, history_record)
            logger.debug(f"History recorded: {action_type} for user {user_id}")
            
        except Exception as e:
            # Don't fail the main operation if history recording fails
            logger.error(f"Failed to record history: {str(e)}", exc_info=True)
