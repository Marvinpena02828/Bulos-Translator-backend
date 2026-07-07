"""Translation service with hybrid approach: Dictionary + Google Translate"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from googletrans import Translator
import aiohttp

from services.database import DatabaseManager
from models.schemas import TranslationRequest, TranslationResponse
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


class TranslationService:
    """Service for translating text using dictionary lookup and Google Translate API"""
    
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
        
        # Dictionary storage
        self.dictionary: Dict[str, Dict[str, str]] = {}
        
        # Google Translate client
        self.translator = Translator()
        
        # Translation timeout from config
        self.timeout = settings.translation_timeout_seconds
        
        logger.info(f"TranslationService initialized with timeout: {self.timeout}s")
    
    async def initialize(self) -> None:
        """
        Load dictionary data for Bulos translations
        
        This method loads the trilingual dictionary for Bulos word lookups.
        English <-> Tagalog translations use Google Translate API.
        """
        logger.info("Initializing translation service...")
        
        try:
            # Load dictionary data
            dictionary_data = await self._load_dictionary_data()
            logger.info(f"Loaded {len(dictionary_data)} entries from dictionary")
            
            # Index dictionary for fast lookups
            self._index_dictionary(dictionary_data)
            
            logger.info("Translation service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {str(e)}", exc_info=True)
            raise
    
    async def _load_dictionary_data(self) -> List[Dict[str, str]]:
        """
        Load and parse dictionary.json to extract trilingual vocabulary
        
        Returns:
            List of dictionaries with 'bul', 'en', and 'tl' translations
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
                        "bul": bulos.lower(),
                        "tl": filipino.lower(),
                        "en": english.lower()
                    })
        
        logger.info(f"Extracted {len(entries)} valid dictionary entries")
        return entries
    
    def _index_dictionary(self, entries: List[Dict[str, str]]) -> None:
        """
        Index dictionary entries for fast lookup
        
        Creates indexes for each language to enable O(1) lookups
        Indexes both individual words AND complete phrases
        """
        self.dictionary = {
            'bul_to_en': {},
            'bul_to_tl': {},
            'en_to_bul': {},
            'tl_to_bul': {},
            # Phrase indexes (for multi-word expressions)
            'phrases_tl_to_bul': {},
            'phrases_en_to_bul': {},
        }
        
        for entry in entries:
            bul = entry['bul']
            tl = entry['tl']
            en = entry['en']
            
            # Check if this is a phrase (contains spaces) or single word
            is_phrase = ' ' in tl or ' ' in bul
            
            if is_phrase:
                # Index as phrase
                # Normalize: lowercase, remove trailing punctuation
                tl_normalized = tl.lower().strip().rstrip('.!?')
                en_normalized = en.lower().strip().rstrip('.!?')
                bul_normalized = bul.lower().strip().rstrip('.!?')
                
                self.dictionary['phrases_tl_to_bul'][tl_normalized] = bul_normalized
                self.dictionary['phrases_en_to_bul'][en_normalized] = bul_normalized
                
                logger.debug(f"Indexed phrase: '{tl_normalized}' -> '{bul_normalized}'")
            
            # Also index as individual words (for word-by-word fallback)
            # Bulos mappings
            self.dictionary['bul_to_en'][bul] = en
            self.dictionary['bul_to_tl'][bul] = tl
            
            # Reverse mappings (for single word lookups)
            self.dictionary['en_to_bul'][en] = bul
            self.dictionary['tl_to_bul'][tl] = bul
        
        logger.info(
            f"Dictionary indexed: "
            f"{len(self.dictionary['bul_to_en'])} Bulos entries, "
            f"{len(self.dictionary['en_to_bul'])} English entries, "
            f"{len(self.dictionary['tl_to_bul'])} Tagalog entries, "
            f"{len(self.dictionary['phrases_tl_to_bul'])} Tagalog phrases, "
            f"{len(self.dictionary['phrases_en_to_bul'])} English phrases"
        )
    
    def _is_single_word(self, text: str) -> bool:
        """Check if text is a single word (no spaces, basic punctuation only)"""
        # Remove common punctuation
        cleaned = text.strip().lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
        return ' ' not in cleaned
    
    async def _translate_with_google(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, float]:
        """
        Translate using Google Translate API
        
        Args:
            text: Text to translate
            source_lang: Source language code (en, tl)
            target_lang: Target language code (en, tl)
            
        Returns:
            Tuple of (translated_text, confidence)
        """
        # Map our codes to Google Translate codes
        google_lang_map = {
            'en': 'en',
            'tl': 'tl',
            'fil': 'tl',
        }
        
        src = google_lang_map.get(source_lang, source_lang)
        dest = google_lang_map.get(target_lang, target_lang)
        
        logger.info(f"Using Google Translate: {text} ({src} -> {dest})")
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.translator.translate(text, src=src, dest=dest)
            )
            
            translated_text = result.text
            # Google Translate doesn't provide confidence, use a high default
            confidence = 0.95
            
            logger.info(f"Google Translate result: {translated_text}")
            return translated_text, confidence
            
        except Exception as e:
            logger.error(f"Google Translate error: {str(e)}", exc_info=True)
            # Fallback to original text
            return text, 0.0
    
    def _lookup_phrase(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Look up complete phrase in dictionary
        
        Args:
            text: Phrase to look up
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated phrase if found, None otherwise
        """
        # Normalize: lowercase, remove trailing punctuation
        text_normalized = text.lower().strip().rstrip('.!?')
        
        # Only support phrase lookup for Tagalog/English -> Bulos
        if target_lang == 'bul':
            if source_lang == 'tl':
                result = self.dictionary['phrases_tl_to_bul'].get(text_normalized)
                if result:
                    logger.info(f"Phrase match: '{text}' ({source_lang}) -> '{result}' ({target_lang})")
                    return result
            elif source_lang == 'en':
                result = self.dictionary['phrases_en_to_bul'].get(text_normalized)
                if result:
                    logger.info(f"Phrase match: '{text}' ({source_lang}) -> '{result}' ({target_lang})")
                    return result
        
        return None
    
    def _lookup_dictionary(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Look up translation in dictionary
        
        Args:
            text: Text to look up (should be single word)
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text if found, None otherwise
        """
        text_lower = text.strip().lower()
        
        # Direct lookup for supported pairs
        lookup_key = f"{source_lang}_to_{target_lang}"
        if lookup_key in self.dictionary:
            result = self.dictionary[lookup_key].get(text_lower)
            if result:
                logger.info(f"Dictionary hit: {text} ({source_lang}) -> {result} ({target_lang})")
                return result
        
        # Bridge through Tagalog for Bulos translations not in direct mapping
        # E.g., en -> bul: first translate en -> tl with Google, then tl -> bul from dictionary
        if source_lang == 'en' and target_lang == 'bul':
            # Check if we have this English word mapped to Bulos
            bul_result = self.dictionary['en_to_bul'].get(text_lower)
            if bul_result:
                return bul_result
        
        if source_lang == 'bul' and target_lang == 'en':
            en_result = self.dictionary['bul_to_en'].get(text_lower)
            if en_result:
                return en_result
        
        logger.debug(f"Dictionary miss: {text} ({source_lang} -> {target_lang})")
        return None
    
    async def _translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Tuple[str, Optional[float]]:
        """
        Synchronous translation logic
        
        Strategy:
        1. If source or target is Bulos: use dictionary lookup (word-by-word for sentences)
        2. If English <-> Tagalog: use Google Translate
        3. For Bulos: if not in dictionary, use Tagalog as fallback for that word
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Tuple of (translated_text, confidence_score)
        """
        text = text.strip()
        is_single_word = self._is_single_word(text)
        
        # Case 1: Bulos involved - always try dictionary first
        if 'bul' in [source_language, target_language]:
            # For single words, try dictionary lookup
            if is_single_word:
                dict_result = self._lookup_dictionary(text, source_language, target_language)
                if dict_result:
                    return dict_result, 1.0  # Perfect confidence for dictionary lookups
            
            # If not found in dictionary and it's Bulos -> X
            if source_language == 'bul':
                if is_single_word:
                    logger.warning(f"Bulos word not in dictionary: {text}")
                    return f"[Unknown: {text}]", 0.0
                else:
                    # Try word-by-word translation
                    return await self._translate_bulos_sentence(text, target_language), 0.8
            
            # If X -> Bulos, use word-by-word translation with Tagalog fallback
            if target_language == 'bul':
                logger.info(f"Translating to Bulos with Tagalog fallback: {text} ({source_language} -> bul)")
                return await self._translate_to_bulos_hybrid(text, source_language), 0.85
        
        # Case 2: English <-> Tagalog - use Google Translate
        if source_language in ['en', 'tl'] and target_language in ['en', 'tl']:
            return await self._translate_with_google(text, source_language, target_language)
        
        # Fallback
        logger.warning(f"Unsupported translation: {text} ({source_language} -> {target_language})")
        return text, 0.0
    
    async def _translate_to_bulos_hybrid(self, text: str, source_language: str) -> str:
        """
        Translate text to Bulos using hybrid phrase + word-by-word approach
        
        Strategy:
        1. Try phrase matching on complete sentence
        2. If not found, split by sentence delimiters and try each segment
        3. For unmatched segments, use word-by-word translation
        4. Words not in dictionary fall back to Tagalog
        
        Args:
            text: Input text
            source_language: Source language (en or tl)
            
        Returns:
            Hybrid Bulos/Tagalog text
        """
        import re
        
        # Step 1: Get Tagalog translation of entire sentence
        if source_language == 'en':
            tagalog_text, _ = await self._translate_with_google(text, 'en', 'tl')
        elif source_language == 'tl':
            tagalog_text = text
        else:
            return text
        
        logger.info(f"Tagalog intermediate: {tagalog_text}")
        
        # Step 2: Try phrase matching on complete text first
        phrase_match = self._lookup_phrase(tagalog_text, 'tl', 'bul')
        if phrase_match:
            logger.info(f"Complete phrase matched: {phrase_match}")
            return phrase_match
        
        # Step 3: Split by sentence boundaries (periods, question marks, exclamations)
        # Keep delimiters for reconstruction
        segments = re.split(r'([.!?]+\s*)', tagalog_text)
        
        translated_segments = []
        
        for i, segment in enumerate(segments):
            # Skip empty segments
            if not segment.strip():
                translated_segments.append(segment)
                continue
            
            # If this is a delimiter (punctuation), keep it as-is
            if re.match(r'^[.!?]+\s*$', segment):
                translated_segments.append(segment)
                continue
            
            # Try phrase matching on this segment
            segment_phrase = self._lookup_phrase(segment.strip(), 'tl', 'bul')
            if segment_phrase:
                logger.info(f"Segment phrase matched: '{segment.strip()}' -> '{segment_phrase}'")
                translated_segments.append(segment_phrase)
                continue
            
            # No phrase match - do word-by-word translation
            logger.debug(f"No phrase match for segment: '{segment}', using word-by-word")
            translated_segment = await self._translate_segment_word_by_word(segment)
            translated_segments.append(translated_segment)
        
        result = ''.join(translated_segments)
        logger.info(f"Hybrid translation result: {result}")
        return result
    
    async def _translate_segment_word_by_word(self, segment: str) -> str:
        """
        Translate a text segment word-by-word
        
        Args:
            segment: Text segment to translate
            
        Returns:
            Translated segment
        """
        import re
        
        # Better word tokenization - split on whitespace and punctuation
        # But keep punctuation attached to words for later reconstruction
        tokens = re.findall(r'\S+|\s+', segment)
        
        translated_tokens = []
        
        for token in tokens:
            # If it's whitespace, keep it as-is
            if token.isspace():
                translated_tokens.append(token)
                continue
            
            # For word tokens, clean and look up
            word = token
            clean_word = word.lower().strip('.,!?;:"\'')
            
            # Handle Tagalog grammatical particles
            # "magandang" = "maganda" + "ng" (linker)
            base_word = clean_word
            suffix = ""
            
            # Check for common Tagalog suffixes and separate them
            if clean_word.endswith('ng') and len(clean_word) > 2:
                base_word = clean_word[:-2]
                suffix = 'ng'
            
            # Try to find Bulos equivalent for base word
            bulos_word = self._lookup_dictionary(base_word, 'tl', 'bul')
            
            if not bulos_word and base_word != clean_word:
                # Try the full word if base word not found
                bulos_word = self._lookup_dictionary(clean_word, 'tl', 'bul')
                suffix = ""  # Reset suffix if we found the full word
            
            if bulos_word:
                # Found in dictionary - use Bulos word
                result_word = bulos_word + suffix
                
                # Preserve original punctuation from the word
                # Find what punctuation was stripped
                prefix_punct = ""
                suffix_punct = ""
                
                # Get leading punctuation
                for i, c in enumerate(word):
                    if c.lower() not in 'abcdefghijklmnopqrstuvwxyzñ':
                        prefix_punct += c
                    else:
                        break
                
                # Get trailing punctuation
                for i in range(len(word) - 1, -1, -1):
                    c = word[i]
                    if c.lower() not in 'abcdefghijklmnopqrstuvwxyzñ':
                        suffix_punct = c + suffix_punct
                    else:
                        break
                
                result_word = prefix_punct + result_word + suffix_punct
                translated_tokens.append(result_word)
                logger.debug(f"Token: {word} -> {result_word} (Bulos)")
            else:
                # Not in dictionary - keep original
                translated_tokens.append(word)
                logger.debug(f"Token: {word} -> {word} (Tagalog fallback)")
        
        return ''.join(translated_tokens)
    
    async def _translate_bulos_sentence(self, text: str, target_language: str) -> str:
        """
        Translate Bulos sentence to target language word-by-word
        
        Args:
            text: Bulos text
            target_language: Target language (en or tl)
            
        Returns:
            Translated text
        """
        words = text.split()
        translated_words = []
        
        for word in words:
            # Clean punctuation for lookup
            clean_word = word.lower().strip('.,!?;:"\'')
            
            # Look up in dictionary
            trans_word = self._lookup_dictionary(clean_word, 'bul', target_language)
            
            if trans_word:
                # Preserve punctuation
                if word != clean_word:
                    punct = word[len(clean_word):]
                    translated_words.append(trans_word + punct)
                else:
                    translated_words.append(trans_word)
            else:
                # Unknown word - keep original
                translated_words.append(f"[{word}]")
        
        return ' '.join(translated_words)
    
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text from source to target language with timeout enforcement
        
        Uses hybrid approach:
        - Bulos translations: Dictionary lookup with Tagalog bridge
        - English <-> Tagalog: Google Translate API
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            user_id: Optional user ID for history tracking
            
        Returns:
            Dictionary with translation result
            
        Raises:
            ValueError: If language pair is not supported
            asyncio.TimeoutError: If translation exceeds timeout
        """
        logger.info(
            f"Translation request: '{text}' ({source_language}->{target_language})"
        )
        
        # Validate language pair
        if (source_language, target_language) not in self.SUPPORTED_PAIRS:
            logger.warning(
                f"Unsupported language pair: {source_language}->{target_language}"
            )
            raise ValueError(
                f"Unsupported language pair: {source_language} -> {target_language}. "
                f"Supported pairs: {self.SUPPORTED_PAIRS}"
            )
        
        try:
            # Run translation with timeout
            translated_text, confidence = await asyncio.wait_for(
                self._translate_sync(text, source_language, target_language),
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
                        "translated_text": translated_text
                    }
                )
            
            logger.info(f"Translation completed: '{translated_text}'")
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": confidence
            }
            
        except asyncio.TimeoutError:
            logger.error(
                f"Translation timeout ({self.timeout}s) exceeded for: {text}"
            )
            
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
            
            raise asyncio.TimeoutError(
                f"Translation exceeded timeout of {self.timeout} seconds"
            )
            
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
