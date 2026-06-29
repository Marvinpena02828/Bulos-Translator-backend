"""Translation service with LSTM model integration and history tracking"""
import asyncio
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from services.database import DatabaseManager
from models.schemas import TranslationRequest, TranslationResponse
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


class TranslationService:
    """Service for translating text using LSTM encoder-decoder models"""
    
    # Supported language pairs (bidirectional)
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
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Tuple[Tokenizer, Tokenizer]] = {}
        self.max_sequence_length = 20
        
        # Model directory from config
        self.models_dir = Path(settings.translation_models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Translation timeout from config
        self.timeout = settings.translation_timeout_seconds
        
        logger.info(f"TranslationService initialized with models directory: {self.models_dir}")
    
    async def initialize(self) -> None:
        """
        Load or train LSTM models for all language pairs
        
        This method loads dictionary data, trains models if needed, and prepares
        the service for translation requests.
        """
        logger.info("Initializing translation models...")
        
        try:
            # Load dictionary data
            training_data = await self._load_dictionary_data()
            logger.info(f"Loaded {len(training_data)} training entries from dictionary")
            
            # Initialize models for each language pair
            for source_lang, target_lang in self.SUPPORTED_PAIRS:
                pair_key = f"{source_lang}_{target_lang}"
                model_path = self.models_dir / f"{pair_key}_model.h5"
                tokenizer_path = self.models_dir / f"{pair_key}_tokenizers.pkl"
                
                # Check if model already exists
                if model_path.exists() and tokenizer_path.exists():
                    logger.info(f"Loading existing model for {pair_key}")
                    await self._load_model(pair_key, model_path, tokenizer_path)
                else:
                    logger.info(f"Training new model for {pair_key}")
                    await self._train_model(pair_key, source_lang, target_lang, training_data)
            
            logger.info("All translation models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation models: {str(e)}", exc_info=True)
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
        training_data = []
        for category in dictionary.get("categories", []):
            for entry in category.get("entries", []):
                bulos = entry.get("BULOS", "").strip()
                filipino = entry.get("FILIPINO", "").strip()
                english = entry.get("ENGLISH", "").strip()
                
                # Skip entries with missing translations or placeholder values
                if bulos and filipino and english and bulos != "—" and filipino != "—":
                    training_data.append({
                        "bul": bulos.lower(),
                        "tl": filipino.lower(),
                        "en": english.lower()
                    })
        
        logger.info(f"Extracted {len(training_data)} valid training entries")
        return training_data
    
    def _prepare_training_pairs(
        self,
        source_lang: str,
        target_lang: str,
        training_data: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str]]:
        """
        Prepare source-target language pairs for training
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            training_data: List of trilingual entries
            
        Returns:
            Tuple of (source_texts, target_texts)
        """
        source_texts = []
        target_texts = []
        
        for entry in training_data:
            source_text = entry.get(source_lang)
            target_text = entry.get(target_lang)
            
            if source_text and target_text:
                source_texts.append(source_text)
                # Add start and end tokens for decoder
                target_texts.append(f"<start> {target_text} <end>")
        
        logger.info(f"Prepared {len(source_texts)} training pairs for {source_lang}->{target_lang}")
        return source_texts, target_texts
    
    def _create_tokenizers(
        self,
        source_texts: List[str],
        target_texts: List[str]
    ) -> Tuple[Tokenizer, Tokenizer]:
        """
        Create and fit tokenizers for source and target languages
        
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts
            
        Returns:
            Tuple of (source_tokenizer, target_tokenizer)
        """
        # Create tokenizers
        source_tokenizer = Tokenizer(char_level=True, filters='', lower=True)
        target_tokenizer = Tokenizer(char_level=True, filters='', lower=True)
        
        # Fit on texts
        source_tokenizer.fit_on_texts(source_texts)
        target_tokenizer.fit_on_texts(target_texts)
        
        logger.info(
            f"Created tokenizers - Source vocab: {len(source_tokenizer.word_index)}, "
            f"Target vocab: {len(target_tokenizer.word_index)}"
        )
        
        return source_tokenizer, target_tokenizer
    
    def _build_lstm_model(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int = 64,
        lstm_units: int = 128
    ) -> keras.Model:
        """
        Build LSTM encoder-decoder model for translation
        
        Args:
            source_vocab_size: Size of source language vocabulary
            target_vocab_size: Size of target language vocabulary
            embedding_dim: Dimension of embedding layer
            lstm_units: Number of LSTM units
            
        Returns:
            Compiled Keras model
        """
        # Encoder
        encoder_inputs = layers.Input(shape=(None,))
        encoder_embedding = layers.Embedding(
            source_vocab_size, embedding_dim, mask_zero=True
        )(encoder_inputs)
        encoder_lstm = layers.LSTM(lstm_units, return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.Input(shape=(None,))
        decoder_embedding = layers.Embedding(
            target_vocab_size, embedding_dim, mask_zero=True
        )(decoder_inputs)
        decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding, initial_state=encoder_states
        )
        decoder_dense = layers.Dense(target_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Build model
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Built LSTM model with {lstm_units} units")
        return model
    
    async def _train_model(
        self,
        pair_key: str,
        source_lang: str,
        target_lang: str,
        training_data: List[Dict[str, str]]
    ) -> None:
        """
        Train LSTM model for a specific language pair
        
        Args:
            pair_key: Language pair key (e.g., "bul_en")
            source_lang: Source language code
            target_lang: Target language code
            training_data: List of trilingual entries
        """
        logger.info(f"Training model for {pair_key}...")
        
        # Prepare training pairs
        source_texts, target_texts = self._prepare_training_pairs(
            source_lang, target_lang, training_data
        )
        
        if len(source_texts) < 10:
            logger.warning(f"Insufficient training data for {pair_key}: {len(source_texts)} samples")
            # Create a simple rule-based fallback
            self.models[pair_key] = "rule_based"
            return
        
        # Create tokenizers
        source_tokenizer, target_tokenizer = self._create_tokenizers(
            source_texts, target_texts
        )
        self.tokenizers[pair_key] = (source_tokenizer, target_tokenizer)
        
        # Convert texts to sequences
        source_sequences = source_tokenizer.texts_to_sequences(source_texts)
        target_sequences = target_tokenizer.texts_to_sequences(target_texts)
        
        # Pad sequences
        source_padded = pad_sequences(
            source_sequences, maxlen=self.max_sequence_length, padding='post'
        )
        target_padded = pad_sequences(
            target_sequences, maxlen=self.max_sequence_length, padding='post'
        )
        
        # Prepare decoder input and output
        decoder_input = target_padded[:, :-1]
        decoder_output = np.expand_dims(target_padded[:, 1:], -1)
        
        # Build model
        source_vocab_size = len(source_tokenizer.word_index) + 1
        target_vocab_size = len(target_tokenizer.word_index) + 1
        
        model = self._build_lstm_model(source_vocab_size, target_vocab_size)
        
        # Train model
        logger.info(f"Starting training for {pair_key}...")
        model.fit(
            [source_padded, decoder_input],
            decoder_output,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            verbose=0
        )
        
        # Save model and tokenizers
        model_path = self.models_dir / f"{pair_key}_model.h5"
        tokenizer_path = self.models_dir / f"{pair_key}_tokenizers.pkl"
        
        model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump((source_tokenizer, target_tokenizer), f)
        
        self.models[pair_key] = model
        
        logger.info(f"Model training completed and saved for {pair_key}")
    
    async def _load_model(
        self,
        pair_key: str,
        model_path: Path,
        tokenizer_path: Path
    ) -> None:
        """
        Load pre-trained model and tokenizers from disk
        
        Args:
            pair_key: Language pair key
            model_path: Path to saved model file
            tokenizer_path: Path to saved tokenizers file
        """
        try:
            # Load model
            model = keras.models.load_model(model_path)
            self.models[pair_key] = model
            
            # Load tokenizers
            with open(tokenizer_path, 'rb') as f:
                source_tokenizer, target_tokenizer = pickle.load(f)
            self.tokenizers[pair_key] = (source_tokenizer, target_tokenizer)
            
            logger.info(f"Successfully loaded model and tokenizers for {pair_key}")
            
        except Exception as e:
            logger.error(f"Failed to load model for {pair_key}: {str(e)}", exc_info=True)
            raise
    
    def _translate_with_model(
        self,
        text: str,
        pair_key: str
    ) -> str:
        """
        Translate text using trained LSTM model
        
        Args:
            text: Input text to translate
            pair_key: Language pair key
            
        Returns:
            Translated text
        """
        model = self.models.get(pair_key)
        
        # Fallback for rule-based or missing models
        if model == "rule_based" or model is None:
            logger.warning(f"Using fallback translation for {pair_key}")
            return f"[Translation: {text}]"
        
        # Get tokenizers
        source_tokenizer, target_tokenizer = self.tokenizers[pair_key]
        
        # Prepare input sequence
        text_lower = text.lower()
        input_seq = source_tokenizer.texts_to_sequences([text_lower])
        input_padded = pad_sequences(
            input_seq, maxlen=self.max_sequence_length, padding='post'
        )
        
        # Create decoder input (start token)
        start_token = target_tokenizer.word_index.get('<start>', 1)
        decoder_input = np.array([[start_token]])
        
        # Generate translation character by character
        translated_chars = []
        for _ in range(self.max_sequence_length):
            predictions = model.predict([input_padded, decoder_input], verbose=0)
            predicted_id = np.argmax(predictions[0, -1, :])
            
            # Check for end token
            if predicted_id == target_tokenizer.word_index.get('<end>', 0):
                break
            
            # Get character
            predicted_char = target_tokenizer.index_word.get(predicted_id, '')
            if predicted_char and predicted_char not in ['<start>', '<end>']:
                translated_chars.append(predicted_char)
            
            # Update decoder input
            decoder_input = np.append(decoder_input, [[predicted_id]], axis=1)
        
        translated_text = ''.join(translated_chars).strip()
        
        # If translation is empty, return input
        if not translated_text:
            logger.warning(f"Empty translation result for: {text}")
            return text
        
        return translated_text
    
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text from source to target language with timeout enforcement
        
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
        
        pair_key = f"{source_language}_{target_language}"
        
        try:
            # Run translation in executor with timeout
            loop = asyncio.get_event_loop()
            translated_text = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._translate_with_model,
                    text,
                    pair_key
                ),
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
                "confidence": None  # LSTM models don't provide confidence scores easily
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
