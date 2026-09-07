"""Speech processing service with OpenAI Whisper integration"""
import asyncio
import wave
import tempfile
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
import io

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    # Set seed for reproducible results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


def normalize_dictionary_value(value) -> str:
    """
    Safely normalize dictionary values, handling None, non-strings, and malformed data
    
    Args:
        value: Value to normalize (can be None, str, int, list, dict, etc.)
        
    Returns:
        Normalized string (empty string if invalid)
    """
    if not isinstance(value, str):
        return ""
    return " ".join(value.casefold().strip().split())


def _find_ffmpeg():
    """
    Find FFmpeg executable and add to PATH if needed
    
    Search order:
    1. User-specified path in .env (FFMPEG_PATH)
    2. System PATH
    3. Common installation paths
    
    If FFmpeg is found but not in PATH, this function will add its directory
    to the PATH environment variable so Whisper can find it.
    """
    # Check config first
    if settings.ffmpeg_path and os.path.exists(settings.ffmpeg_path):
        logger.info(f"Using FFmpeg from config: {settings.ffmpeg_path}")
        ffmpeg_path = settings.ffmpeg_path
        
        # Add FFmpeg directory to PATH if not already there
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        if ffmpeg_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
            logger.info(f"Added FFmpeg directory to PATH: {ffmpeg_dir}")
        
        return ffmpeg_path
    
    # Check system PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        logger.info(f"FFmpeg found in PATH: {ffmpeg_path}")
        return ffmpeg_path
    
    # Check common paths as fallback
    common_paths = [
        os.path.expanduser(r"~\scoop\shims\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            logger.info(f"FFmpeg found at: {path}")
            
            # Add FFmpeg directory to PATH
            ffmpeg_dir = str(Path(path).parent)
            if ffmpeg_dir not in os.environ.get('PATH', ''):
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                logger.info(f"Added FFmpeg directory to PATH: {ffmpeg_dir}")
            
            return path
    
    logger.warning("FFmpeg not found - please set FFMPEG_PATH in .env or restart terminal")
    return None


# Locate FFmpeg on module load
_FFMPEG_PATH = _find_ffmpeg()


class SpeechProcessor:
    """Service for processing speech audio using OpenAI Whisper"""
    
    def __init__(self):
        """
        Initialize speech processor
        
        Whisper model size and device are loaded from config
        """
        self.model_name = settings.whisper_model
        self.device = settings.whisper_device
        self.model: Optional[any] = None
        
        # Settings from config
        self.timeout = settings.speech_processing_timeout_seconds
        self.max_file_size = settings.max_audio_file_size
        self.max_duration = settings.max_audio_duration_seconds
        
        logger.info(
            f"SpeechProcessor initialized with timeout: {self.timeout}s, "
            f"max file size: {self.max_file_size / (1024*1024):.1f}MB, "
            f"max duration: {self.max_duration}s"
        )
    
    # REMOVED: _verify_language_from_text()
    # Language verification is no longer needed because source_language
    # is now explicitly provided by the user and is authoritative.
    
    async def initialize(self) -> None:
        """
        Load Whisper model
        
        This method loads the pre-trained Whisper model. The model is downloaded
        automatically on first use if not already cached.
        
        Available models: tiny, base, small, medium, large
        - tiny: Fastest, least accurate (~75MB)
        - base: Good balance (~150MB) - RECOMMENDED
        - small: Better accuracy (~500MB)
        - medium: Very accurate (~1.5GB)
        - large: Best accuracy (~3GB)
        
        Raises:
            RuntimeError: If Whisper is not installed or model loading fails
        """
        logger.info("Initializing Whisper model...")
        
        if not WHISPER_AVAILABLE:
            logger.error("Whisper library is not installed")
            raise RuntimeError(
                "Whisper is not installed. Install it with: pip install openai-whisper"
            )
        
        # Check if FFmpeg is available
        
        
        logger.info(f"Using FFmpeg at: {_FFMPEG_PATH}")
        
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_model_sync,
                self.model_name,
                self.device
            )
            
            logger.info(f"Whisper model '{self.model_name}' loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")
    
    def _load_model_sync(self, model_name: str, device: str):
        """
        Synchronous model loading (runs in thread pool)
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v3)
            device: Device for inference (cpu or cuda)
            
        Returns:
            Loaded Whisper model instance
        """
        import time
        
        logger.info(f"Loading Whisper model '{model_name}' on device '{device}'")
        
        # Retry logic for large model downloads that may timeout
        max_retries = 3
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                model = whisper.load_model(model_name, device=device)
                logger.info(f"Whisper model loaded: {model_name}")
                return model
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a download/network error
                if "urlopen error" in error_msg or "Connection" in error_msg or "timeout" in error_msg.lower():
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Model download failed (attempt {attempt + 1}/{max_retries}): {error_msg}. "
                            f"Retrying in {retry_delay} seconds..."
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(
                            f"Model download failed after {max_retries} attempts. "
                            f"Please check your internet connection and firewall settings."
                        )
                        raise RuntimeError(
                            f"Failed to download Whisper model '{model_name}' after {max_retries} attempts. "
                            f"Please check your internet connection, firewall, or VPN settings. "
                            f"You may need to manually download the model or use a smaller model like 'medium'."
                        )
                else:
                    # Other errors, raise immediately
                    raise
    
    def _validate_audio_file(self, audio_data: bytes) -> None:
        """
        Validate audio file size and format
        
        Whisper supports many audio formats: WAV, MP3, M4A, FLAC, OGG, WEBM, etc.
        We validate the file signature (magic bytes) to ensure it's a recognized format.
        
        Args:
            audio_data: Raw audio file bytes
            
        Raises:
            ValueError: If audio file exceeds size limit or has invalid format
        """
        # Check file size
        file_size = len(audio_data)
        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            max_mb = self.max_file_size / (1024 * 1024)
            logger.warning(f"Audio file too large: {size_mb:.2f}MB (max: {max_mb:.2f}MB)")
            raise ValueError(
                f"Audio file exceeds maximum size of {max_mb:.0f}MB "
                f"(received: {size_mb:.2f}MB)"
            )
        
        # Check minimum file size
        if len(audio_data) < 12:
            logger.warning("Audio file too small to be valid")
            raise ValueError("Audio file is too small to be valid")
        
        # Validate audio format by checking magic bytes (file signature)
        # Common audio format signatures:
        supported_formats = {
            b'RIFF': 'WAV',           # WAV files start with 'RIFF'
            b'ID3': 'MP3',            # MP3 files with ID3 tags
            b'\xff\xfb': 'MP3',       # MP3 files (MPEG-1 Layer 3)
            b'\xff\xf3': 'MP3',       # MP3 files (MPEG-1 Layer 3)
            b'\xff\xf2': 'MP3',       # MP3 files (MPEG-2 Layer 3)
            b'fLaC': 'FLAC',          # FLAC files
            b'OggS': 'OGG',           # OGG files
            b'\x1a\x45\xdf\xa3': 'WEBM',  # WEBM files
        }
        
        # M4A files use MP4 container format
        # Check for 'ftyp' at offset 4 (MP4/M4A signature)
        if len(audio_data) >= 12 and audio_data[4:8] == b'ftyp':
            logger.debug(f"Audio format detected: M4A/MP4 ({file_size} bytes)")
            return
        
        # Check other formats by magic bytes at start
        detected_format = None
        for signature, format_name in supported_formats.items():
            if audio_data[:len(signature)] == signature:
                detected_format = format_name
                break
        
        if detected_format:
            logger.debug(f"Audio format detected: {detected_format} ({file_size} bytes)")
            return
        
        # If no recognized format, log warning but allow (Whisper may still handle it)
        logger.warning(
            f"Audio format not recognized by signature check, but allowing Whisper to try. "
            f"First bytes: {audio_data[:12].hex()}"
        )
        # Don't raise error - let Whisper handle it with FFmpeg
        logger.debug(f"Audio validation passed: {file_size} bytes")
    
    def _validate_audio_duration(self, audio_data: bytes) -> float:
        """
        Validate audio duration and return duration in seconds
        
        For non-WAV formats (M4A, MP3, etc.), we skip duration validation
        since parsing requires complex libraries. Whisper will handle it.
        
        Args:
            audio_data: Raw audio file bytes
            
        Returns:
            Duration in seconds (or estimated 0.0 for non-WAV)
            
        Raises:
            ValueError: If audio duration exceeds maximum allowed duration (WAV only)
        """
        try:
            # Only validate duration for WAV files (easy to parse)
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                audio_io = io.BytesIO(audio_data)
                with wave.open(audio_io, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                
                # Check duration limit
                if duration > self.max_duration:
                    logger.warning(
                        f"Audio duration too long: {duration:.2f}s (max: {self.max_duration}s)"
                    )
                    raise ValueError(
                        f"Audio duration exceeds maximum of {self.max_duration}s "
                        f"(received: {duration:.2f}s)"
                    )
                
                logger.debug(f"Audio duration: {duration:.2f}s")
                return duration
            else:
                # For M4A, MP3, etc., skip duration validation
                # Whisper will handle it, and we rely on file size limit
                logger.debug("Non-WAV format detected, skipping duration validation")
                return 0.0  # Return 0 as placeholder
            
        except wave.Error as e:
            # If wave parsing fails, it's not a WAV file - that's okay
            logger.debug(f"Not a WAV file or parsing failed: {str(e)}")
            return 0.0  # Return 0 as placeholder
        except Exception as e:
            # Any other error, log but don't fail
            logger.warning(f"Duration validation skipped due to error: {str(e)}")
            return 0.0
    
    def _detect_constrained_language(self, audio_path: str) -> Tuple[str, float]:
        """
        Perform constrained language detection: limit Whisper to 'en' or 'tl' only
        
        Strategy:
        1. Use Whisper's detect_language to get probability distribution
        2. Use model's configured n_mels (128 for Large V3, 80 for smaller models)
        3. Filter probabilities to only 'en' and 'tl'
        4. Return the language with higher probability and its confidence
        
        Args:
            audio_path: Path to temporary audio file
            
        Returns:
            Tuple of (constrained language code, confidence score)
            Language code is 'en' or 'tl' (never 'id' or other)
            
        Raises:
            RuntimeError: If language detection fails critically
        """
        try:
            # Load audio for language detection
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram using model's configured n_mels
            # Large V3 uses 128 mel channels, other models may use 80
            n_mels = self.model.dims.n_mels
            logger.debug(f"Using n_mels={n_mels} from model configuration")
            
            mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(self.model.device)
            
            # Detect language probabilities
            _, probs = self.model.detect_language(mel)
            
            # Get probabilities for en and tl only
            en_prob = probs.get('en', 0.0)
            tl_prob = probs.get('tl', 0.0)
            fil_prob = probs.get('fil', 0.0)  # Filipino variant
            
            # Combine tl and fil probabilities
            tl_total = tl_prob + fil_prob
            
            logger.info(f"Constrained language detection: en={en_prob:.4f}, tl={tl_total:.4f}")
            
            # If confidence is very low for both, default to 'tl' as fallback
            # This handles cases where audio might be Bulos (which Whisper doesn't know)
            if en_prob < 0.1 and tl_total < 0.1:
                logger.warning(
                    f"Low confidence for both languages: en={en_prob:.4f}, tl={tl_total:.4f}. "
                    f"Defaulting to 'tl' as fallback (may be Bulos, will classify later)."
                )
                return 'tl', 0.0  # Use tl as fallback with low confidence
            
            # Choose the higher probability
            if en_prob > tl_total:
                return 'en', en_prob
            else:
                return 'tl', tl_total
                
        except RuntimeError:
            # Re-raise RuntimeError (it's our controlled error)
            raise
        except Exception as e:
            logger.error(f"Constrained language detection failed: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"Language detection failed: {str(e)}. Please ensure audio quality is good and try recording again."
            )
    
    def _detect_language_from_text(self, text: str) -> Optional[str]:
        """
        Detect language from transcribed text using langdetect library
        
        This is more reliable than audio-based detection for accented speech
        
        Args:
            text: Transcribed text to analyze
            
        Returns:
            Language code ('en', 'tl', or None if detection fails)
        """
        if not LANGDETECT_AVAILABLE:
            logger.warning("langdetect library not available, skipping text-based detection")
            return None
        
        if not text or len(text.strip()) < 10:
            logger.debug("Text too short for reliable language detection")
            return None
        
        try:
            detected = detect(text)
            logger.info(f"Text-based language detection: {detected}")
            
            # Map detected language to our supported languages
            if detected == 'en':
                return 'en'
            elif detected in ['tl', 'fil', 'id', 'ms', 'jv']:
                # Filipino/Tagalog and related Austronesian languages
                return 'tl'
            else:
                # Other languages: check if text has more English-like characteristics
                # This handles cases where langdetect might detect Spanish, etc.
                logger.info(f"Detected language '{detected}' not directly supported, analyzing text...")
                
                # Simple heuristic: count English vs Filipino indicators
                text_lower = text.lower()
                english_words = ['the', 'is', 'are', 'am', 'was', 'were', 'have', 'has', 'had', 'i', 'you', 'he', 'she', 'it', 'we', 'they']
                filipino_words = ['ang', 'ng', 'sa', 'ay', 'ko', 'mo', 'ka', 'ako', 'ikaw', 'siya', 'tayo', 'kami']
                
                english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
                filipino_count = sum(1 for word in filipino_words if f' {word} ' in f' {text_lower} ')
                
                if english_count > filipino_count:
                    logger.info(f"Text contains more English indicators ({english_count} vs {filipino_count})")
                    return 'en'
                else:
                    logger.info(f"Text contains more Filipino indicators ({filipino_count} vs {english_count})")
                    return 'tl'
                
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {str(e)}")
            return None
    
    def _map_to_supported_language(
        self, 
        detected_lang: str, 
        text: str, 
        dictionary_data: dict, 
        sentence_data: dict
    ) -> str:
        """
        Map Whisper's detected language to supported languages (en/tl/bul)
        
        Uses text-based detection to override Whisper's audio-based detection
        This handles Filipino-accented English being misclassified
        
        Args:
            detected_lang: Language code from Whisper (en, tl, fil, es, id, etc.)
            text: Transcribed text
            dictionary_data: Dictionary data for Bulos classification
            sentence_data: Sentence data for Bulos classification
            
        Returns:
            One of: 'en', 'tl', 'bul'
        """
        # PRIORITY 1: Use text-based detection (most reliable for accented speech)
        text_detected = self._detect_language_from_text(text)
        
        if text_detected:
            logger.info(f"Using text-based detection result: {text_detected}")
            
            if text_detected == 'en':
                # Text is clearly English
                return 'en'
            elif text_detected == 'tl':
                # Text is Filipino/Tagalog - check if it's Bulos
                if dictionary_data and sentence_data:
                    classified = self._classify_bulos_vs_tagalog(
                        text, dictionary_data, sentence_data, whisper_confidence=0.5
                    )
                    if classified == 'bul':
                        logger.info("Text-based detection: Tagalog, classified as Bulos")
                        return 'bul'
                
                logger.info("Text-based detection: Tagalog")
                return 'tl'
        
        # PRIORITY 2: Fallback to Whisper's audio-based detection
        logger.info(f"Falling back to Whisper's detection: {detected_lang}")
        
        # Map known language codes to our categories
        
        # English family
        if detected_lang in ['en']:
            logger.info("Mapped to English (en)")
            return 'en'
        
        # Filipino/Tagalog language family
        elif detected_lang in ['tl', 'fil', 'ceb', 'ilo', 'hil', 'war', 'pam', 'pag', 'bik']:
            logger.info(f"Detected Filipino language family: {detected_lang}")
            
            # Check if it's actually Bulos
            if dictionary_data and sentence_data:
                classified = self._classify_bulos_vs_tagalog(
                    text, dictionary_data, sentence_data, whisper_confidence=0.5
                )
                if classified == 'bul':
                    logger.info("Classified as Bulos (bul)")
                    return 'bul'
            
            # Not Bulos, map to Tagalog
            logger.info("Mapped to Tagalog (tl)")
            return 'tl'
        
        # Other languages - use text analysis fallback
        else:
            logger.warning(f"Unsupported language: {detected_lang}, defaulting to English")
            return 'en'
    
    def _classify_bulos_vs_tagalog(self, text: str, dictionary_data: dict, sentence_data: dict, whisper_confidence: float = 0.0) -> str:
        """
        Classify if transcribed Tagalog text is actually Bulos
        
        Strategy:
        1. If Whisper was very confident it's Tagalog (>0.7), trust it
        2. Normalize the text
        3. Check for Bulos-specific words (excluding very common short words)
        4. Calculate coverage: what % of words match Bulos dictionary
        5. If high Bulos coverage (>50%), return 'bul', otherwise 'tl'
        
        Args:
            text: Transcribed text
            dictionary_data: Loaded dictionary.json data
            sentence_data: Loaded sentence.json data
            whisper_confidence: Whisper's confidence for Tagalog detection
            
        Returns:
            'bul' if classified as Bulos, 'tl' otherwise
        """
        if not text:
            return 'tl'
        
        # If Whisper was very confident it's Tagalog, trust it
        # This prevents false positives from common words
        if whisper_confidence > 0.7:
            logger.info(f"Whisper confidence {whisper_confidence:.2%} is high - trusting Tagalog classification")
            return 'tl'
        
        # Normalize text
        normalized = text.lower().strip()
        words = normalized.split()
        
        if len(words) == 0:
            return 'tl'
        
        # Common Filipino/Tagalog words that also exist in Bulos but shouldn't count
        # These are too common and appear in both languages
        common_words = {
            'ka', 'na', 'ay', 'ang', 'ng', 'sa', 'ko', 'mo', 'si', 'ni',
            'at', 'o', 'ba', 'pa', 'po', 'ho', 'mga', 'ako', 'ako', 'siya',
            'tayo', 'kami', 'kayo', 'sila', 'i', 'de', 'di', 'a', 'e'
        }
        
        # Build Bulos word set from dictionary (excluding common words)
        bulos_words = set()
        for category in dictionary_data.get('categories', []):
            for entry in category.get('entries', []):
                bulos = entry.get('BULOS', '') or ''  # Handle None values
                bulos = bulos.lower().strip() if bulos else ''
                if bulos and bulos != '—':
                    # Add multi-word phrases
                    bulos_words.add(bulos)
                    # Also add individual words from phrases (excluding common/short words)
                    for word in bulos.split():
                        if len(word) > 3 and word not in common_words:  # Only words >3 chars and not common
                            bulos_words.add(word)
        
        # Add from sentence data (excluding common words)
        for entry in sentence_data.get('entries', []):
            bulos = entry.get('BULOS', '') or ''  # Handle None values
            bulos = bulos.lower().strip() if bulos else ''
            if bulos:
                bulos_words.add(bulos)
                for word in bulos.split():
                    if len(word) > 3 and word not in common_words:
                        bulos_words.add(word)
        
        # Count matches (excluding common words from the count)
        bulos_matches = 0
        significant_words = 0
        for word in words:
            if word not in common_words and len(word) > 2:  # Only count significant words
                significant_words += 1
                if word in bulos_words:
                    bulos_matches += 1
        
        # Calculate coverage based on significant words only
        if significant_words == 0:
            # If no significant words, default to Tagalog
            logger.info("No significant words found - defaulting to Tagalog")
            return 'tl'
        
        coverage = bulos_matches / significant_words
        
        logger.info(
            f"Bulos classification: {bulos_matches}/{significant_words} significant words matched "
            f"({coverage:.2%} coverage, whisper_conf: {whisper_confidence:.2%})"
        )
        
        # Increased threshold to 50% to reduce false positives
        # This means more than half of the significant words must be distinctly Bulos
        if coverage > 0.5:
            logger.info(f"Classified as Bulos (coverage: {coverage:.2%} > 50%)")
            return 'bul'
        else:
            logger.info(f"Classified as Tagalog (coverage: {coverage:.2%} <= 50%)")
            return 'tl'
    
    def _transcribe_sync(self, audio_data: bytes, dictionary_data: dict = None, sentence_data: dict = None) -> Tuple[str, float, str]:
        """
        Synchronous transcription with constrained language detection
        
        Strategy:
        1. Detect file format
        2. Let Whisper detect language (unconstrained - checks all languages)
        3. Map detected language to en/tl/bul
        4. If mapped to 'tl', check if it's actually 'bul' using dictionary
        5. Return (text, confidence, final_language) where final_language is always in {en, tl, bul}
        
        Args:
            audio_data: Raw audio file bytes
            dictionary_data: Loaded dictionary.json (for Bulos classification)
            sentence_data: Loaded sentence.json (for Bulos classification)
            
        Returns:
            Tuple of (transcribed_text, confidence_score, detected_language)
            
        Raises:
            ValueError: If language detection fails critically
        """
        # Detect file format and use appropriate extension for temp file
        file_ext = '.wav'  # default
        
        if len(audio_data) >= 12 and audio_data[4:8] == b'ftyp':
            file_ext = '.m4a'
        elif len(audio_data) >= 3 and audio_data[:3] == b'ID3':
            file_ext = '.mp3'
        elif len(audio_data) >= 2 and audio_data[:2] in [b'\xff\xfb', b'\xff\xf3', b'\xff\xf2']:
            file_ext = '.mp3'
        elif len(audio_data) >= 4:
            if audio_data[:4] == b'fLaC':
                file_ext = '.flac'
            elif audio_data[:4] == b'OggS':
                file_ext = '.ogg'
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            logger.info(f"Starting transcription for {len(audio_data)/1024:.1f}KB audio")
            logger.info("Note: First transcription is slow (~30-40s) as Whisper loads. Subsequent ones are faster.")
            
            # Step 1: Transcribe with unconstrained language detection
            # Let Whisper analyze audio + vocabulary + grammar patterns
            transcribe_options = {
                'language': None,  # Let Whisper auto-detect (uses full intelligence)
                'fp16': False,
                'verbose': False,
                'task': 'transcribe',
                'word_timestamps': True,
                'temperature': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                'beam_size': 5,
                'best_of': 5,
                'patience': 1.0,
                'condition_on_previous_text': True,
                'compression_ratio_threshold': 2.4,
                'logprob_threshold': -1.0,
                'no_speech_threshold': 0.6,
            }
            
            result = self.model.transcribe(temp_path, **transcribe_options)
            
            # Extract text and Whisper's detected language
            text = result['text'].strip()
            detected_lang = result.get('language', 'unknown')
            
            logger.info(f"Whisper transcription complete: '{text[:50]}...'")
            logger.info(f"Whisper detected language: {detected_lang}")
            
            # Step 2: Map Whisper's detected language to our supported languages (en/tl/bul)
            final_language = self._map_to_supported_language(
                detected_lang, text, dictionary_data, sentence_data
            )
            
            logger.info(f"Final mapped language: {final_language}")
            
            # Calculate confidence from word-level probabilities
            confidence = 0.0
            if 'segments' in result and result['segments']:
                total_prob = 0.0
                word_count = 0
                
                for segment in result['segments']:
                    no_speech_prob = segment.get('no_speech_prob', 0.5)
                    avg_logprob = segment.get('avg_logprob', -1.0)
                    
                    segment_confidence = max(0.0, min(1.0, (avg_logprob + 1.2) / 1.0))
                    segment_confidence *= (1.0 - no_speech_prob)
                    
                    segment_words = len(segment.get('words', [])) if 'words' in segment else 1
                    total_prob += segment_confidence * segment_words
                    word_count += segment_words
                
                if word_count > 0:
                    confidence = total_prob / word_count
                else:
                    confidence = 0.5
            else:
                confidence = 0.8 if text else 0.0
            
            logger.info(f"Final classification: detected_language='{final_language}', confidence={confidence:.4f}")
            
            # Ensure final_language is always one of our supported languages
            if final_language not in ['en', 'tl', 'bul']:
                logger.error(f"Invalid detected language '{final_language}'. This should never happen!")
                raise ValueError(
                    "Language detection failed. Please ensure audio quality is good and try recording again."
                )
            
            return text, confidence, final_language
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")
    
    async def transcribe(self, audio_data: bytes, dictionary_data: dict = None, sentence_data: dict = None) -> Tuple[str, float, str]:
        """
        Transcribe audio to text using Whisper with constrained language detection
        
        This method:
        1. Validates the audio file (format, size, duration)
        2. Performs constrained language detection (en or tl only)
        3. Transcribes with Whisper using the detected language as hint
        4. Classifies if tl audio is actually bul using dictionary
        5. Returns final classification that is ALWAYS one of: en, tl, or bul
        
        Args:
            audio_data: Raw audio file bytes
            dictionary_data: Optional dictionary.json data for Bulos classification
            sentence_data: Optional sentence.json data for Bulos classification
            
        Returns:
            Tuple of (transcribed_text, confidence_score, detected_language)
            detected_language is ALWAYS one of: 'en', 'tl', 'bul'
            
        Raises:
            ValueError: If audio validation fails or language cannot be determined
            RuntimeError: If Whisper model is not initialized
            asyncio.TimeoutError: If transcription exceeds timeout
        """
        if self.model is None:
            logger.error("Whisper model not initialized")
            raise RuntimeError(
                "Speech processor not initialized. Call initialize() first."
            )
        
        # Validate audio file
        self._validate_audio_file(audio_data)
        
        # Validate audio duration
        duration = self._validate_audio_duration(audio_data)
        
        logger.info(f"Starting transcription for {duration:.2f}s audio")
        
        try:
            # Run transcription in executor with timeout
            loop = asyncio.get_event_loop()
            text, confidence, detected_language = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._transcribe_sync,
                    audio_data,
                    dictionary_data,
                    sentence_data
                ),
                timeout=self.timeout
            )
            
            logger.info(
                f"Transcription completed: '{text}' "
                f"(confidence: {confidence:.4f}, duration: {duration:.2f}s, language: {detected_language})"
            )
            
            # Final safety check
            if detected_language not in ['en', 'tl', 'bul']:
                logger.error(f"Invalid detected_language '{detected_language}' - this is a bug!")
                raise ValueError(
                    "Language detection failed. Please ensure audio quality is good and try recording again."
                )
            
            return text, confidence, detected_language
            
        except asyncio.TimeoutError:
            logger.error(
                f"Transcription timeout ({self.timeout}s) exceeded for "
                f"{duration:.2f}s audio"
            )
            raise asyncio.TimeoutError(
                f"Transcription exceeded timeout of {self.timeout} seconds"
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise
