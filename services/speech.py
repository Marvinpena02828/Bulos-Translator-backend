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

from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _find_ffmpeg():
    """
    Find FFmpeg executable
    
    Search order:
    1. User-specified path in .env (FFMPEG_PATH)
    2. System PATH
    3. Common installation paths
    """
    # Check config first
    if settings.ffmpeg_path and os.path.exists(settings.ffmpeg_path):
        logger.info(f"Using FFmpeg from config: {settings.ffmpeg_path}")
        return settings.ffmpeg_path
    
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
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device for inference (cpu or cuda)
            
        Returns:
            Loaded Whisper model instance
        """
        logger.info(f"Loading Whisper model '{model_name}' on device '{device}'")
        model = whisper.load_model(model_name, device=device)
        logger.info(f"Whisper model loaded: {model_name}")
        return model
    
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
    
    def _transcribe_sync(self, audio_data: bytes, language: str = None) -> Tuple[str, float, str]:
        """
        Synchronous transcription (runs in thread pool)
        
        Args:
            audio_data: Raw audio file bytes (any format supported by FFmpeg)
            language: Optional language code (bul, en, tl)
            
        Returns:
            Tuple of (transcribed_text, confidence_score, detected_language)
        """
        # Map our language codes to Whisper language codes
        # Whisper uses ISO 639-1 codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
        whisper_language_map = {
            'bul': None,  # Bulos not in Whisper - let it auto-detect
            'en': 'en',   # English
            'tl': 'tl',   # Tagalog (Filipino)
            'fil': 'tl',  # Filipino → Tagalog
        }
        
        whisper_lang = whisper_language_map.get(language, None) if language else None
        
        # Detect file format and use appropriate extension for temp file
        # This helps FFmpeg identify the format correctly
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
        
        # Save audio to temporary file (Whisper requires file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            logger.info(f"Transcribing {file_ext} file (language: {language} → whisper: {whisper_lang})...")
            logger.info("Note: First transcription is slow (~30-40s) as Whisper loads. Subsequent ones are faster.")
            
            # Transcribe using Whisper
            # Whisper handles multiple audio formats and sample rates automatically via FFmpeg
            # Note: fp16=False is important for CPU inference
            # Note: First run is slow because Whisper loads models into memory
            transcribe_options = {
                'fp16': False,  # Use FP32 for CPU compatibility
                'verbose': False,  # Reduce logging
                'task': 'transcribe',  # We want transcription, not translation
            }
            
            # Only add language if we have a valid mapping
            # For Bulos or unknown languages, let Whisper auto-detect
            if whisper_lang:
                transcribe_options['language'] = whisper_lang
                logger.debug(f"Using language hint: {whisper_lang}")
            else:
                logger.debug("No language hint - Whisper will auto-detect")
            
            result = self.model.transcribe(temp_path, **transcribe_options)
            
            # Extract text and confidence
            text = result['text'].strip()
            
            # Get detected language from Whisper result
            detected_language = result.get('language', 'unknown')
            logger.info(f"Detected language: {detected_language}")
            
            # Whisper doesn't provide per-segment confidence like DeepSpeech
            # We'll use a proxy: if result has segments with avg_logprob, use that
            confidence = 0.0
            if 'segments' in result and result['segments']:
                # Average log probability across segments (convert to 0-1 scale)
                avg_logprob = sum(s.get('avg_logprob', -1.0) for s in result['segments']) / len(result['segments'])
                # Convert log probability to approximate confidence (0-1)
                # Typical range: -0.5 (confident) to -1.5 (less confident)
                confidence = max(0.0, min(1.0, (avg_logprob + 1.5) / 1.0))
            else:
                # Default confidence if no segments
                confidence = 0.9 if text else 0.0
            
            logger.info(f"Transcription complete: '{text[:50]}...' (confidence: {confidence:.4f})")
            return text, confidence, detected_language
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")
    
    async def transcribe(self, audio_data: bytes, language: str = None) -> Tuple[str, float, str]:
        """
        Transcribe audio to text using Whisper
        
        This method validates the audio file (format, size, duration), processes
        it through Whisper with timeout enforcement, and returns the transcribed
        text with a confidence score and detected language.
        
        Whisper is more flexible than DeepSpeech:
        - Supports multiple audio formats (WAV, MP3, etc.)
        - Handles various sample rates automatically
        - Works with stereo and mono audio
        - Multilingual support (99 languages)
        
        Args:
            audio_data: Raw audio file bytes (WAV preferred)
            language: Optional language code for better accuracy (en, tl, etc.)
            
        Returns:
            Tuple of (transcribed_text, confidence_score, detected_language)
            
        Raises:
            ValueError: If audio validation fails (format, size, or duration)
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
                    language
                ),
                timeout=self.timeout
            )
            
            logger.info(
                f"Transcription completed: '{text}' "
                f"(confidence: {confidence:.4f}, duration: {duration:.2f}s, language: {detected_language})"
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
