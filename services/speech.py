"""Speech processing service with DeepSpeech integration"""
import asyncio
import wave
from pathlib import Path
from typing import Tuple, Optional
import io

try:
    import deepspeech
    DEEPSPEECH_AVAILABLE = True
except ImportError:
    DEEPSPEECH_AVAILABLE = False

from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SpeechProcessor:
    """Service for processing speech audio using Mozilla DeepSpeech"""
    
    def __init__(self):
        """
        Initialize speech processor
        
        DeepSpeech model and scorer paths are loaded from config
        """
        self.model_path = settings.deepspeech_model_path
        self.scorer_path = settings.deepspeech_scorer_path
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
        Load DeepSpeech model and scorer
        
        This method loads the pre-trained DeepSpeech model and language scorer
        from the configured paths. Must be called before transcribing audio.
        
        Raises:
            FileNotFoundError: If model or scorer files are not found
            RuntimeError: If DeepSpeech is not installed or model loading fails
        """
        logger.info("Initializing DeepSpeech model...")
        
        if not DEEPSPEECH_AVAILABLE:
            logger.error("DeepSpeech library is not installed")
            raise RuntimeError(
                "DeepSpeech is not installed. Install it with: pip install deepspeech"
            )
        
        # Check if model files exist
        model_path = Path(self.model_path)
        scorer_path = Path(self.scorer_path)
        
        if not model_path.exists():
            logger.error(f"DeepSpeech model not found at {model_path}")
            raise FileNotFoundError(f"DeepSpeech model not found: {model_path}")
        
        if not scorer_path.exists():
            logger.error(f"DeepSpeech scorer not found at {scorer_path}")
            raise FileNotFoundError(f"DeepSpeech scorer not found: {scorer_path}")
        
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_model_sync,
                str(model_path),
                str(scorer_path)
            )
            
            logger.info("DeepSpeech model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load DeepSpeech model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load DeepSpeech model: {str(e)}")
    
    def _load_model_sync(self, model_path: str, scorer_path: str):
        """
        Synchronous model loading (runs in thread pool)
        
        Args:
            model_path: Path to DeepSpeech model file (.pbmm)
            scorer_path: Path to DeepSpeech scorer file (.scorer)
            
        Returns:
            Loaded DeepSpeech model instance
        """
        logger.info(f"Loading DeepSpeech model from {model_path}")
        model = deepspeech.Model(model_path)
        
        logger.info(f"Enabling external scorer from {scorer_path}")
        model.enableExternalScorer(scorer_path)
        
        return model
    
    def _validate_audio_file(self, audio_data: bytes) -> None:
        """
        Validate audio file size and format
        
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
        
        # Validate WAV format by checking header
        if len(audio_data) < 44:  # Minimum WAV header size
            logger.warning("Audio file too small to be valid WAV")
            raise ValueError("Audio file is too small to be a valid WAV file")
        
        # Check for RIFF header
        if audio_data[:4] != b'RIFF':
            logger.warning("Audio file missing RIFF header")
            raise ValueError("Invalid audio format: must be WAV file (RIFF header missing)")
        
        # Check for WAVE identifier
        if audio_data[8:12] != b'WAVE':
            logger.warning("Audio file missing WAVE identifier")
            raise ValueError("Invalid audio format: must be WAV file (WAVE identifier missing)")
        
        logger.debug(f"Audio validation passed: {file_size} bytes")
    
    def _validate_audio_duration(self, audio_data: bytes) -> float:
        """
        Validate audio duration and return duration in seconds
        
        Args:
            audio_data: Raw audio file bytes
            
        Returns:
            Duration in seconds
            
        Raises:
            ValueError: If audio duration exceeds maximum allowed duration
        """
        try:
            # Parse WAV file to get duration
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
            
        except wave.Error as e:
            logger.error(f"Failed to parse WAV file: {str(e)}")
            raise ValueError(f"Invalid WAV file format: {str(e)}")
    
    def _prepare_audio_for_deepspeech(self, audio_data: bytes) -> Tuple[bytes, int]:
        """
        Prepare audio data for DeepSpeech processing
        
        DeepSpeech requires 16kHz 16-bit mono audio. This method extracts
        the raw audio samples from the WAV file.
        
        Args:
            audio_data: Raw WAV file bytes
            
        Returns:
            Tuple of (raw_audio_samples, sample_rate)
            
        Raises:
            ValueError: If audio format is incompatible with DeepSpeech
        """
        try:
            audio_io = io.BytesIO(audio_data)
            with wave.open(audio_io, 'rb') as wav_file:
                # Get audio parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                
                # DeepSpeech expects 16kHz 16-bit mono
                if channels != 1:
                    logger.warning(f"Audio has {channels} channels, expected mono (1)")
                    raise ValueError(
                        f"Audio must be mono (1 channel), received {channels} channels"
                    )
                
                if sample_width != 2:  # 2 bytes = 16 bits
                    logger.warning(f"Audio is {sample_width*8}-bit, expected 16-bit")
                    raise ValueError(
                        f"Audio must be 16-bit, received {sample_width*8}-bit"
                    )
                
                if sample_rate != 16000:
                    logger.warning(f"Audio sample rate is {sample_rate}Hz, expected 16000Hz")
                    raise ValueError(
                        f"Audio must be 16kHz sample rate, received {sample_rate}Hz"
                    )
                
                # Extract raw audio samples
                raw_audio = wav_file.readframes(wav_file.getnframes())
                
                logger.debug(
                    f"Audio prepared: {len(raw_audio)} bytes, "
                    f"{sample_rate}Hz, {channels}ch, {sample_width*8}bit"
                )
                
                return raw_audio, sample_rate
                
        except wave.Error as e:
            logger.error(f"Failed to prepare audio: {str(e)}")
            raise ValueError(f"Invalid WAV file: {str(e)}")
    
    def _transcribe_sync(self, audio_data: bytes) -> Tuple[str, float]:
        """
        Synchronous transcription (runs in thread pool)
        
        Args:
            audio_data: Raw WAV audio file bytes
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        # Prepare audio for DeepSpeech
        raw_audio, sample_rate = self._prepare_audio_for_deepspeech(audio_data)
        
        # Convert bytes to numpy array (int16)
        import numpy as np
        audio_array = np.frombuffer(raw_audio, dtype=np.int16)
        
        # Transcribe using DeepSpeech
        logger.debug("Running DeepSpeech transcription...")
        
        # Get metadata for confidence score
        metadata = self.model.sttWithMetadata(audio_array)
        
        # Extract best transcription
        if metadata.transcripts:
            transcript = metadata.transcripts[0]
            text = ''.join(token.text for token in transcript.tokens)
            confidence = transcript.confidence
            
            logger.debug(f"Transcription: '{text}' (confidence: {confidence:.4f})")
            return text, confidence
        else:
            logger.warning("DeepSpeech returned no transcription")
            return "", 0.0
    
    async def transcribe(self, audio_data: bytes) -> Tuple[str, float]:
        """
        Transcribe audio to text using DeepSpeech
        
        This method validates the audio file (format, size, duration), processes
        it through DeepSpeech with timeout enforcement, and returns the transcribed
        text with a confidence score.
        
        Args:
            audio_data: Raw WAV audio file bytes
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
            
        Raises:
            ValueError: If audio validation fails (format, size, or duration)
            RuntimeError: If DeepSpeech model is not initialized
            asyncio.TimeoutError: If transcription exceeds timeout
        """
        if self.model is None:
            logger.error("DeepSpeech model not initialized")
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
            text, confidence = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._transcribe_sync,
                    audio_data
                ),
                timeout=self.timeout
            )
            
            logger.info(
                f"Transcription completed: '{text}' "
                f"(confidence: {confidence:.4f}, duration: {duration:.2f}s)"
            )
            
            return text, confidence
            
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
