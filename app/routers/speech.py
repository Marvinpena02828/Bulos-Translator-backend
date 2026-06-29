"""Speech processing API endpoints"""
import asyncio
import time
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status

from models.schemas import SpeechProcessResponse, SUPPORTED_LANGUAGES
from services.database import DatabaseManager
from services.auth import get_current_user
from config import settings
from utils.logging_config import get_logger

# Lazy import for SpeechProcessor to avoid importing DeepSpeech at module load time
if TYPE_CHECKING:
    from services.speech import SpeechProcessor

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/speech",
    tags=["Speech Processing"]
)

# Database manager singleton (will be initialized on startup)
_db_manager: DatabaseManager = None
_speech_processor: "SpeechProcessor" = None


def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    global _db_manager
    if _db_manager is None:
        from services.database import initialize_db_manager
        _db_manager = initialize_db_manager(
            connection_string=settings.mongodb_url,
            database_name=settings.mongodb_database,
            max_pool_size=settings.mongodb_max_pool_size,
            min_pool_size=settings.mongodb_min_pool_size
        )
    return _db_manager


async def get_speech_processor() -> "SpeechProcessor":
    """Get speech processor instance (singleton pattern)"""
    # Import here to avoid DeepSpeech import at module load time
    from services.speech import SpeechProcessor
    
    global _speech_processor
    if _speech_processor is None:
        _speech_processor = SpeechProcessor()
        # Initialize model on first access
        try:
            await _speech_processor.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize speech processor: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Speech processor initialization failed: {str(e)}"
            )
    return _speech_processor


@router.post(
    "/transcribe",
    response_model=SpeechProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe audio to text",
    description="Transcribe speech from WAV audio file to text using Mozilla DeepSpeech"
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="WAV audio file to transcribe (max 10MB, max 60s duration)"),
    language: str = Form(..., description="Language of the audio (bul, en, tl)"),
    user_id: str = Depends(get_current_user),
    speech_processor: "SpeechProcessor" = Depends(get_speech_processor)
):
    """
    Transcribe speech from audio file to text
    
    - **audio**: WAV audio file (16kHz, 16-bit, mono, max 10MB, max 60s)
    - **language**: Language code of the audio (bul, en, tl)
    
    Audio Requirements:
    - Format: WAV (RIFF/WAVE)
    - Sample rate: 16000 Hz (16 kHz)
    - Bit depth: 16-bit
    - Channels: 1 (mono)
    - Max file size: 10 MB
    - Max duration: 60 seconds
    
    Returns:
    - **transcribed_text**: The transcribed text from the audio
    - **confidence**: Transcription confidence score (0-1)
    - **processing_time**: Time taken to process the audio in seconds
    
    Raises:
    - **400**: Invalid audio format, file too large, or duration too long
    - **408**: Processing timeout (exceeds 10 seconds)
    - **503**: Speech processor not available
    """
    start_time = time.time()
    
    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(
            f"Invalid language from user {user_id}: {language}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {language}. Must be one of {SUPPORTED_LANGUAGES}"
        )
    
    # Validate file format by content type and filename
    if not audio.filename.lower().endswith('.wav'):
        logger.warning(
            f"Invalid file format from user {user_id}: {audio.filename}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file must be in WAV format (.wav extension required)"
        )
    
    try:
        logger.info(
            f"Transcription request from user {user_id}: "
            f"file={audio.filename}, language={language}"
        )
        
        # Read audio file content
        audio_data = await audio.read()
        
        logger.debug(f"Read audio file: {len(audio_data)} bytes")
        
        # Transcribe audio using speech processor
        # The processor handles all validation (format, size, duration)
        transcribed_text, confidence = await speech_processor.transcribe(audio_data)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Transcription completed for user {user_id}: "
            f"text='{transcribed_text[:50]}...' "
            f"(confidence: {confidence:.4f}, time: {processing_time:.2f}s)"
        )
        
        return SpeechProcessResponse(
            transcribed_text=transcribed_text,
            confidence=confidence,
            processing_time=round(processing_time, 3)
        )
    
    except ValueError as e:
        # Validation errors from speech processor
        logger.warning(
            f"Invalid audio from user {user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except asyncio.TimeoutError:
        # Processing timeout
        processing_time = time.time() - start_time
        logger.error(
            f"Transcription timeout for user {user_id} after {processing_time:.2f}s"
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Transcription timed out after {settings.speech_processing_timeout_seconds} seconds"
        )
    
    except RuntimeError as e:
        # Speech processor not initialized or processing error
        logger.error(
            f"Speech processing error for user {user_id}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Speech processing service error: {str(e)}"
        )
    
    except Exception as e:
        # General error
        processing_time = time.time() - start_time
        logger.error(
            f"Transcription error for user {user_id} after {processing_time:.2f}s: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process audio transcription request"
        )
