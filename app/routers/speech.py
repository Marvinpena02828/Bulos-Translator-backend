"""Speech processing API endpoints"""
import asyncio
import time
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status

from models.schemas import SpeechProcessResponse, SpeechTranslateResponse, SUPPORTED_LANGUAGES
from services.database import DatabaseManager
from utils.device_id import get_device_id
from config import settings
from utils.logging_config import get_logger

# Lazy import for SpeechProcessor to avoid importing Whisper at module load time
if TYPE_CHECKING:
    from services.speech import SpeechProcessor

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/speech",
    tags=["Speech Processing"]
)

# Speech processor singleton
_speech_processor: "SpeechProcessor" = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance from main app"""
    from app import main
    if main._db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    return main._db_manager


async def get_speech_processor() -> "SpeechProcessor":
    """Get speech processor instance (singleton pattern)"""
    # Import here to avoid Whisper import at module load time
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
    description="Transcribe speech from audio file to text using OpenAI Whisper (supports WAV, MP3, M4A, FLAC, OGG)"
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe (WAV/MP3/M4A/FLAC/OGG, max 10MB, max 60s duration)"),
    language: str = Form(..., description="Language of the audio (bul, en, tl)"),
    device_id: str = Depends(get_device_id),
    speech_processor: "SpeechProcessor" = Depends(get_speech_processor)
):
    """
    Transcribe speech from audio file to text
    
    - **audio**: Audio file (WAV, MP3, M4A, FLAC, OGG supported, max 10MB, max 60s)
    - **language**: Language code of the audio (bul, en, tl)
    
    Audio Requirements:
    - Format: WAV, MP3, M4A (iPhone/Apple), FLAC, OGG, WEBM
    - Max file size: 10 MB
    - Max duration: 60 seconds (validated for WAV only, enforced via file size for others)
    - Sample rate: Any (Whisper auto-converts via FFmpeg)
    - Channels: Mono or Stereo (Whisper handles both)
    - Codec: Any (Whisper uses FFmpeg for decoding)
    
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
            f"Invalid language from device {device_id}: {language}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {language}. Must be one of {SUPPORTED_LANGUAGES}"
        )
    
    # Validate file format by extension
    # Accept common audio formats supported by Whisper/FFmpeg
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.mp4']
    file_ext = audio.filename.lower().split('.')[-1] if audio.filename else ''
    
    if not any(audio.filename.lower().endswith(ext) for ext in allowed_extensions):
        logger.warning(
            f"Invalid file format from device {device_id}: {audio.filename}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file must be in supported format: {', '.join(allowed_extensions)}"
        )
    
    try:
        logger.info(
            f"Transcription request from device {device_id}: "
            f"file={audio.filename}, language={language}"
        )
        
        # Read audio file content
        audio_data = await audio.read()
        
        logger.debug(f"Read audio file: {len(audio_data)} bytes")
        
        # Transcribe audio using speech processor
        # The processor handles all validation (format, size, duration)
        # Pass language hint to Whisper for better accuracy
        transcribed_text, confidence, detected_language = await speech_processor.transcribe(audio_data, language)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Transcription completed for device {device_id}: "
            f"text='{transcribed_text[:50]}...' "
            f"(confidence: {confidence:.4f}, time: {processing_time:.2f}s, detected: {detected_language})"
        )
        
        return SpeechProcessResponse(
            transcribed_text=transcribed_text,
            confidence=confidence,
            processing_time=round(processing_time, 3)
        )
    
    except ValueError as e:
        # Validation errors from speech processor
        logger.warning(
            f"Invalid audio from device {device_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except asyncio.TimeoutError:
        # Processing timeout
        processing_time = time.time() - start_time
        logger.error(
            f"Transcription timeout for device {device_id} after {processing_time:.2f}s"
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Transcription timed out after {settings.speech_processing_timeout_seconds} seconds"
        )
    
    except RuntimeError as e:
        # Speech processor not initialized or processing error
        logger.error(
            f"Speech processing error for device {device_id}: {str(e)}",
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
            f"Transcription error for device {device_id} after {processing_time:.2f}s: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process audio transcription request"
        )


@router.post(
    "/transcribe-and-translate",
    response_model=SpeechTranslateResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe audio and translate to target language",
    description="Transcribe speech from audio file and automatically translate to target language"
)
async def transcribe_and_translate_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe (WAV/MP3/M4A/FLAC/OGG, max 10MB, max 60s duration)"),
    target_language: str = Form(..., description="Target language for translation (bul, en, tl)"),
    device_id: str = Depends(get_device_id),
    speech_processor: "SpeechProcessor" = Depends(get_speech_processor),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Transcribe speech from audio file and translate to target language
    
    This endpoint combines speech recognition and translation in a single request:
    1. Transcribes the audio (auto-detects source language)
    2. Translates the transcribed text to the target language
    
    - **audio**: Audio file (WAV, MP3, M4A, FLAC, OGG supported, max 10MB, max 60s)
    - **target_language**: Target language code (bul, en, tl)
    
    Audio Requirements:
    - Format: WAV, MP3, M4A (iPhone/Apple), FLAC, OGG, WEBM
    - Max file size: 10 MB
    - Max duration: 60 seconds (validated for WAV only, enforced via file size for others)
    - Sample rate: Any (Whisper auto-converts via FFmpeg)
    - Channels: Mono or Stereo (Whisper handles both)
    - Codec: Any (Whisper uses FFmpeg for decoding)
    
    Returns:
    - **transcribed_text**: The transcribed text from the audio (in detected language)
    - **translated_text**: The translated text (in target language)
    - **detected_language**: Detected source language from audio
    - **target_language**: Target language for translation
    - **transcription_confidence**: Transcription confidence score (0-1)
    - **translation_confidence**: Translation confidence score (0-1)
    - **processing_time**: Total time taken to process in seconds
    
    Raises:
    - **400**: Invalid audio format, file too large, or duration too long
    - **408**: Processing timeout
    - **503**: Speech processor or translation service not available
    """
    start_time = time.time()
    
    # Validate target language
    if target_language not in SUPPORTED_LANGUAGES:
        logger.warning(
            f"Invalid target language from device {device_id}: {target_language}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported target language: {target_language}. Must be one of {SUPPORTED_LANGUAGES}"
        )
    
    # Validate file format by extension
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.mp4']
    
    if not any(audio.filename.lower().endswith(ext) for ext in allowed_extensions):
        logger.warning(
            f"Invalid file format from device {device_id}: {audio.filename}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file must be in supported format: {', '.join(allowed_extensions)}"
        )
    
    try:
        logger.info(
            f"Transcribe-and-translate request from device {device_id}: "
            f"file={audio.filename}, target_language={target_language}"
        )
        
        # Read audio file content
        audio_data = await audio.read()
        
        logger.debug(f"Read audio file: {len(audio_data)} bytes")
        
        # Step 1: Transcribe audio (auto-detect source language)
        transcribed_text, transcription_confidence, detected_language = await speech_processor.transcribe(audio_data, language=None)
        
        logger.info(
            f"Transcription complete: text='{transcribed_text[:50]}...' "
            f"(detected: {detected_language}, confidence: {transcription_confidence:.4f})"
        )
        
        # Map Whisper language codes to our language codes
        whisper_to_our_lang = {
            'en': 'en',
            'tl': 'tl',
            'fil': 'tl',  # Filipino → Tagalog
            'tagalog': 'tl',
        }
        
        source_language = whisper_to_our_lang.get(detected_language, 'en')
        
        # Step 2: Check if translation is needed
        if source_language == target_language:
            # No translation needed - source and target are the same
            logger.info(f"No translation needed: source={source_language}, target={target_language}")
            processing_time = time.time() - start_time
            
            return SpeechTranslateResponse(
                transcribed_text=transcribed_text,
                translated_text=transcribed_text,
                detected_language=source_language,
                target_language=target_language,
                transcription_confidence=transcription_confidence,
                translation_confidence=1.0,  # Perfect confidence when no translation needed
                processing_time=round(processing_time, 3)
            )
        
        # Step 3: Translate the transcribed text
        # Import translation service
        from app import main
        
        if main._translation_service is None:
            logger.error("Translation service not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Translation service not available"
            )
        
        translation_result = await main._translation_service.translate(
            text=transcribed_text,
            source_language=source_language,
            target_language=target_language,
            user_id=device_id  # Pass device_id as user_id for backward compatibility
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Transcribe-and-translate completed for device {device_id}: "
            f"'{transcribed_text[:30]}...' ({detected_language}) → '{translation_result['translated_text'][:30]}...' ({target_language}) "
            f"(time: {processing_time:.2f}s)"
        )
        
        return SpeechTranslateResponse(
            transcribed_text=transcribed_text,
            translated_text=translation_result['translated_text'],
            detected_language=source_language,
            target_language=target_language,
            transcription_confidence=transcription_confidence,
            translation_confidence=translation_result.get('confidence'),
            processing_time=round(processing_time, 3)
        )
    
    except ValueError as e:
        # Validation errors from speech processor
        logger.warning(
            f"Invalid audio from device {device_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except asyncio.TimeoutError:
        # Processing timeout
        processing_time = time.time() - start_time
        logger.error(
            f"Transcribe-and-translate timeout for device {device_id} after {processing_time:.2f}s"
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Processing timed out after {settings.speech_processing_timeout_seconds} seconds"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # General error
        processing_time = time.time() - start_time
        logger.error(
            f"Transcribe-and-translate error for device {device_id} after {processing_time:.2f}s: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process audio transcription and translation request"
        )
