"""Translation API endpoints"""
import asyncio
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, status

from models.schemas import TranslationRequest, TranslationResponse
from services.database import DatabaseManager
from services.auth import get_current_user
from config import settings
from utils.logging_config import get_logger

# Lazy import for TranslationService to avoid importing TensorFlow at module load time
if TYPE_CHECKING:
    from services.translation import TranslationService

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/translate",
    tags=["Translation"]
)

# Database manager singleton (will be initialized on startup)
_db_manager: DatabaseManager = None
_translation_service: "TranslationService" = None


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


async def get_translation_service(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> "TranslationService":
    """Get translation service instance (singleton pattern)"""
    # Import here to avoid TensorFlow import at module load time
    from services.translation import TranslationService
    
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService(db_manager)
        # Initialize models on first access
        await _translation_service.initialize()
    return _translation_service


@router.post(
    "/",
    response_model=TranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate text",
    description="Translate text between supported language pairs (bul↔en, bul↔tl, en↔tl)"
)
async def translate_text(
    request: TranslationRequest,
    user_id: str = Depends(get_current_user),
    translation_service: "TranslationService" = Depends(get_translation_service)
):
    """
    Translate text from source language to target language
    
    - **text**: Text to translate (1-500 characters)
    - **source_language**: Source language code (bul, en, tl)
    - **target_language**: Target language code (bul, en, tl)
    
    Supported language pairs:
    - bul ↔ en (Bulos ↔ English)
    - bul ↔ tl (Bulos ↔ Tagalog/Filipino)
    - en ↔ tl (English ↔ Tagalog/Filipino)
    
    Returns:
    - **original_text**: The original input text
    - **translated_text**: The translated text
    - **source_language**: Source language code
    - **target_language**: Target language code
    - **confidence**: Translation confidence score (if available)
    """
    try:
        logger.info(
            f"Translation request from user {user_id}: "
            f"{request.source_language}->{request.target_language}"
        )
        
        # Call translation service with user_id for history tracking
        result = await translation_service.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            user_id=user_id
        )
        
        return TranslationResponse(**result)
    
    except ValueError as e:
        # Unsupported language pair
        logger.warning(
            f"Invalid translation request from user {user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except asyncio.TimeoutError:
        # Translation timeout
        logger.error(
            f"Translation timeout for user {user_id}: "
            f"{request.source_language}->{request.target_language}"
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Translation request timed out after {settings.translation_timeout_seconds} seconds"
        )
    
    except Exception as e:
        # General error
        logger.error(
            f"Translation error for user {user_id}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete translation request"
        )
