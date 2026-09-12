"""Translation API endpoints"""
import asyncio
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, status

from models.schemas import TranslationRequest, TranslationResponse
from utils.device_id import get_device_id
from config import settings
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from services.translation import TranslationService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/translate",
    tags=["Translation"]
)

_translation_service: "TranslationService" = None


def get_translation_service() -> "TranslationService":
    from app import main
    if main._translation_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Translation service not available"
        )
    return main._translation_service


@router.post(
    "/",
    response_model=TranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate text",
    description="Translate text between supported language pairs (bul↔en, bul↔tl, en↔tl)"
)
async def translate_text(
    request: TranslationRequest,
    device_id: str = Depends(get_device_id),
    translation_service: "TranslationService" = Depends(get_translation_service),
):
    """
    Translate text from source language to target language.

    - **text**: Text to translate (1-500 characters)
    - **source_language**: Source language code (bul, en, tl)
    - **target_language**: Target language code (bul, en, tl)
    """
    try:
        logger.info(
            f"Translation request from device {device_id}: "
            f"{request.source_language}->{request.target_language}"
        )

        result = await translation_service.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            user_id=device_id,
        )

        return TranslationResponse(**result)

    except ValueError as e:
        logger.warning(f"Invalid translation request from device {device_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except asyncio.TimeoutError:
        logger.error(f"Translation timeout for device {device_id}")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Translation request timed out after {settings.translation_timeout_seconds} seconds",
        )

    except Exception as e:
        logger.error(f"Translation error for device {device_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete translation request",
        )
