"""Vocabulary management API endpoints"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from models.schemas import VocabularyCreate
from services.vocabulary import VocabularyService
from utils.device_id import get_device_id
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/vocabulary",
    tags=["Vocabulary"]
)

# Singleton service (in-memory store lives inside the module)
_vocab_service = VocabularyService()


def get_vocabulary_service() -> VocabularyService:
    return _vocab_service


@router.post("/", status_code=status.HTTP_201_CREATED,
             summary="Create vocabulary item")
async def create_vocabulary(
    vocab: VocabularyCreate,
    device_id: str = Depends(get_device_id),
    vocab_service: VocabularyService = Depends(get_vocabulary_service),
):
    try:
        return await vocab_service.create_vocabulary(device_id, vocab)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating vocabulary: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to create vocabulary item")


@router.get("/", summary="Get vocabulary items")
async def get_vocabulary(
    source_language: Optional[str] = Query(None),
    target_language: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    device_id: str = Depends(get_device_id),
    vocab_service: VocabularyService = Depends(get_vocabulary_service),
):
    try:
        return await vocab_service.get_vocabulary(
            device_id=device_id,
            source_language=source_language,
            target_language=target_language,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        logger.error(f"Error retrieving vocabulary: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to retrieve vocabulary items")


@router.put("/{vocab_id}", summary="Update vocabulary item")
async def update_vocabulary(
    vocab_id: str,
    vocab: VocabularyCreate,
    device_id: str = Depends(get_device_id),
    vocab_service: VocabularyService = Depends(get_vocabulary_service),
):
    try:
        return await vocab_service.update_vocabulary(vocab_id, device_id, vocab)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating vocabulary: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to update vocabulary item")


@router.delete("/{vocab_id}", summary="Delete vocabulary item")
async def delete_vocabulary(
    vocab_id: str,
    device_id: str = Depends(get_device_id),
    vocab_service: VocabularyService = Depends(get_vocabulary_service),
):
    try:
        return await vocab_service.delete_vocabulary(vocab_id, device_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting vocabulary: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to delete vocabulary item")
