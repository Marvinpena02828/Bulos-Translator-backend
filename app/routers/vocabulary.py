"""Vocabulary management API endpoints"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from models.schemas import VocabularyCreate
from services.vocabulary import VocabularyService
from services.database import DatabaseManager
from utils.device_id import get_device_id
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/vocabulary",
    tags=["Vocabulary"]
)


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance from main app"""
    from app import main
    if main._db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    return main._db_manager


def get_vocabulary_service(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> VocabularyService:
    """Get vocabulary service instance"""
    return VocabularyService(db_manager)


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create vocabulary item",
    description="Create a new vocabulary item for the device"
)
async def create_vocabulary(
    vocab: VocabularyCreate,
    device_id: str = Depends(get_device_id),
    vocab_service: VocabularyService = Depends(get_vocabulary_service)
):
    """
    Create a new vocabulary item
    
    Requires X-Device-ID header
    
    - **word**: The word to store
    - **translation**: Translation of the word
    - **source_language**: Source language code (bul, en, tl)
    - **target_language**: Target language code (bul, en, tl)
    - **notes**: Optional notes about the word
    
    Returns the created vocabulary ID
    """
    try:
        result = await vocab_service.create_vocabulary(device_id, vocab)
        return result
    
    except ValueError as e:
        # Duplicate vocabulary item
        logger.warning(f"Duplicate vocabulary for device {device_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Error creating vocabulary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create vocabulary item"
        )


@router.get(
    "/",
    summary="Get vocabulary items",
    description="Retrieve vocabulary items with optional filtering and pagination"
)
async def get_vocabulary(
    source_language: Optional[str] = Query(
        None,
        description="Filter by source language (bul, en, tl)"
    ),
    target_language: Optional[str] = Query(
        None,
        description="Filter by target language (bul, en, tl)"
    ),
    page: int = Query(
        0,
        ge=0,
        description="Page number (0-indexed)"
    ),
    page_size: int = Query(
        20,
        ge=1,
        le=100,
        description="Number of items per page (max 100)"
    ),
    device_id: str = Depends(get_device_id),
    vocab_service: VocabularyService = Depends(get_vocabulary_service)
):
    """
    Get vocabulary items for the device
    
    Requires X-Device-ID header
    
    Supports filtering by language pair and pagination
    
    Returns:
    - **items**: List of vocabulary items
    - **total**: Total number of items matching the filter
    - **page**: Current page number
    - **page_size**: Items per page
    - **total_pages**: Total number of pages
    """
    try:
        result = await vocab_service.get_vocabulary(
            device_id=device_id,
            source_language=source_language,
            target_language=target_language,
            page=page,
            page_size=page_size
        )
        return result
    
    except Exception as e:
        logger.error(f"Error retrieving vocabulary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve vocabulary items"
        )


@router.put(
    "/{vocab_id}",
    summary="Update vocabulary item",
    description="Update an existing vocabulary item"
)
async def update_vocabulary(
    vocab_id: str,
    vocab: VocabularyCreate,
    device_id: str = Depends(get_device_id),
    vocab_service: VocabularyService = Depends(get_vocabulary_service)
):
    """
    Update an existing vocabulary item
    
    Requires X-Device-ID header
    
    Only the device owner can update their vocabulary items
    
    - **vocab_id**: ID of the vocabulary item to update
    - **word**: Updated word
    - **translation**: Updated translation
    - **source_language**: Updated source language
    - **target_language**: Updated target language
    - **notes**: Updated notes
    """
    try:
        result = await vocab_service.update_vocabulary(
            vocab_id=vocab_id,
            device_id=device_id,
            update_data=vocab
        )
        return result
    
    except ValueError as e:
        # Not found or unauthorized
        logger.warning(
            f"Update failed for vocab {vocab_id}, device {device_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Error updating vocabulary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update vocabulary item"
        )


@router.delete(
    "/{vocab_id}",
    summary="Delete vocabulary item",
    description="Delete an existing vocabulary item"
)
async def delete_vocabulary(
    vocab_id: str,
    device_id: str = Depends(get_device_id),
    vocab_service: VocabularyService = Depends(get_vocabulary_service)
):
    """
    Delete a vocabulary item
    
    Requires X-Device-ID header
    
    Only the device owner can delete their vocabulary items
    
    - **vocab_id**: ID of the vocabulary item to delete
    """
    try:
        result = await vocab_service.delete_vocabulary(
            vocab_id=vocab_id,
            device_id=device_id
        )
        return result
    
    except ValueError as e:
        # Not found or unauthorized
        logger.warning(
            f"Delete failed for vocab {vocab_id}, device {device_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Error deleting vocabulary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete vocabulary item"
        )
