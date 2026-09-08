"""Alphabet API endpoints for Dumaget Bulos alphabet with examples"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status

from models.schemas import (
    AlphabetEntry,
    ImportResult
)
from services.alphabet import AlphabetService
from services.database import DatabaseManager
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/alphabet",
    tags=["Alphabet"]
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


def get_alphabet_service(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> AlphabetService:
    """Get alphabet service instance"""
    return AlphabetService(db_manager)


def verify_admin_key(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    """
    Verify admin API key from request header
    
    Args:
        x_admin_key: Admin API key from X-Admin-Key header
        
    Raises:
        HTTPException: If admin key is missing or invalid
    """
    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required"
        )
    
    if x_admin_key != settings.admin_api_key:
        logger.warning("Invalid admin API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key"
        )
    
    return x_admin_key


@router.post(
    "/import",
    response_model=ImportResult,
    status_code=status.HTTP_200_OK,
    summary="Import alphabet data (Admin only)",
    description="Import alphabet.json into MongoDB. Requires admin authentication."
)
async def import_alphabet(
    force_reimport: bool = False,
    admin_key: str = Depends(verify_admin_key),
    alphabet_service: AlphabetService = Depends(get_alphabet_service)
):
    """
    Import alphabet from JSON file (Admin only)
    
    This endpoint triggers the import of alphabet.json into MongoDB.
    Requires admin API key in X-Admin-Key header.
    
    - **force_reimport**: If True, drops existing collection and reimports all data
    
    Returns import statistics including:
    - **total_entries**: Total number of alphabet entries processed
    - **inserted**: Number of entries successfully inserted
    - **skipped**: Number of entries skipped (duplicates)
    - **errors**: Number of errors encountered
    - **duration**: Import duration in seconds
    """
    try:
        logger.info("Alphabet import requested by admin")
        
        alphabet_path = "alphabet.json"
        result = await alphabet_service.import_from_json(
            alphabet_path,
            force_reimport=force_reimport,
            dry_run=False
        )
        
        logger.info(
            f"Alphabet import completed: {result['inserted']} letters"
        )
        
        return result
        
    except FileNotFoundError as e:
        logger.error(f"Alphabet file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alphabet file not found"
        )
    except ValueError as e:
        logger.error(f"Invalid alphabet format: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Alphabet import failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Import failed: {str(e)}"
        )


@router.get(
    "/letter/{letter}",
    response_model=AlphabetEntry,
    summary="Get alphabet entry by letter",
    description="Get alphabet entry with examples for a specific letter"
)
async def get_alphabet_letter(
    letter: str,
    alphabet_service: AlphabetService = Depends(get_alphabet_service)
):
    """
    Get alphabet entry by letter
    
    Returns the alphabet entry with position-based examples (initial, middle, final)
    for the specified letter.
    
    - **letter**: Letter to lookup (e.g., 'Aa', 'Bb', 'Ng ng')
    
    Returns the alphabet entry with all examples.
    """
    try:
        result = await alphabet_service.get_by_letter(letter)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alphabet entry not found for letter: {letter}"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alphabet letter: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alphabet letter: {str(e)}"
        )


@router.get(
    "/",
    response_model=List[AlphabetEntry],
    summary="Get all alphabet entries",
    description="Get all alphabet entries with pagination support"
)
async def get_all_alphabet(
    skip: int = 0,
    limit: int = 100,
    alphabet_service: AlphabetService = Depends(get_alphabet_service)
):
    """
    Get all alphabet entries with pagination
    
    Returns all alphabet entries with their position-based examples.
    
    - **skip**: Number of entries to skip (for pagination)
    - **limit**: Maximum number of entries to return (default: 100, max: 200)
    """
    try:
        # Limit maximum to 200 (reasonable since alphabet is small)
        limit = min(limit, 200)
        
        results = await alphabet_service.get_all_letters(
            skip=skip,
            limit=limit
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get alphabet entries: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alphabet entries: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get alphabet statistics",
    description="Get alphabet statistics including total letter count"
)
async def get_alphabet_stats(
    alphabet_service: AlphabetService = Depends(get_alphabet_service)
):
    """
    Get alphabet statistics
    
    Returns comprehensive alphabet statistics:
    - **total_letters**: Total number of alphabet entries (letters)
    """
    try:
        total_letters = await alphabet_service.count_letters()
        
        return {
            "total_letters": total_letters
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )
