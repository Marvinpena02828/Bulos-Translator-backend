"""Alphabet API endpoints"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status

from models.schemas import AlphabetEntry, ImportResult
from services.alphabet import AlphabetService
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/alphabet",
    tags=["Alphabet"]
)

_alphabet_service = AlphabetService()


def get_alphabet_service() -> AlphabetService:
    return _alphabet_service


def verify_admin_key(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    if not x_admin_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Admin API key required")
    if x_admin_key != settings.admin_api_key:
        logger.warning("Invalid admin API key attempt")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid admin API key")
    return x_admin_key


@router.post("/import", response_model=ImportResult, status_code=status.HTTP_200_OK,
             summary="Reload alphabet data (Admin only)")
async def import_alphabet(
    force_reimport: bool = False,
    admin_key: str = Depends(verify_admin_key),
    alphabet_service: AlphabetService = Depends(get_alphabet_service),
):
    """Reload alphabet.json into the in-memory cache. Requires X-Admin-Key header."""
    try:
        return await alphabet_service.import_from_json("alphabet.json",
                                                        force_reimport=force_reimport)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Alphabet file not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Alphabet reload failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Reload failed: {str(e)}")


@router.get("/letter/{letter}", response_model=AlphabetEntry,
            summary="Get alphabet entry by letter")
async def get_alphabet_letter(
    letter: str,
    alphabet_service: AlphabetService = Depends(get_alphabet_service),
):
    try:
        result = await alphabet_service.get_by_letter(letter)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Alphabet entry not found for letter: {letter}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alphabet letter: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to get alphabet letter: {str(e)}")


@router.get("/", response_model=List[AlphabetEntry],
            summary="Get all alphabet entries")
async def get_all_alphabet(
    skip: int = 0,
    limit: int = 100,
    alphabet_service: AlphabetService = Depends(get_alphabet_service),
):
    try:
        return await alphabet_service.get_all_letters(skip=skip, limit=min(limit, 200))
    except Exception as e:
        logger.error(f"Failed to get alphabet entries: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to get alphabet entries: {str(e)}")


@router.get("/stats", summary="Get alphabet statistics")
async def get_alphabet_stats(
    alphabet_service: AlphabetService = Depends(get_alphabet_service),
):
    try:
        return {"total_letters": await alphabet_service.count_letters()}
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to get statistics: {str(e)}")
