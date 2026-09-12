"""Dictionary API endpoints"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status

from models.schemas import (
    DictionaryEntry,
    DictionaryLookupRequest,
    DictionarySearchRequest,
    ImportResult,
    CategoryInfo,
)
from services.dictionary import DictionaryService
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/dictionary",
    tags=["Dictionary"]
)

_dict_service = DictionaryService()


def get_dictionary_service() -> DictionaryService:
    return _dict_service


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
             summary="Reload dictionary data (Admin only)")
async def import_dictionary(
    force_reimport: bool = False,
    admin_key: str = Depends(verify_admin_key),
    dict_service: DictionaryService = Depends(get_dictionary_service),
):
    """Reload dictionary.json into the in-memory cache. Requires X-Admin-Key header."""
    try:
        result = await dict_service.import_from_json("dictionary.json",
                                                      force_reimport=force_reimport)
        await dict_service.update_version_metadata(
            changelog=f"Dictionary reloaded: {result['inserted']} entries"
        )
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Dictionary file not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Dictionary reload failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Reload failed: {str(e)}")


@router.post("/lookup", response_model=List[DictionaryEntry],
             summary="Look up a word")
async def lookup_word(
    request: DictionaryLookupRequest,
    dict_service: DictionaryService = Depends(get_dictionary_service),
):
    try:
        return await dict_service.lookup_word(request.word, request.source_language)
    except Exception as e:
        logger.error(f"Dictionary lookup failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Lookup failed: {str(e)}")


@router.post("/search", response_model=List[DictionaryEntry],
             summary="Search for words")
async def search_words(
    request: DictionarySearchRequest,
    dict_service: DictionaryService = Depends(get_dictionary_service),
):
    try:
        return await dict_service.search_words(request.query, request.language, request.limit)
    except Exception as e:
        logger.error(f"Dictionary search failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Search failed: {str(e)}")


@router.get("/categories", response_model=List[CategoryInfo],
            summary="Get all categories")
async def get_categories(
    dict_service: DictionaryService = Depends(get_dictionary_service),
):
    try:
        return await dict_service.get_all_categories()
    except Exception as e:
        logger.error(f"Failed to get categories: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to get categories: {str(e)}")


@router.get("/stats", summary="Get dictionary statistics")
async def get_dictionary_stats(
    dict_service: DictionaryService = Depends(get_dictionary_service),
):
    try:
        total = await dict_service.count_entries()
        categories = await dict_service.get_all_categories()
        return {"total_entries": total, "total_categories": len(categories),
                "categories": categories}
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to get statistics: {str(e)}")


@router.get("/category/{category_key}", response_model=List[DictionaryEntry],
            summary="Get entries by category")
async def get_category_entries(
    category_key: str,
    skip: int = 0,
    limit: int = 100,
    dict_service: DictionaryService = Depends(get_dictionary_service),
):
    try:
        return await dict_service.get_by_category(category_key, skip=skip,
                                                   limit=min(limit, 500))
    except Exception as e:
        logger.error(f"Failed to get category entries: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to get category entries: {str(e)}")


@router.get("/version", summary="Get dictionary version")
async def get_dictionary_version(
    dict_service: DictionaryService = Depends(get_dictionary_service),
):
    try:
        return await dict_service.get_version_metadata()
    except Exception as e:
        logger.error(f"Failed to get version metadata: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to get version metadata: {str(e)}")
