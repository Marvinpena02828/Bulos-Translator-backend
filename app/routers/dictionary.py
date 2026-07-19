"""Dictionary API endpoints for vocabulary lookup and search"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status

from models.schemas import (
    DictionaryEntry,
    DictionaryLookupRequest,
    DictionarySearchRequest,
    ImportResult,
    CategoryInfo
)
from services.dictionary import DictionaryService
from services.database import DatabaseManager
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/dictionary",
    tags=["Dictionary"]
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


def get_dictionary_service(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> DictionaryService:
    """Get dictionary service instance"""
    return DictionaryService(db_manager)


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
    summary="Import dictionary data (Admin only)",
    description="Import dictionary.json into MongoDB. Requires admin authentication."
)
async def import_dictionary(
    force_reimport: bool = False,
    admin_key: str = Depends(verify_admin_key),
    dict_service: DictionaryService = Depends(get_dictionary_service)
):
    """
    Import dictionary from JSON file (Admin only)
    
    This endpoint triggers the import of dictionary.json into MongoDB.
    Requires admin API key in X-Admin-Key header.
    
    - **force_reimport**: If True, drops existing collection and reimports all data
    
    Returns import statistics including:
    - **total_entries**: Total number of entries processed
    - **inserted**: Number of entries successfully inserted
    - **skipped**: Number of entries skipped (duplicates)
    - **errors**: Number of errors encountered
    - **duration**: Import duration in seconds
    """
    try:
        logger.info("Dictionary import requested by admin")
        
        dictionary_path = "dictionary.json"
        result = await dict_service.import_from_json(
            dictionary_path,
            force_reimport=force_reimport,
            dry_run=False
        )
        
        # Update version metadata after successful import
        await dict_service.update_version_metadata(
            changelog=f"Dictionary import: {result['inserted']} entries added"
        )
        
        logger.info(
            f"Dictionary import completed: {result['inserted']} entries"
        )
        
        return result
        
    except FileNotFoundError as e:
        logger.error(f"Dictionary file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dictionary file not found"
        )
    except ValueError as e:
        logger.error(f"Invalid dictionary format: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Dictionary import failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Import failed: {str(e)}"
        )


@router.post(
    "/lookup",
    response_model=List[DictionaryEntry],
    summary="Look up a word",
    description="Look up a word in the dictionary by language"
)
async def lookup_word(
    request: DictionaryLookupRequest,
    dict_service: DictionaryService = Depends(get_dictionary_service)
):
    """
    Look up a word in the dictionary
    
    Searches for exact or partial matches in the specified language.
    
    - **word**: Word to look up
    - **source_language**: Language code (bul, tl, en)
    
    Returns a list of matching dictionary entries with translations.
    """
    try:
        results = await dict_service.lookup_word(
            request.word,
            request.source_language
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Dictionary lookup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lookup failed: {str(e)}"
        )


@router.post(
    "/search",
    response_model=List[DictionaryEntry],
    summary="Search for words",
    description="Search for words using fuzzy text search"
)
async def search_words(
    request: DictionarySearchRequest,
    dict_service: DictionaryService = Depends(get_dictionary_service)
):
    """
    Search for words in the dictionary
    
    Performs fuzzy text search across all languages or specific language.
    
    - **query**: Search query string
    - **language**: Language to search (bul, tl, en, or 'all' for all languages)
    - **limit**: Maximum number of results (default: 50, max: 200)
    
    Returns a list of matching dictionary entries.
    """
    try:
        results = await dict_service.search_words(
            request.query,
            request.language,
            request.limit
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Dictionary search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/categories",
    response_model=List[CategoryInfo],
    summary="Get all categories",
    description="Get all dictionary categories with entry counts"
)
async def get_categories(
    dict_service: DictionaryService = Depends(get_dictionary_service)
):
    """
    Get all dictionary categories
    
    Returns a list of all categories with their entry counts, sorted by count (descending).
    
    Each category includes:
    - **category**: Full category name
    - **category_key**: Normalized category key (kebab-case)
    - **entry_count**: Number of entries in this category
    """
    try:
        categories = await dict_service.get_all_categories()
        return categories
        
    except Exception as e:
        logger.error(f"Failed to get categories: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get categories: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get dictionary statistics",
    description="Get dictionary statistics including total entries and categories"
)
async def get_dictionary_stats(
    dict_service: DictionaryService = Depends(get_dictionary_service)
):
    """
    Get dictionary statistics
    
    Returns comprehensive dictionary statistics:
    - **total_entries**: Total number of dictionary entries
    - **total_categories**: Total number of categories
    - **categories**: List of all categories with counts
    """
    try:
        total_entries = await dict_service.count_entries()
        categories = await dict_service.get_all_categories()
        
        return {
            "total_entries": total_entries,
            "total_categories": len(categories),
            "categories": categories
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get(
    "/category/{category_key}",
    response_model=List[DictionaryEntry],
    summary="Get entries by category",
    description="Get all dictionary entries in a specific category"
)
async def get_category_entries(
    category_key: str,
    skip: int = 0,
    limit: int = 100,
    dict_service: DictionaryService = Depends(get_dictionary_service)
):
    """
    Get entries by category
    
    Returns all dictionary entries in a specific category with pagination support.
    
    - **category_key**: Category key (kebab-case, e.g., "parts-of-body")
    - **skip**: Number of entries to skip (for pagination)
    - **limit**: Maximum number of entries to return (default: 100, max: 500)
    """
    try:
        # Limit maximum to 500
        limit = min(limit, 500)
        
        results = await dict_service.get_by_category(
            category_key,
            skip=skip,
            limit=limit
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get category entries: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get category entries: {str(e)}"
        )


@router.get(
    "/version",
    summary="Get dictionary version",
    description="Get dictionary version and metadata"
)
async def get_dictionary_version(
    dict_service: DictionaryService = Depends(get_dictionary_service)
):
    """
    Get dictionary version metadata
    
    Returns dictionary version information for client-side update detection:
    - **version**: Current dictionary version (e.g., "1.2")
    - **last_updated**: Timestamp of last update
    - **total_entries**: Total number of dictionary entries
    - **changelog**: Description of latest changes
    """
    try:
        metadata = await dict_service.get_version_metadata()
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get version metadata: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get version metadata: {str(e)}"
        )
