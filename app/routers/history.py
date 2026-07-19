"""History tracking API endpoints"""
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException, Query, status

from models.schemas import HistoryResponse
from services.database import DatabaseManager
from utils.device_id import get_device_id
from config import settings
from utils.logging_config import get_logger

# Lazy import for HistoryService
if TYPE_CHECKING:
    from services.history import HistoryService

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/history",
    tags=["History"]
)

# History service singleton
_history_service: "HistoryService" = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance from main app"""
    from app import main
    if main._db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    return main._db_manager


async def get_history_service(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> "HistoryService":
    """Get history service instance (singleton pattern)"""
    from services.history import HistoryService
    
    global _history_service
    if _history_service is None:
        _history_service = HistoryService(db_manager)
    return _history_service


@router.get(
    "/",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get device action history",
    description="Retrieve device action history with optional filtering by action type and date range"
)
async def get_history(
    device_id: str = Depends(get_device_id),
    action_type: Optional[str] = Query(None, description="Filter by action type (e.g., 'vocabulary_create', 'translation')"),
    start_date: Optional[datetime] = Query(None, description="Filter actions after this date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="Filter actions before this date (ISO format)"),
    page: int = Query(0, ge=0, description="Page number (0-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of records per page"),
    history_service: "HistoryService" = Depends(get_history_service)
):
    """
    Retrieve device action history with filtering and pagination
    
    Requires X-Device-ID header
    
    This endpoint returns a paginated list of all actions performed by the device,
    with optional filters for action type and date range.
    
    Query Parameters:
    - **action_type**: Filter by specific action type (optional)
      - Examples: 'vocabulary_create', 'vocabulary_update', 'vocabulary_delete'
      - 'translation', 'pronunciation_evaluation'
      - 'session_start', 'session_end'
    - **start_date**: Filter actions after this date (ISO 8601 format, optional)
    - **end_date**: Filter actions before this date (ISO 8601 format, optional)
    - **page**: Page number, 0-indexed (default: 0)
    - **page_size**: Number of records per page, 1-100 (default: 20)
    
    Returns:
    - **records**: List of history records with the following fields:
      - **id**: History record ID
      - **device_id**: Device ID that performed the action
      - **action_type**: Type of action performed
      - **resource_id**: ID of the resource affected (if applicable)
      - **resource_type**: Type of resource (e.g., 'vocabulary', 'evaluation')
      - **outcome**: Outcome of the action ('success', 'error', 'failure')
      - **timestamp**: When the action was performed
      - **details**: Additional details about the action
    - **total**: Total number of records matching the query
    - **page**: Current page number
    - **page_size**: Number of records per page
    
    Common Action Types:
    - Vocabulary: vocabulary_create, vocabulary_update, vocabulary_delete
    - Translation: translation
    - Speech: pronunciation_evaluation
    - Sessions: session_start, session_end
    """
    try:
        logger.info(
            f"Get history request from device {device_id} "
            f"(action_type: {action_type}, page: {page}, size: {page_size})"
        )
        
        result = await history_service.get_history(
            device_id=device_id,
            action_type=action_type,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size
        )
        
        return HistoryResponse(**result)
    
    except Exception as e:
        logger.error(
            f"Failed to retrieve history for device {device_id}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve history records"
        )


@router.get(
    "/summary",
    status_code=status.HTTP_200_OK,
    summary="Get action summary",
    description="Get a summary of device actions with statistics"
)
async def get_action_summary(
    device_id: str = Depends(get_device_id),
    start_date: Optional[datetime] = Query(None, description="Filter actions after this date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="Filter actions before this date (ISO format)"),
    history_service: "HistoryService" = Depends(get_history_service)
):
    """
    Get a summary of device actions with statistics
    
    Requires X-Device-ID header
    
    This endpoint provides aggregate statistics on device actions, including
    counts by action type and outcome.
    
    Query Parameters:
    - **start_date**: Filter actions after this date (ISO 8601 format, optional)
    - **end_date**: Filter actions before this date (ISO 8601 format, optional)
    
    Returns:
    - **device_id**: Device ID
    - **total_actions**: Total number of actions performed
    - **actions_by_type**: Dictionary with count of each action type
    - **actions_by_outcome**: Dictionary with count of each outcome (success/error)
    """
    try:
        logger.info(
            f"Get action summary request from device {device_id}"
        )
        
        result = await history_service.get_action_summary(
            device_id=device_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return result
    
    except Exception as e:
        logger.error(
            f"Failed to retrieve action summary for device {device_id}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve action summary"
        )
