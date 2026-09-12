"""History endpoints — returns empty results (no database)."""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query, status

from utils.device_id import get_device_id
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/history",
    tags=["History"]
)


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Get device action history",
    description="History is not persisted in this deployment — always returns empty.",
)
async def get_history(
    device_id: str = Depends(get_device_id),
    action_type: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
):
    return {"records": [], "total": 0, "page": page, "page_size": page_size}


@router.get(
    "/summary",
    status_code=status.HTTP_200_OK,
    summary="Get action summary",
    description="History is not persisted in this deployment — always returns empty.",
)
async def get_action_summary(
    device_id: str = Depends(get_device_id),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
):
    return {
        "device_id": device_id,
        "total_actions": 0,
        "actions_by_type": {},
        "actions_by_outcome": {},
    }
