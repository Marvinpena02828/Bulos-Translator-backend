"""Health check API endpoint"""
from typing import TYPE_CHECKING, Optional
from fastapi import APIRouter, Depends, Response, status

from utils.logging_config import get_logger

if TYPE_CHECKING:
    from services.health import HealthCheckService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/health",
    tags=["Health"]
)

_health_service: Optional["HealthCheckService"] = None


def get_health_service() -> "HealthCheckService":
    from services.health import HealthCheckService
    global _health_service
    if _health_service is None:
        _health_service = HealthCheckService()
    return _health_service


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the API (no authentication required)"
)
async def health_check(
    response: Response,
    health_service: "HealthCheckService" = Depends(get_health_service),
):
    """Check API health status. Returns status, timestamp, and component details."""
    try:
        result = await health_service.check_health()
        if result["status"] == "unhealthy":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        from datetime import datetime
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {"api": {"status": "unknown", "message": "Health check failed"}},
            "message": f"Health check failed: {str(e)}",
        }
