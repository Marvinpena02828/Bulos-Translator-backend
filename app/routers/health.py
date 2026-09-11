"""Health check API endpoint"""
from typing import TYPE_CHECKING, Optional
from fastapi import APIRouter, Depends, Response, status

from services.database import DatabaseManager
from config import settings
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from services.health import HealthCheckService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/health",
    tags=["Health"]
)

# Global instances (will be initialized on startup)
_db_manager: DatabaseManager = None
_health_service: Optional["HealthCheckService"] = None


def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    global _db_manager
    if _db_manager is None:
        from services.database import initialize_db_manager
        _db_manager = initialize_db_manager(
            connection_string=settings.mongodb_url,
            database_name=settings.mongodb_database,
            max_pool_size=settings.mongodb_max_pool_size,
            min_pool_size=settings.mongodb_min_pool_size
        )
    return _db_manager


async def get_health_service(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> "HealthCheckService":
    """Get health service instance"""
    from services.health import HealthCheckService

    global _health_service
    _health_service = HealthCheckService(db_manager=db_manager)
    return _health_service


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of all system components (no authentication required)"
)
async def health_check(
    response: Response,
    health_service: "HealthCheckService" = Depends(get_health_service)
):
    """
    Check system health status.

    Components checked:
    - **Database**: MongoDB connectivity and availability

    Returns:
    - **status**: Overall status ('healthy' or 'unhealthy')
    - **timestamp**: ISO timestamp of the health check
    - **components**: Per-component status details
    - **message**: Overall status message

    Status codes:
    - **200 OK**: All components healthy
    - **503 Service Unavailable**: One or more components unhealthy
    """
    try:
        result = await health_service.check_health()

        if result["status"] == "unhealthy":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            logger.warning(f"Health check failed: {result['message']}")
        else:
            response.status_code = status.HTTP_200_OK
            logger.debug("Health check passed")

        return result

    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        from datetime import datetime
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": {
                    "status": "unknown",
                    "message": "Health check failed"
                }
            },
            "message": f"Health check failed: {str(e)}"
        }
