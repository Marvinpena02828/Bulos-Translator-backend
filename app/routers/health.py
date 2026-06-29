"""Health check API endpoint"""
from typing import TYPE_CHECKING, Optional
from fastapi import APIRouter, Depends, Response, status

from services.database import DatabaseManager
from config import settings
from utils.logging_config import get_logger

# Lazy import for health service
if TYPE_CHECKING:
    from services.health import HealthCheckService
    from services.speech import SpeechProcessor

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/health",
    tags=["Health"]
)

# Global instances (will be initialized on startup)
_db_manager: DatabaseManager = None
_speech_processor: Optional["SpeechProcessor"] = None
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


def get_speech_processor() -> Optional["SpeechProcessor"]:
    """Get speech processor instance (may be None if not initialized)"""
    # Import here to avoid loading at module load time
    from services.speech import SpeechProcessor
    
    global _speech_processor
    # Return the global instance if it exists, otherwise None
    return _speech_processor


async def get_health_service(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> "HealthCheckService":
    """Get health service instance (singleton pattern)"""
    from services.health import HealthCheckService
    
    global _health_service, _speech_processor
    
    # Always create a fresh instance with current speech processor state
    # This ensures we check the actual state of components
    _health_service = HealthCheckService(
        db_manager=db_manager,
        speech_processor=_speech_processor
    )
    
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
    Check system health status
    
    This endpoint checks the health of all system components and returns
    a comprehensive status report. No authentication is required.
    
    Components Checked:
    - **Database**: MongoDB connectivity and availability
    - **Speech Processor**: DeepSpeech model availability (optional)
    
    Returns:
    - **status**: Overall status ('healthy' or 'unhealthy')
    - **timestamp**: ISO timestamp of the health check
    - **components**: Dictionary with status of each component
      - **database**: Database connection status and message
      - **speech_processor**: Speech processor status and message
    - **message**: Overall status message
    
    Status Codes:
    - **200 OK**: All components are healthy
    - **503 Service Unavailable**: One or more components are unhealthy
    
    The health check completes within 2 seconds to ensure responsiveness.
    """
    try:
        # Perform health check
        result = await health_service.check_health()
        
        # Set appropriate status code based on health
        if result["status"] == "unhealthy":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            logger.warning(f"Health check failed: {result['message']}")
        else:
            response.status_code = status.HTTP_200_OK
            logger.debug("Health check passed")
        
        return result
    
    except Exception as e:
        # If health check itself fails, return unhealthy status
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
                },
                "speech_processor": {
                    "status": "unknown",
                    "message": "Health check failed"
                }
            },
            "message": f"Health check failed: {str(e)}"
        }
