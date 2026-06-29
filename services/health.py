"""Health check service for monitoring system components"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from services.database import DatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class HealthCheckService:
    """Service for checking health status of system components"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        speech_processor: Optional[Any] = None
    ):
        """
        Initialize health check service
        
        Args:
            db_manager: Database manager instance
            speech_processor: Optional speech processor instance
        """
        self.db = db_manager
        self.speech_processor = speech_processor
        self.timeout = 2  # 2-second timeout for health checks
    
    async def check_database(self) -> Dict[str, Any]:
        """
        Check database connectivity using admin ping command
        
        Returns:
            Dictionary with status and message:
            - status: 'healthy' or 'unhealthy'
            - message: Description of the health status
        """
        try:
            # Use MongoDB admin ping command to check connectivity
            await self.db.db.command('ping')
            
            logger.debug("Database health check: healthy")
            
            return {
                "status": "healthy",
                "message": "Database connection is active"
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}", exc_info=True)
            
            return {
                "status": "unhealthy",
                "message": f"Database connection failed: {str(e)}"
            }
    
    async def check_speech_processor(self) -> Dict[str, Any]:
        """
        Check speech processor model availability
        
        Returns:
            Dictionary with status and message:
            - status: 'healthy' or 'unhealthy'
            - message: Description of the health status
        """
        try:
            if self.speech_processor is None:
                logger.debug("Speech processor not initialized (optional component)")
                return {
                    "status": "healthy",
                    "message": "Speech processor not initialized (optional)"
                }
            
            # Check if model is loaded
            if self.speech_processor.model is None:
                logger.warning("Speech processor model not loaded")
                return {
                    "status": "unhealthy",
                    "message": "Speech processor model not loaded"
                }
            
            logger.debug("Speech processor health check: healthy")
            
            return {
                "status": "healthy",
                "message": "Speech processor model is loaded and ready"
            }
            
        except Exception as e:
            logger.error(f"Speech processor health check failed: {str(e)}", exc_info=True)
            
            return {
                "status": "unhealthy",
                "message": f"Speech processor check failed: {str(e)}"
            }
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check health status of all system components
        
        Performs health checks on:
        - Database connectivity
        - Speech processor availability
        
        Returns comprehensive health status with component-level details.
        Health check completes within 2 seconds (enforced by timeout).
        
        Returns:
            Dictionary with:
            - status: Overall status ('healthy' if all components healthy, 'unhealthy' otherwise)
            - timestamp: ISO timestamp of the health check
            - components: Dictionary with status of each component
            - message: Overall status message
        """
        logger.info("Starting comprehensive health check")
        
        try:
            # Run health check with timeout
            result = await asyncio.wait_for(
                self._perform_health_check(),
                timeout=self.timeout
            )
            
            logger.info(f"Health check completed: {result['status']}")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Health check timeout ({self.timeout}s) exceeded")
            
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "database": {
                        "status": "unknown",
                        "message": "Health check timed out"
                    },
                    "speech_processor": {
                        "status": "unknown",
                        "message": "Health check timed out"
                    }
                },
                "message": f"Health check exceeded timeout of {self.timeout} seconds"
            }
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """
        Internal method to perform the actual health check
        
        Returns:
            Dictionary with health check results
        """
        # Run component checks concurrently
        database_check, speech_check = await asyncio.gather(
            self.check_database(),
            self.check_speech_processor(),
            return_exceptions=True
        )
        
        # Handle exceptions from gather
        if isinstance(database_check, Exception):
            database_check = {
                "status": "unhealthy",
                "message": f"Database check error: {str(database_check)}"
            }
        
        if isinstance(speech_check, Exception):
            speech_check = {
                "status": "unhealthy",
                "message": f"Speech processor check error: {str(speech_check)}"
            }
        
        # Build components status
        components = {
            "database": database_check,
            "speech_processor": speech_check
        }
        
        # Determine overall status
        # Overall is healthy only if ALL components are healthy
        all_healthy = all(
            comp["status"] == "healthy" 
            for comp in components.values()
        )
        
        overall_status = "healthy" if all_healthy else "unhealthy"
        
        # Build response
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
            "message": (
                "All systems operational" if all_healthy 
                else "One or more components are unhealthy"
            )
        }
