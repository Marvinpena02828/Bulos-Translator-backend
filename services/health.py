"""Health check service for monitoring system components"""
import asyncio
from datetime import datetime
from typing import Dict, Any

from services.database import DatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class HealthCheckService:
    """Service for checking health status of system components"""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize health check service.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        self.timeout = 2  # seconds

    async def check_database(self) -> Dict[str, Any]:
        """
        Check database connectivity using admin ping.

        Returns:
            Dict with 'status' ('healthy'/'unhealthy') and 'message'.
        """
        try:
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

    async def check_health(self) -> Dict[str, Any]:
        """
        Check health status of all system components.

        Runs with a 2-second timeout to ensure responsiveness.

        Returns:
            Dict with overall status, timestamp, per-component details, and message.
        """
        logger.info("Starting health check")

        try:
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
                    }
                },
                "message": f"Health check exceeded timeout of {self.timeout} seconds"
            }

    async def _perform_health_check(self) -> Dict[str, Any]:
        """Internal health check — runs component checks concurrently."""
        database_check = await self.check_database()

        if isinstance(database_check, Exception):
            database_check = {
                "status": "unhealthy",
                "message": f"Database check error: {str(database_check)}"
            }

        components = {
            "database": database_check
        }

        all_healthy = all(
            comp["status"] == "healthy"
            for comp in components.values()
        )

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
            "message": (
                "All systems operational" if all_healthy
                else "One or more components are unhealthy"
            )
        }
