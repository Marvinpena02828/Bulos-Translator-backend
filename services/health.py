"""Health check service — simple in-process status check (no database)"""
import asyncio
from datetime import datetime
from typing import Dict, Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


class HealthCheckService:
    """Service for checking application health status."""

    def __init__(self):
        self.timeout = 2  # seconds

    async def check_health(self) -> Dict[str, Any]:
        """
        Return a simple health status for the running application.

        Returns:
            Dict with overall status, timestamp, components, and message.
        """
        logger.debug("Health check requested")

        try:
            result = await asyncio.wait_for(
                self._perform_health_check(),
                timeout=self.timeout
            )
            logger.debug(f"Health check result: {result['status']}")
            return result

        except asyncio.TimeoutError:
            logger.error(f"Health check timeout ({self.timeout}s)")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "api": {"status": "unknown", "message": "Health check timed out"}
                },
                "message": f"Health check exceeded timeout of {self.timeout} seconds"
            }

    async def _perform_health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "api": {"status": "healthy", "message": "API is running"}
            },
            "message": "All systems operational"
        }
