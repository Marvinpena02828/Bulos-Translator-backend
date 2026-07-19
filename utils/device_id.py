"""Device identification utility for device-based tracking"""
from fastapi import Header, HTTPException, status
from typing import Optional
import re
import logging

logger = logging.getLogger(__name__)

# UUID v4 validation pattern
UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)


def validate_device_id(device_id: str) -> bool:
    """
    Validate device ID format (UUID v4).
    
    Args:
        device_id: Device identifier string
        
    Returns:
        bool: True if valid UUID v4 format
    """
    if not device_id:
        return False
    return bool(UUID_PATTERN.match(device_id))


async def get_device_id(
    x_device_id: Optional[str] = Header(None, alias="X-Device-ID")
) -> str:
    """
    FastAPI dependency to extract and validate device ID from headers.
    
    Args:
        x_device_id: Device ID from X-Device-ID header
        
    Returns:
        str: Validated device ID
        
    Raises:
        HTTPException: 400 if device ID is missing or invalid
        
    Example:
        >>> @router.get("/my-endpoint")
        >>> async def my_endpoint(device_id: str = Depends(get_device_id)):
        >>>     # device_id is automatically extracted and validated
        >>>     pass
    """
    if not x_device_id:
        logger.warning("Request missing X-Device-ID header")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Device ID is required. Please provide X-Device-ID header."
        )
    
    if not validate_device_id(x_device_id):
        logger.warning(f"Invalid device ID format: {x_device_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid device ID format. Must be a valid UUID v4."
        )
    
    logger.debug(f"Device ID validated: {x_device_id}")
    return x_device_id
