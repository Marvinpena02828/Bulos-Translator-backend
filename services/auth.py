"""Authentication and authorization module using JWT tokens"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import logging

from config import settings


# HTTP Bearer security scheme for JWT token authentication
security = HTTPBearer()

# Logger for authentication operations
logger = logging.getLogger(__name__)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token with configurable expiration.
    
    Args:
        data: Dictionary containing claims to encode in the token (typically {"sub": user_id})
        expires_delta: Optional custom expiration time. If not provided, uses settings default
        
    Returns:
        str: Encoded JWT token
        
    Example:
        >>> token = create_access_token({"sub": "user123"})
        >>> # Token valid for default duration (30 minutes)
        
        >>> token = create_access_token({"sub": "user123"}, expires_delta=timedelta(hours=1))
        >>> # Token valid for 1 hour
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    # Add expiration claim to token payload
    to_encode.update({"exp": expire})
    
    # Encode the JWT token
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    
    logger.debug(f"Created access token with expiration: {expire}")
    
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Dependency to extract and validate the current user from JWT token.
    
    This function is used as a FastAPI dependency to protect endpoints.
    It extracts the Bearer token from the Authorization header, validates it,
    and returns the user_id from the token's "sub" claim.
    
    Args:
        credentials: HTTP Bearer credentials automatically extracted by FastAPI
        
    Returns:
        str: User ID extracted from the token's "sub" claim
        
    Raises:
        HTTPException: 401 status if token is invalid, expired, or missing required claims
        
    Example:
        >>> @app.get("/protected")
        >>> async def protected_route(user_id: str = Depends(get_current_user)):
        >>>     return {"user_id": user_id}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Decode and validate JWT token
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        
        # Extract user_id from the "sub" claim
        user_id: str = payload.get("sub")
        
        if user_id is None:
            logger.warning("Token validation failed: missing 'sub' claim")
            raise credentials_exception
        
        logger.debug(f"Token validated successfully for user: {user_id}")
        return user_id
        
    except JWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        raise credentials_exception


def verify_resource_ownership(user_id: str, resource_user_id: str) -> None:
    """
    Verify that a user owns a specific resource.
    
    This function enforces user data isolation by ensuring that users
    can only access their own resources. Call this function in service
    methods before allowing access to user-specific data.
    
    Args:
        user_id: ID of the currently authenticated user (from JWT token)
        resource_user_id: ID of the user who owns the resource being accessed
        
    Raises:
        HTTPException: 403 status if user_id does not match resource_user_id
        
    Example:
        >>> # In a service method
        >>> vocabulary = await db.find_one("vocabulary", {"_id": vocab_id})
        >>> verify_resource_ownership(current_user_id, vocabulary["user_id"])
        >>> # Raises 403 if user doesn't own the vocabulary item
    """
    if user_id != resource_user_id:
        logger.warning(
            f"Authorization failed: user {user_id} attempted to access "
            f"resource owned by user {resource_user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this resource"
        )
    
    logger.debug(f"Resource ownership verified for user: {user_id}")


def create_user_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Convenience function to create a token for a specific user.
    
    This is a wrapper around create_access_token that follows the standard
    pattern of encoding user_id in the "sub" claim.
    
    Args:
        user_id: ID of the user to create token for
        expires_delta: Optional custom expiration time
        
    Returns:
        str: Encoded JWT token
        
    Example:
        >>> token = create_user_token("user123")
        >>> # Returns token with {"sub": "user123"} claim
    """
    return create_access_token(
        data={"sub": user_id},
        expires_delta=expires_delta
    )


# Optional: Helper function for getting user from token without dependency injection
def decode_token(token: str) -> Optional[str]:
    """
    Decode a JWT token and extract the user_id without raising exceptions.
    
    This is useful for scenarios where you want to check a token
    without enforcing authentication (e.g., optional authentication).
    
    Args:
        token: JWT token string
        
    Returns:
        Optional[str]: User ID if token is valid, None otherwise
        
    Example:
        >>> user_id = decode_token(token_string)
        >>> if user_id:
        >>>     # Token is valid
        >>> else:
        >>>     # Token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        return payload.get("sub")
    except JWTError:
        return None
