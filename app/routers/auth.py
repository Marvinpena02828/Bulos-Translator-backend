"""Authentication API endpoints for user registration and login"""
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import timedelta

from models.schemas import UserRegister, UserLogin, TokenResponse, UserResponse
from services.auth import create_user_token, get_current_user
from services.database import DatabaseManager
from config import settings
from utils.logging_config import get_logger
import hashlib
from datetime import datetime
from bson import ObjectId

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"]
)


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance from main app"""
    from app import main
    if main._db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    return main._db_manager


def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(plain_password) == hashed_password


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with username, email, and password"
)
async def register(
    user_data: UserRegister,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Register a new user account
    
    - **username**: Unique username (3-50 characters)
    - **email**: Valid email address
    - **password**: Password (minimum 6 characters)
    - **full_name**: Optional full name
    
    Returns the created user information (without password)
    """
    try:
        # Check if username already exists
        existing_user = await db_manager.find_one(
            "users",
            {"username": user_data.username}
        )
        
        if existing_user:
            logger.warning(f"Registration failed: username '{user_data.username}' already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        existing_email = await db_manager.find_one(
            "users",
            {"email": user_data.email}
        )
        
        if existing_email:
            logger.warning(f"Registration failed: email '{user_data.email}' already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user document
        user_doc = {
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": hash_password(user_data.password),
            "full_name": user_data.full_name,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Insert user into database
        result_id = await db_manager.insert_one("users", user_doc)
        
        # Fetch created user
        created_user = await db_manager.find_one(
            "users",
            {"_id": ObjectId(result_id)}
        )
        
        logger.info(f"User registered successfully: {user_data.username}")
        
        # Return user data (without password)
        return UserResponse(
            _id=str(created_user["_id"]),
            username=created_user["username"],
            email=created_user["email"],
            full_name=created_user.get("full_name"),
            created_at=created_user["created_at"],
            is_active=created_user["is_active"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User login",
    description="Authenticate user and receive JWT access token"
)
async def login(
    credentials: UserLogin,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Authenticate user and receive access token
    
    - **username**: Username or email
    - **password**: User password
    
    Returns:
    - **access_token**: JWT token for authentication
    - **token_type**: Token type (always "bearer")
    - **user_id**: Authenticated user ID
    - **username**: Authenticated username
    - **expires_in**: Token expiration time in seconds
    
    Use the access_token in the Authorization header for protected endpoints:
    ```
    Authorization: Bearer <access_token>
    ```
    """
    try:
        # Find user by username or email
        user = await db_manager.find_one(
            "users",
            {"$or": [
                {"username": credentials.username},
                {"email": credentials.username}
            ]}
        )
        
        if not user:
            logger.warning(f"Login failed: user '{credentials.username}' not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Verify password
        if not verify_password(credentials.password, user["password_hash"]):
            logger.warning(f"Login failed: invalid password for user '{credentials.username}'")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Check if user is active
        if not user.get("is_active", True):
            logger.warning(f"Login failed: user '{credentials.username}' is inactive")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Create access token
        user_id = str(user["_id"])
        access_token = create_user_token(user_id)
        
        logger.info(f"User logged in successfully: {user['username']}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user_id,
            username=user["username"],
            expires_in=settings.access_token_expire_minutes * 60  # Convert to seconds
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process login"
        )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current user",
    description="Get information about the currently authenticated user"
)
async def get_current_user_info(
    user_id: str = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Get current user information
    
    Requires authentication. Returns information about the user
    associated with the provided JWT token.
    """
    try:
        user = await db_manager.find_one(
            "users",
            {"_id": ObjectId(user_id)}
        )
        
        if not user:
            logger.error(f"User not found: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            _id=str(user["_id"]),
            username=user["username"],
            email=user["email"],
            full_name=user.get("full_name"),
            created_at=user["created_at"],
            is_active=user["is_active"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )
