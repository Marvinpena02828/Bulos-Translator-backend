"""FastAPI application entry point with middleware configuration"""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from utils.logging_config import configure_logging, get_logger
from app.routers import vocabulary_router, translation_router, history_router, health_router, dictionary_router, evaluation_router, alphabet_router

# Configure logging before anything else
configure_logging()
logger = get_logger(__name__)

# Global service instances
_db_manager = None
_translation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown tasks
    
    On startup:
    - Configure logging (already done globally)
    - Connect to database
    - Initialize translation service

    On shutdown:
    - Disconnect from database
    """
    global _db_manager, _translation_service
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"CORS origins: {settings.cors_origins}")
    
    try:
        # Initialize database connection
        logger.info("Connecting to database...")
        from services.database import initialize_db_manager
        _db_manager = initialize_db_manager(
            connection_string=settings.mongodb_url,
            database_name=settings.mongodb_database,
            max_pool_size=settings.mongodb_max_pool_size,
            min_pool_size=settings.mongodb_min_pool_size
        )
        await _db_manager.connect()
        logger.info("Database connected successfully")
        
        # Initialize translation service
        logger.info("Initializing translation service...")
        try:
            from services.translation import TranslationService
            _translation_service = TranslationService(_db_manager)
            await _translation_service.initialize()
            logger.info("Translation service initialized successfully")
        except Exception as e:
            logger.warning(f"Translation service initialization failed: {str(e)}")
            logger.warning("Translation endpoints may not be available")
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        logger.error("Application startup failed - some services may be unavailable")
    
    # Yield control to application
    yield
    
    # Shutdown
    logger.info("Application shutdown initiated")
    
    try:
        if _db_manager:
            logger.info("Disconnecting from database...")
            await _db_manager.disconnect()
            logger.info("Database disconnected successfully")
    except Exception as e:
        logger.error(f"Error during database disconnection: {str(e)}", exc_info=True)
    
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """Create and configure FastAPI application instance"""
    
    app = FastAPI(
        title=settings.app_name,
        description="REST API service for Bulos language translation and vocabulary management",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.debug,
        lifespan=lifespan  # Use lifespan context manager
    )
    
    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Request logging middleware with timing information
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests with timing and status information"""
        start_time = time.time()
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log request completion
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"Status: {response.status_code} Duration: {process_time:.3f}s"
            )
            
            # Add timing header to response
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Calculate processing time even on error
            process_time = time.time() - start_time
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"Duration: {process_time:.3f}s Error: {str(e)}",
                exc_info=True
            )
            raise
    
    # Global exception handler for unhandled 500 errors
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions with consistent error response"""
        logger.error(
            f"Unhandled exception in {request.method} {request.url.path}: {str(exc)}",
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred while processing your request",
                "status_code": 500
            }
        )
    
    # HTTPException handler for 4xx errors
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent error response format"""
        logger.warning(
            f"HTTP exception in {request.method} {request.url.path}: "
            f"Status {exc.status_code} - {exc.detail}"
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "message": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check endpoint"""
        return {
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.app_version
        }
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint"""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health"
        }
    
    # Register routers
    app.include_router(vocabulary_router)
    app.include_router(translation_router)
    app.include_router(history_router)
    app.include_router(health_router)
    app.include_router(dictionary_router)
    app.include_router(evaluation_router)
    app.include_router(alphabet_router)
    
    return app


# Create application instance
app = create_application()
