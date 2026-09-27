"""FastAPI application entry point with middleware configuration"""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from utils.logging_config import configure_logging, get_logger
from app.routers import (
    vocabulary_router,
    translation_router,
    history_router,
    health_router,
    dictionary_router,
    evaluation_router,
    alphabet_router,
)

configure_logging()
logger = get_logger(__name__)

# Global translation service instance (shared across requests)
_translation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise translation service and LSTM models. Shutdown: nothing to close."""
    global _translation_service

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"CORS origins: {settings.cors_origins}")

    try:
        logger.info("Initializing translation service...")
        from services.translation import TranslationService
        _translation_service = TranslationService()
        await _translation_service.initialize()
        phrase_count = sum(len(v) for v in _translation_service.phrase_index.values())
        logger.info(f"Translation service initialized — {phrase_count} indexed phrases")
        if phrase_count == 0:
            logger.error("CRITICAL: phrase_index is EMPTY after initialization — translation will not work!")
    except Exception as e:
        logger.error(f"Translation service initialization FAILED: {e}", exc_info=True)
        logger.warning("Translation endpoints may not be available")

    # Warm-load LSTM models at startup so the first translation request
    # doesn't pay the model-loading cost (~5-10s per model × 6 models).
    try:
        logger.info("Loading LSTM translation models...")
        from services.translation import _get_lstm
        lstm = _get_lstm()
        if lstm is not None:
            logger.info(
                f"LSTM models loaded — available directions: {lstm.available_directions}"
            )
        else:
            logger.warning(
                "LSTM models not available — translation will use dictionary/Google only"
            )
    except Exception as e:
        logger.warning(f"LSTM model loading failed at startup: {e}")
        logger.warning("Translation will continue without LSTM fallback")

    logger.info("Application startup complete")

    yield

    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    app = FastAPI(
        title=settings.app_name,
        description="REST API service for Bulos language translation and vocabulary management",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"Status: {response.status_code} Duration: {process_time:.3f}s"
            )
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"Duration: {process_time:.3f}s Error: {str(e)}",
                exc_info=True,
            )
            raise

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception in {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "status_code": 500,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(
            f"HTTP exception in {request.method} {request.url.path}: "
            f"Status {exc.status_code} - {exc.detail}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "message": exc.detail,
                     "status_code": exc.status_code},
        )

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "healthy", "service": settings.app_name,
                "version": settings.app_version}

    @app.get("/debug/translation", tags=["Debug"])
    async def debug_translation():
        """Shows translation service state — use to verify phrase index is loaded."""
        if _translation_service is None:
            return {"status": "not_initialized", "phrase_count": 0}
        phrase_count = sum(len(v) for v in _translation_service.phrase_index.values())
        sample_keys = list(_translation_service.phrase_index.keys())[:5]
        return {
            "status": "initialized",
            "phrase_count": phrase_count,
            "sample_keys": sample_keys,
        }

    @app.get("/debug/lstm", tags=["Debug"])
    async def debug_lstm():
        """Shows LSTM model load status — use to diagnose translation failures on Render."""
        from services.translation import _get_lstm
        lstm = _get_lstm()
        if lstm is None:
            return {
                "status": "unavailable",
                "available_directions": [],
                "note": "LSTM failed to load — check startup logs for the error.",
            }
        return {
            "status": "loaded",
            "available_directions": lstm.available_directions,
            "models_loaded": len(lstm.available_directions),
            "last_load_error": getattr(lstm, "last_load_error", ""),
            "per_key_errors": getattr(lstm, "per_key_errors", {}),
        }

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": f"Welcome to {settings.app_name}",
                "version": settings.app_version,
                "docs": "/docs", "health": "/health"}

    app.include_router(vocabulary_router)
    app.include_router(translation_router)
    app.include_router(history_router)
    app.include_router(health_router)
    app.include_router(dictionary_router)
    app.include_router(evaluation_router)
    app.include_router(alphabet_router)

    return app


app = create_application()
