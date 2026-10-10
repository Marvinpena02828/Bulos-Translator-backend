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
    audio_router,
    stt_router,
)

configure_logging()
logger = get_logger(__name__)

# Global translation service instance (shared across requests)
_translation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise translation service and OPUS-MT models. Shutdown: nothing to close."""
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

    # Warm-load OPUS-MT CTranslate2 models at startup so the first en↔tl
    # request doesn't pay the cold-load cost.
    try:
        logger.info("Loading OPUS-MT CTranslate2 models (en↔tl)...")
        from services.translation import _get_opus
        opus = _get_opus()
        if opus is not None:
            logger.info(
                f"OPUS-MT models loaded — available directions: {opus.available_directions}"
            )
        else:
            logger.warning(
                "OPUS-MT models not available — en↔tl will rely on Google Translate"
            )
    except Exception as e:
        logger.warning(f"OPUS-MT model loading failed at startup: {e}")
        logger.warning("en↔tl translation will fall back to Google Translate only")

    # Initialize Audio Storage (connects to bulos_audio database)
    if settings.mongodb_uri:
        try:
            logger.info("Initializing Audio Storage Service (bulos_audio)...")
            from services.audio_storage import initialize_audio_storage
            audio_storage = initialize_audio_storage(
                mongodb_url=settings.mongodb_uri,
                database_name=settings.mongodb_audio_db,
            )
            await audio_storage.ensure_indexes()

            # Sync existing GridFS audio files into training pairs
            sync_stats = await audio_storage.sync_existing_audio_to_training_pairs()
            logger.info(
                f"Audio training data: {sync_stats['total_files']} files, "
                f"{sync_stats['newly_created']} new pairs synced"
            )
        except Exception as e:
            logger.warning(f"Audio Storage initialization failed: {e}")
            logger.warning("STT training data endpoints may not be available")

    # Load STT model
    if settings.stt_enabled:
        try:
            logger.info(f"Loading STT model: {settings.stt_model_path}...")
            from services.stt_service import initialize_stt_service
            initialize_stt_service(
                model_size_or_path=settings.stt_model_path,
                device=settings.stt_device,
                compute_type=settings.stt_compute_type,
            )
            logger.info("STT model loaded successfully")
        except Exception as e:
            logger.warning(f"STT model loading failed: {e}")
            logger.warning("Transcription endpoints will return 503 until model is loaded")

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

    @app.get("/debug/opus", tags=["Debug"])
    async def debug_opus():
        """Shows OPUS-MT CTranslate2 model load status."""
        from services.translation import _get_opus
        opus = _get_opus()
        if opus is None:
            return {
                "status": "unavailable",
                "available_directions": [],
                "note": "OPUS-MT failed to load — check startup logs.",
            }
        return {
            "status": "loaded",
            "available_directions": opus.available_directions,
            "models_loaded": len(opus.available_directions),
            "per_key_errors": getattr(opus, "per_key_errors", {}),
        }

    @app.get("/debug/stt", tags=["Debug"])
    async def debug_stt():
        """Shows STT model status and training data stats."""
        from services.stt_service import get_model_info
        info = get_model_info()
        try:
            from services.audio_storage import get_audio_storage
            storage = get_audio_storage()
            info["training_pairs_total"] = await storage.count_training_pairs("bul")
            info["training_pairs_validated"] = await storage.count_training_pairs("bul", validated_only=True)
            info["total_audio_files"] = await storage.count_audio_files()
        except Exception:
            info["audio_storage"] = "not_available"
        return info

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
    app.include_router(audio_router)
    app.include_router(stt_router)

    return app


app = create_application()
