"""API routers package"""
from app.routers.auth import router as auth_router
from app.routers.vocabulary import router as vocabulary_router
from app.routers.translation import router as translation_router
from app.routers.speech import router as speech_router
from app.routers.history import router as history_router
from app.routers.health import router as health_router
from app.routers.dictionary import router as dictionary_router

__all__ = ["auth_router", "vocabulary_router", "translation_router", "speech_router", "history_router", "health_router", "dictionary_router"]
