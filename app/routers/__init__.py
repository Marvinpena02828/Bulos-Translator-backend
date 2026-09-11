"""API routers package"""
from app.routers.vocabulary import router as vocabulary_router
from app.routers.translation import router as translation_router
from app.routers.history import router as history_router
from app.routers.health import router as health_router
from app.routers.dictionary import router as dictionary_router
from app.routers.evaluation import router as evaluation_router
from app.routers.alphabet import router as alphabet_router

__all__ = ["vocabulary_router", "translation_router", "history_router", "health_router", "dictionary_router", "evaluation_router", "alphabet_router"]
