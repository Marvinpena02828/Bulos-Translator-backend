"""
Bulos Translator Backend - Main Application Entry Point

This module serves as the entry point for running the FastAPI application.
It imports the configured app instance and can be used with uvicorn to start the server.

Usage:
    Development (with auto-reload):
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
    
    Production (with multiple workers):
        uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    
    With gunicorn (production):
        gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

from app.main import app

# The app instance is imported from app.main where it's created with all
# configurations, middleware, and routers. The lifespan context manager
# handles all service initialization (database, translation) and cleanup on shutdown.

if __name__ == "__main__":
    import uvicorn
    from config import settings
    
    # Run the application with uvicorn
    # This is primarily for development. In production, use the uvicorn CLI
    # or a production-grade ASGI server like gunicorn with uvicorn workers.
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,  # Auto-reload in debug mode
        log_level=settings.log_level.lower(),
        access_log=True
    )
