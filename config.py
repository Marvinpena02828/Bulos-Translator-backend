"""Configuration management using Pydantic BaseSettings"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Application Settings
    app_name: str = Field(default="Bulos Translator API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # MongoDB Settings
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URL")
    mongodb_database: str = Field(default="bulos_translator", description="MongoDB database name")
    mongodb_max_pool_size: int = Field(default=50, description="MongoDB connection pool max size")
    mongodb_min_pool_size: int = Field(default=10, description="MongoDB connection pool min size")
    
    # Authentication Settings
    secret_key: str = Field(..., description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration time in minutes")
    
    # DeepSpeech Settings
    deepspeech_model_path: str = Field(..., description="Path to DeepSpeech model file")
    deepspeech_scorer_path: str = Field(..., description="Path to DeepSpeech scorer file")
    
    # Translation Model Settings
    translation_models_dir: str = Field(default="./models/translation", description="Directory containing translation models")
    
    # File Upload Settings
    max_audio_file_size: int = Field(default=10*1024*1024, description="Maximum audio file size in bytes (default 10MB)")
    max_audio_duration_seconds: int = Field(default=60, description="Maximum audio duration in seconds")
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_file_path: str = Field(default="./logs/app.log", description="Log file path")
    log_max_bytes: int = Field(default=10*1024*1024, description="Maximum log file size before rotation")
    log_backup_count: int = Field(default=5, description="Number of backup log files to keep")
    
    # Performance Settings
    max_concurrent_requests: int = Field(default=100, description="Maximum concurrent API requests")
    translation_timeout_seconds: int = Field(default=3, description="Translation operation timeout")
    speech_processing_timeout_seconds: int = Field(default=10, description="Speech processing timeout")
    
    # CORS Settings
    cors_origins: list = Field(default=["*"], description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_allow_methods: list = Field(default=["GET", "POST", "PUT", "DELETE"], description="Allowed HTTP methods")
    cors_allow_headers: list = Field(default=["*"], description="Allowed HTTP headers")
    
    # Supported Languages
    supported_languages: list = Field(
        default=["bul", "en", "tl"],
        description="List of supported language codes (bul=Bulos, en=English, tl=Tagalog)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow the app to work without .env file by using defaults
        env_ignore_empty = True
        extra = "ignore"


# Singleton instance - gracefully handle missing .env file
try:
    settings = Settings()
except Exception:
    # If .env file doesn't exist, create with minimal defaults
    import os
    os.environ.setdefault('MONGODB_URL', 'mongodb://localhost:27017')
    os.environ.setdefault('SECRET_KEY', 'change-this-secret-key-in-production')
    os.environ.setdefault('DEEPSPEECH_MODEL_PATH', './models/deepspeech.pbmm')
    os.environ.setdefault('DEEPSPEECH_SCORER_PATH', './models/deepspeech.scorer')
    settings = Settings()
