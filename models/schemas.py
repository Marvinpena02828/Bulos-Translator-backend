"""Pydantic data models and request/response schemas for API endpoints"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom Pydantic type for MongoDB ObjectId validation."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


# Supported language codes
SUPPORTED_LANGUAGES = ['bul', 'en', 'tl']  # Bulos, English, Tagalog



# Vocabulary Models

class VocabularyCreate(BaseModel):
    """Schema for creating a new vocabulary item."""
    
    word: str = Field(..., min_length=1, max_length=200, description="Word or phrase in source language")
    translation: str = Field(..., min_length=1, max_length=200, description="Translation in target language")
    source_language: str = Field(..., min_length=2, max_length=5, description="Source language code")
    target_language: str = Field(..., min_length=2, max_length=5, description="Target language code")
    notes: Optional[str] = Field(None, max_length=1000, description="Optional notes or context")
    
    @validator('source_language', 'target_language')
    def validate_language_code(cls, v):
        """Validate that language code is in supported languages list."""
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Must be one of {SUPPORTED_LANGUAGES}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "word": "kumusta",
                "translation": "hello",
                "source_language": "tl",
                "target_language": "en",
                "notes": "Common greeting"
            }
        }


class VocabularyResponse(BaseModel):
    """Schema for vocabulary item response."""
    
    id: str = Field(..., alias="_id", description="Vocabulary item ID")
    user_id: str = Field(..., description="ID of user who owns this vocabulary")
    word: str = Field(..., description="Word or phrase in source language")
    translation: str = Field(..., description="Translation in target language")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    notes: Optional[str] = Field(None, description="Optional notes or context")
    created_at: datetime = Field(..., description="Timestamp when vocabulary was created")
    updated_at: datetime = Field(..., description="Timestamp when vocabulary was last updated")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "user_id": "user123",
                "word": "kumusta",
                "translation": "hello",
                "source_language": "tl",
                "target_language": "en",
                "notes": "Common greeting",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }



# Translation Models
class TranslationRequest(BaseModel):
    """Schema for translation request."""
    
    text: str = Field(..., min_length=1, max_length=500, description="Text to translate")
    source_language: str = Field(..., min_length=2, max_length=5, description="Source language code")
    target_language: str = Field(..., min_length=2, max_length=5, description="Target language code")
    
    @validator('source_language', 'target_language')
    def validate_language_code(cls, v):
        """Validate that language code is in supported languages list."""
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Must be one of {SUPPORTED_LANGUAGES}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "source_language": "en",
                "target_language": "tl"
            }
        }


class TranslationResponse(BaseModel):
    """Schema for translation response."""
    
    original_text: str = Field(..., description="Original text that was translated")
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Translation confidence score (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "original_text": "Hello, how are you?",
                "translated_text": "Kamusta ka?",
                "source_language": "en",
                "target_language": "tl",
                "confidence": 0.95
            }
        }



# Speech Processing Models
class SpeechProcessRequest(BaseModel):
    """Schema for speech processing request (handled as file upload in endpoint)."""
    
    language: str = Field(..., min_length=2, max_length=5, description="Language of the audio")
    
    @validator('language')
    def validate_language_code(cls, v):
        """Validate that language code is in supported languages list."""
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Must be one of {SUPPORTED_LANGUAGES}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "language": "en"
            }
        }


class SpeechProcessResponse(BaseModel):
    """Schema for speech processing response."""
    
    transcribed_text: str = Field(..., description="Transcribed text from audio")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Transcription confidence score (0-1)")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "transcribed_text": "Hello, how are you?",
                "confidence": 0.92,
                "processing_time": 1.23
            }
        }



# History Models
class HistoryRecord(BaseModel):
    """Schema for a single history record."""
    
    id: str = Field(..., alias="_id", description="History record ID")
    user_id: str = Field(..., description="ID of user who performed the action")
    action_type: str = Field(..., description="Type of action (e.g., 'vocabulary_create', 'translation')")
    resource_id: Optional[str] = Field(None, description="ID of the resource affected by the action")
    resource_type: Optional[str] = Field(None, description="Type of resource (e.g., 'vocabulary', 'translation')")
    outcome: str = Field(..., description="Outcome of the action (e.g., 'success', 'error')")
    timestamp: datetime = Field(..., description="Timestamp when action was performed")
    details: Optional[dict] = Field(None, description="Additional details about the action")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439014",
                "user_id": "user123",
                "action_type": "vocabulary_create",
                "resource_id": "507f1f77bcf86cd799439011",
                "resource_type": "vocabulary",
                "outcome": "success",
                "timestamp": "2024-01-15T10:30:00Z",
                "details": {
                    "word": "kumusta",
                    "translation": "hello"
                }
            }
        }


class HistoryResponse(BaseModel):
    """Schema for paginated history response."""
    
    records: List[HistoryRecord] = Field(..., description="List of history records")
    total: int = Field(..., ge=0, description="Total number of records matching the query")
    page: int = Field(..., ge=0, description="Current page number")
    page_size: int = Field(..., ge=1, description="Number of records per page")
    
    class Config:
        schema_extra = {
            "example": {
                "records": [
                    {
                        "_id": "507f1f77bcf86cd799439014",
                        "user_id": "user123",
                        "action_type": "vocabulary_create",
                        "resource_id": "507f1f77bcf86cd799439011",
                        "resource_type": "vocabulary",
                        "outcome": "success",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "details": {"word": "kumusta"}
                    }
                ],
                "total": 50,
                "page": 0,
                "page_size": 20
            }
        }



# Common Response Model
class MessageResponse(BaseModel):
    """Schema for simple message responses."""
    
    message: str = Field(..., description="Response message")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Operation completed successfully"
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "message": "Invalid language code provided",
                "status_code": 400
            }
        }
