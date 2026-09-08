"""Pydantic data models and request/response schemas for API endpoints"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
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
        json_schema_extra = {
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
    device_id: str = Field(..., description="Device ID that owns this vocabulary")
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
                "device_id": "550e8400-e29b-41d4-a716-446655440000",
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
        json_schema_extra = {
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
    intermediate_language: Optional[str] = Field(None, description="Intermediate language used (e.g., 'tl' when en→tl→bul only reached tl)")
    translation_method: Optional[str] = Field(None, description="Method used for translation (e.g., 'dictionary', 'google', 'two-step')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "Hello, how are you?",
                "translated_text": "Kamusta ka?",
                "source_language": "en",
                "target_language": "tl",
                "confidence": 0.95,
                "intermediate_language": None,
                "translation_method": "google"
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
        json_schema_extra = {
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
        json_schema_extra = {
            "example": {
                "transcribed_text": "Hello, how are you?",
                "confidence": 0.92,
                "processing_time": 1.23
            }
        }


class SpeechTranslateResponse(BaseModel):
    """Schema for speech transcription + translation response."""
    
    transcribed_text: str = Field(..., description="Transcribed text from audio (in detected language)")
    translated_text: str = Field(..., description="Translated text (in target language)")
    detected_language: str = Field(..., description="Detected source language code")
    target_language: str = Field(..., description="Target language code")
    transcription_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Transcription confidence score (0-1)")
    translation_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Translation confidence score (0-1)")
    processing_time: float = Field(..., ge=0, description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transcribed_text": "Hello, how are you?",
                "translated_text": "Kamusta ka?",
                "detected_language": "en",
                "target_language": "tl",
                "transcription_confidence": 0.92,
                "translation_confidence": 0.95,
                "processing_time": 2.45
            }
        }



# History Models
class HistoryRecord(BaseModel):
    """Schema for a single history record."""
    
    id: str = Field(..., alias="_id", description="History record ID")
    device_id: str = Field(..., description="Device ID that performed the action")
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
                "device_id": "550e8400-e29b-41d4-a716-446655440000",
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
        json_schema_extra = {
            "example": {
                "records": [
                    {
                        "_id": "507f1f77bcf86cd799439014",
                        "device_id": "550e8400-e29b-41d4-a716-446655440000",
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
        json_schema_extra = {
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
        json_schema_extra = {
            "example": {
                "error": "Validation Error",
                "message": "Invalid language code provided",
                "status_code": 400
            }
        }



# Dictionary Models

class DictionaryEntry(BaseModel):
    """Schema for dictionary entry from lexicon."""
    
    id: Optional[str] = Field(None, alias="_id", description="Dictionary entry ID")
    bulos: str = Field(..., description="Bulos word/phrase")
    filipino: str = Field(..., description="Filipino translation")
    english: str = Field(..., description="English translation")
    category: str = Field(..., description="Full category name")
    category_key: str = Field(..., description="Normalized category key (kebab-case)")
    created_at: Optional[datetime] = Field(None, description="Entry creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Entry last update timestamp")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
    
    class Config:
        populate_by_name = True
        from_attributes = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "bulos": "ulù",
                "filipino": "ulo",
                "english": "head",
                "category": "BAHAGI NI LAWES (Parts of the Body)",
                "category_key": "parts-of-body",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "metadata": {
                    "source": "dictionary.json",
                    "import_version": "1.0"
                }
            }
        }


class CategoryInfo(BaseModel):
    """Schema for category information."""
    
    category: str = Field(..., description="Full category name")
    category_key: str = Field(..., description="Normalized category key")
    entry_count: int = Field(..., ge=0, description="Number of entries in this category")
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "BAHAGI NI LAWES (Parts of the Body)",
                "category_key": "parts-of-body",
                "entry_count": 45
            }
        }


class ImportResult(BaseModel):
    """Schema for dictionary import operation result."""
    
    total_entries: int = Field(..., ge=0, description="Total number of entries processed")
    inserted: int = Field(..., ge=0, description="Number of entries successfully inserted")
    skipped: int = Field(..., ge=0, description="Number of entries skipped (duplicates)")
    errors: int = Field(..., ge=0, description="Number of errors encountered")
    duration: float = Field(..., ge=0, description="Import duration in seconds")
    error_details: List[str] = Field(default_factory=list, description="List of error messages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_entries": 450,
                "inserted": 445,
                "skipped": 5,
                "errors": 0,
                "duration": 2.34,
                "error_details": []
            }
        }


class AlphabetExample(BaseModel):
    """Schema for a position-based alphabet example."""

    bulos: str = Field(..., description="Bulos example word")
    english: str = Field(..., description="English translation")
    filipino: str = Field(..., description="Filipino translation")

    class Config:
        json_schema_extra = {
            "example": {
                "bulos": "atis",
                "english": "sugar apple",
                "filipino": "atis"
            }
        }


class AlphabetEntry(BaseModel):
    """Schema for a single alphabet entry with position-based examples."""

    id: Optional[str] = Field(None, alias="_id", description="Alphabet entry ID")
    letter: str = Field(..., description="The alphabet letter (e.g., 'Aa', 'Bb', 'Ng ng')")
    position: int = Field(..., ge=1, description="Position of the letter in the alphabet")
    initial_examples: List[AlphabetExample] = Field(
        default_factory=list, description="Examples where the letter appears initially"
    )
    middle_examples: List[AlphabetExample] = Field(
        default_factory=list, description="Examples where the letter appears in the middle"
    )
    final_examples: List[AlphabetExample] = Field(
        default_factory=list, description="Examples where the letter appears finally"
    )
    created_at: Optional[datetime] = Field(None, description="Entry creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Entry last update timestamp")

    class Config:
        populate_by_name = True
        from_attributes = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "letter": "Aa",
                "position": 1,
                "initial_examples": [
                    {"bulos": "atis", "english": "sugar apple", "filipino": "atis"}
                ],
                "middle_examples": [],
                "final_examples": [],
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


class DictionaryLookupRequest(BaseModel):
    """Schema for dictionary lookup request."""
    
    word: str = Field(..., min_length=1, max_length=200, description="Word to lookup")
    source_language: str = Field(
        default="bul",
        min_length=2,
        max_length=5,
        description="Source language code (bul, tl, en)"
    )
    
    @validator('source_language')
    def validate_language_code(cls, v):
        """Validate that language code is in supported languages list."""
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Must be one of {SUPPORTED_LANGUAGES}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "word": "ulù",
                "source_language": "bul"
            }
        }


class DictionarySearchRequest(BaseModel):
    """Schema for dictionary search request."""
    
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    language: str = Field(
        default="all",
        description="Language to search (bul, tl, en, or 'all' for all languages)"
    )
    limit: int = Field(default=50, ge=1, le=200, description="Maximum number of results")
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language parameter."""
        valid_languages = SUPPORTED_LANGUAGES + ['all']
        if v not in valid_languages:
            raise ValueError(f"Invalid language: {v}. Must be one of {valid_languages}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "head",
                "language": "all",
                "limit": 50
            }
        }



# Evaluation Models (ISO/IEC 25010 & TAM)

class ISO25010Evaluation(BaseModel):
    """Schema for ISO/IEC 25010 Software Quality evaluation submission."""
    
    functional_suitability: int = Field(
        ..., ge=1, le=5,
        description="Rating for functional suitability (1-5 Likert scale)"
    )
    usability: int = Field(
        ..., ge=1, le=5,
        description="Rating for usability (1-5 Likert scale)"
    )
    performance_efficiency: int = Field(
        ..., ge=1, le=5,
        description="Rating for performance efficiency (1-5 Likert scale)"
    )
    reliability: int = Field(
        ..., ge=1, le=5,
        description="Rating for reliability (1-5 Likert scale)"
    )
    maintainability: int = Field(
        ..., ge=1, le=5,
        description="Rating for maintainability (1-5 Likert scale)"
    )
    portability: int = Field(
        ..., ge=1, le=5,
        description="Rating for portability (1-5 Likert scale)"
    )
    comments: Optional[str] = Field(
        None, max_length=1000,
        description="Optional comments or feedback"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "functional_suitability": 5,
                "usability": 4,
                "performance_efficiency": 5,
                "reliability": 4,
                "maintainability": 4,
                "portability": 5,
                "comments": "The system is very helpful for learning Bulos!"
            }
        }


class TAMEvaluation(BaseModel):
    """Schema for Technology Acceptance Model (TAM) evaluation submission."""
    
    perceived_usefulness: int = Field(
        ..., ge=1, le=5,
        description="Rating for perceived usefulness (1-5 Likert scale)"
    )
    perceived_ease_of_use: int = Field(
        ..., ge=1, le=5,
        description="Rating for perceived ease of use (1-5 Likert scale)"
    )
    behavioral_intention: int = Field(
        ..., ge=1, le=5,
        description="Rating for behavioral intention to use (1-5 Likert scale)"
    )
    comments: Optional[str] = Field(
        None, max_length=1000,
        description="Optional comments or feedback"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "perceived_usefulness": 5,
                "perceived_ease_of_use": 4,
                "behavioral_intention": 5,
                "comments": "I will definitely use this app to communicate with Dumagat speakers!"
            }
        }


class EvaluationResponse(BaseModel):
    """Schema for evaluation submission response."""
    
    evaluation_id: str = Field(..., description="ID of the submitted evaluation")
    message: str = Field(..., description="Success message")
    timestamp: datetime = Field(..., description="Submission timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "evaluation_id": "507f1f77bcf86cd799439011",
                "message": "ISO/IEC 25010 evaluation submitted successfully",
                "timestamp": "2026-07-10T10:30:00Z"
            }
        }


class EvaluationResultsResponse(BaseModel):
    """Schema for aggregated evaluation results."""
    
    total_evaluations: int = Field(..., ge=0, description="Total number of evaluations")
    iso25010_results: Dict[str, float] = Field(..., description="ISO/IEC 25010 mean scores")
    tam_results: Dict[str, float] = Field(..., description="TAM mean scores")
    iso25010_interpretation: str = Field(..., description="Interpretation of ISO/IEC 25010 overall mean")
    tam_interpretation: str = Field(..., description="Interpretation of TAM overall mean")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_evaluations": 25,
                "iso25010_results": {
                    "functional_suitability": 4.52,
                    "usability": 4.32,
                    "performance_efficiency": 4.48,
                    "reliability": 4.20,
                    "maintainability": 4.12,
                    "portability": 4.60,
                    "overall_mean": 4.37
                },
                "tam_results": {
                    "perceived_usefulness": 4.68,
                    "perceived_ease_of_use": 4.44,
                    "behavioral_intention": 4.72,
                    "overall_mean": 4.61
                },
                "iso25010_interpretation": "Acceptable",
                "tam_interpretation": "Highly Acceptable"
            }
        }
