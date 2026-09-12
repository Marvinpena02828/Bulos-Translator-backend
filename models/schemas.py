"""Pydantic data models and request/response schemas for API endpoints"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime


# Supported language codes
SUPPORTED_LANGUAGES = ['bul', 'en', 'tl']  # Bulos, English, Tagalog


# ── Vocabulary ────────────────────────────────────────────────────────────────

class VocabularyCreate(BaseModel):
    """Schema for creating a new vocabulary item."""

    word: str = Field(..., min_length=1, max_length=200)
    translation: str = Field(..., min_length=1, max_length=200)
    source_language: str = Field(..., min_length=2, max_length=5)
    target_language: str = Field(..., min_length=2, max_length=5)
    notes: Optional[str] = Field(None, max_length=1000)

    @validator('source_language', 'target_language')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Must be one of {SUPPORTED_LANGUAGES}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "word": "kumusta", "translation": "hello",
                "source_language": "tl", "target_language": "en",
                "notes": "Common greeting"
            }
        }


class VocabularyResponse(BaseModel):
    """Schema for vocabulary item response."""

    id: str = Field(..., description="Vocabulary item ID")
    device_id: str
    word: str
    translation: str
    source_language: str
    target_language: str
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "device_id": "550e8400-e29b-41d4-a716-446655440001",
                "word": "kumusta", "translation": "hello",
                "source_language": "tl", "target_language": "en",
                "notes": "Common greeting",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


# ── Translation ───────────────────────────────────────────────────────────────

class TranslationRequest(BaseModel):
    """Schema for translation request."""

    text: str = Field(..., min_length=1, max_length=500)
    source_language: str = Field(..., min_length=2, max_length=5)
    target_language: str = Field(..., min_length=2, max_length=5)

    @validator('source_language', 'target_language')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Must be one of {SUPPORTED_LANGUAGES}")
        return v

    class Config:
        json_schema_extra = {
            "example": {"text": "Hello, how are you?",
                        "source_language": "en", "target_language": "tl"}
        }


class TranslationResponse(BaseModel):
    """Schema for translation response."""

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    intermediate_language: Optional[str] = None
    translation_method: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "Hello, how are you?",
                "translated_text": "Kamusta ka?",
                "source_language": "en", "target_language": "tl",
                "confidence": 0.95,
                "intermediate_language": None,
                "translation_method": "google"
            }
        }


# ── Common ────────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str

    class Config:
        json_schema_extra = {"example": {"message": "Operation completed successfully"}}


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int

    class Config:
        json_schema_extra = {
            "example": {"error": "Validation Error",
                        "message": "Invalid language code provided",
                        "status_code": 400}
        }


# ── Dictionary ────────────────────────────────────────────────────────────────

class DictionaryEntry(BaseModel):
    """Schema for dictionary entry from lexicon."""

    id: Optional[str] = Field(None, description="Entry ID")
    bulos: str
    filipino: str
    english: str
    category: str
    category_key: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "bulos": "ulù", "filipino": "ulo", "english": "head",
                "category": "BAHAGI NI LAWES (Parts of the Body)",
                "category_key": "parts-of-body"
            }
        }


class CategoryInfo(BaseModel):
    category: str
    category_key: str
    entry_count: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {"category": "BAHAGI NI LAWES (Parts of the Body)",
                        "category_key": "parts-of-body", "entry_count": 45}
        }


class ImportResult(BaseModel):
    total_entries: int = Field(..., ge=0)
    inserted: int = Field(..., ge=0)
    skipped: int = Field(..., ge=0)
    errors: int = Field(..., ge=0)
    duration: float = Field(..., ge=0)
    error_details: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {"total_entries": 450, "inserted": 445, "skipped": 5,
                        "errors": 0, "duration": 2.34, "error_details": []}
        }


# ── Alphabet ──────────────────────────────────────────────────────────────────

class AlphabetExample(BaseModel):
    bulos: str
    english: str
    filipino: str

    class Config:
        json_schema_extra = {
            "example": {"bulos": "atis", "english": "sugar apple", "filipino": "atis"}
        }


class AlphabetEntry(BaseModel):
    id: Optional[str] = Field(None, description="Alphabet entry ID")
    letter: str
    position: int = Field(..., ge=1)
    initial_examples: List[AlphabetExample] = Field(default_factory=list)
    middle_examples: List[AlphabetExample] = Field(default_factory=list)
    final_examples: List[AlphabetExample] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "letter": "Aa", "position": 1,
                "initial_examples": [{"bulos": "atis", "english": "sugar apple", "filipino": "atis"}],
                "middle_examples": [], "final_examples": []
            }
        }


class DictionaryLookupRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=200)
    source_language: str = Field(default="bul", min_length=2, max_length=5)

    @validator('source_language')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Must be one of {SUPPORTED_LANGUAGES}")
        return v

    class Config:
        json_schema_extra = {"example": {"word": "ulù", "source_language": "bul"}}


class DictionarySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    language: str = Field(default="all")
    limit: int = Field(default=50, ge=1, le=200)

    @validator('language')
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES + ['all']:
            raise ValueError(f"Invalid language: {v}. Must be one of {SUPPORTED_LANGUAGES + ['all']}")
        return v

    class Config:
        json_schema_extra = {"example": {"query": "head", "language": "all", "limit": 50}}


# ── Evaluation ────────────────────────────────────────────────────────────────

class ISO25010Evaluation(BaseModel):
    functional_suitability: int = Field(..., ge=1, le=5)
    usability: int = Field(..., ge=1, le=5)
    performance_efficiency: int = Field(..., ge=1, le=5)
    reliability: int = Field(..., ge=1, le=5)
    maintainability: int = Field(..., ge=1, le=5)
    portability: int = Field(..., ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "functional_suitability": 5, "usability": 4,
                "performance_efficiency": 5, "reliability": 4,
                "maintainability": 4, "portability": 5,
                "comments": "The system is very helpful for learning Bulos!"
            }
        }


class TAMEvaluation(BaseModel):
    perceived_usefulness: int = Field(..., ge=1, le=5)
    perceived_ease_of_use: int = Field(..., ge=1, le=5)
    behavioral_intention: int = Field(..., ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "perceived_usefulness": 5, "perceived_ease_of_use": 4,
                "behavioral_intention": 5,
                "comments": "I will definitely use this app!"
            }
        }


class EvaluationResponse(BaseModel):
    evaluation_id: str
    message: str
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "evaluation_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "ISO/IEC 25010 evaluation submitted successfully",
                "timestamp": "2026-07-10T10:30:00Z"
            }
        }


class EvaluationResultsResponse(BaseModel):
    total_evaluations: int = Field(..., ge=0)
    iso25010_results: Dict[str, float]
    tam_results: Dict[str, float]
    iso25010_interpretation: str
    tam_interpretation: str

    class Config:
        json_schema_extra = {
            "example": {
                "total_evaluations": 25,
                "iso25010_results": {"functional_suitability": 4.52, "overall_mean": 4.37},
                "tam_results": {"perceived_usefulness": 4.68, "overall_mean": 4.61},
                "iso25010_interpretation": "Acceptable",
                "tam_interpretation": "Highly Acceptable"
            }
        }
