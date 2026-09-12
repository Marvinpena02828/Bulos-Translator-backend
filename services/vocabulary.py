"""Vocabulary management service — in-memory store (no database)."""
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from models.schemas import VocabularyCreate
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Module-level in-memory store: { device_id: { vocab_id: dict } }
_store: Dict[str, Dict[str, Dict[str, Any]]] = {}


class VocabularyService:
    """Manages per-device vocabulary items in memory."""

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #

    async def create_vocabulary(
        self,
        device_id: str,
        vocab_data: VocabularyCreate,
    ) -> Dict[str, Any]:
        device_store = _store.setdefault(device_id, {})

        # Duplicate check (same word + language pair)
        for item in device_store.values():
            if (
                item["word"] == vocab_data.word
                and item["source_language"] == vocab_data.source_language
                and item["target_language"] == vocab_data.target_language
            ):
                raise ValueError(
                    f"Vocabulary item '{vocab_data.word}' already exists for this language pair"
                )

        vocab_id = str(uuid.uuid4())
        now = datetime.utcnow()
        document = {
            "id": vocab_id,
            "device_id": device_id,
            "word": vocab_data.word,
            "translation": vocab_data.translation,
            "source_language": vocab_data.source_language,
            "target_language": vocab_data.target_language,
            "notes": vocab_data.notes,
            "created_at": now,
            "updated_at": now,
        }
        device_store[vocab_id] = document
        logger.info(f"Vocabulary created: {vocab_id} for device {device_id}")
        return {"id": vocab_id, "message": "Vocabulary created successfully"}

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    async def get_vocabulary(
        self,
        device_id: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        page: int = 0,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        items = list(_store.get(device_id, {}).values())

        if source_language:
            items = [i for i in items if i["source_language"] == source_language]
        if target_language:
            items = [i for i in items if i["target_language"] == target_language]

        # Sort newest first
        items.sort(key=lambda i: i["created_at"], reverse=True)

        total = len(items)
        start = page * page_size
        page_items = items[start : start + page_size]

        return {
            "items": page_items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    async def update_vocabulary(
        self,
        vocab_id: str,
        device_id: str,
        update_data: VocabularyCreate,
    ) -> Dict[str, str]:
        device_store = _store.get(device_id, {})
        item = device_store.get(vocab_id)

        if not item:
            raise ValueError(
                "Vocabulary item not found or you don't have permission to update it"
            )

        item.update(
            {
                "word": update_data.word,
                "translation": update_data.translation,
                "source_language": update_data.source_language,
                "target_language": update_data.target_language,
                "notes": update_data.notes,
                "updated_at": datetime.utcnow(),
            }
        )
        logger.info(f"Vocabulary updated: {vocab_id}")
        return {"message": "Vocabulary updated successfully"}

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #

    async def delete_vocabulary(
        self,
        vocab_id: str,
        device_id: str,
    ) -> Dict[str, str]:
        device_store = _store.get(device_id, {})

        if vocab_id not in device_store:
            raise ValueError(
                "Vocabulary item not found or you don't have permission to delete it"
            )

        del device_store[vocab_id]
        logger.info(f"Vocabulary deleted: {vocab_id}")
        return {"message": "Vocabulary deleted successfully"}
