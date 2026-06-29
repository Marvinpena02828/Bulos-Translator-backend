"""Vocabulary management service with CRUD operations and history tracking"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from bson import ObjectId

from services.database import DatabaseManager
from models.schemas import VocabularyCreate, VocabularyResponse
from utils.logging_config import get_logger

logger = get_logger(__name__)


class VocabularyService:
    """Service for managing vocabulary items with user isolation and history tracking"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize vocabulary service
        
        Args:
            db_manager: Database manager instance for MongoDB operations
        """
        self.db = db_manager
        self.collection = "vocabulary"
        self.history_collection = "history"
    
    async def create_vocabulary(
        self, 
        user_id: str, 
        vocab_data: VocabularyCreate
    ) -> Dict[str, Any]:
        """
        Create a new vocabulary item with duplicate detection
        
        Args:
            user_id: User ID from JWT token
            vocab_data: Vocabulary creation data
            
        Returns:
            Dictionary with created vocabulary ID and success message
            
        Raises:
            ValueError: If duplicate vocabulary item exists (same user_id + word)
        """
        logger.info(
            f"Creating vocabulary for user {user_id}: "
            f"{vocab_data.word} ({vocab_data.source_language} -> {vocab_data.target_language})"
        )
        
        # Check for duplicates
        existing = await self.db.find_one(
            self.collection,
            {
                "user_id": user_id,
                "word": vocab_data.word,
                "source_language": vocab_data.source_language,
                "target_language": vocab_data.target_language
            }
        )
        
        if existing:
            logger.warning(
                f"Duplicate vocabulary item for user {user_id}: {vocab_data.word}"
            )
            raise ValueError(
                f"Vocabulary item '{vocab_data.word}' already exists for this language pair"
            )
        
        # Create document
        document = {
            "user_id": user_id,
            "word": vocab_data.word,
            "translation": vocab_data.translation,
            "source_language": vocab_data.source_language,
            "target_language": vocab_data.target_language,
            "notes": vocab_data.notes,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert into database
        vocab_id = await self.db.insert_one(self.collection, document)
        
        # Record in history
        await self._record_history(
            user_id=user_id,
            action_type="vocabulary_create",
            resource_id=vocab_id,
            outcome="success",
            details={
                "word": vocab_data.word,
                "translation": vocab_data.translation,
                "languages": f"{vocab_data.source_language}->{vocab_data.target_language}"
            }
        )
        
        logger.info(f"Vocabulary created successfully with ID: {vocab_id}")
        
        return {
            "id": vocab_id,
            "message": "Vocabulary created successfully"
        }
    
    async def get_vocabulary(
        self,
        user_id: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        page: int = 0,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Retrieve vocabulary items with filtering and pagination
        
        Args:
            user_id: User ID from JWT token
            source_language: Optional filter for source language
            target_language: Optional filter for target language
            page: Page number (0-indexed)
            page_size: Number of items per page
            
        Returns:
            Dictionary with vocabulary items, pagination info, and total count
        """
        logger.info(
            f"Retrieving vocabulary for user {user_id} "
            f"(source: {source_language}, target: {target_language}, "
            f"page: {page}, size: {page_size})"
        )
        
        # Build query
        query = {"user_id": user_id}
        
        if source_language:
            query["source_language"] = source_language
        
        if target_language:
            query["target_language"] = target_language
        
        # Get total count
        total = await self.db.count_documents(self.collection, query)
        
        # Get paginated results
        items = await self.db.find_many(
            self.collection,
            query,
            skip=page * page_size,
            limit=page_size,
            sort=[("created_at", -1)]  # Most recent first
        )
        
        # Convert ObjectId to string for JSON serialization
        for item in items:
            if "_id" in item:
                item["_id"] = str(item["_id"])
        
        logger.info(f"Retrieved {len(items)} vocabulary items (total: {total})")
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    
    async def update_vocabulary(
        self,
        vocab_id: str,
        user_id: str,
        update_data: VocabularyCreate
    ) -> Dict[str, str]:
        """
        Update an existing vocabulary item with user ownership verification
        
        Args:
            vocab_id: Vocabulary item ID
            user_id: User ID from JWT token
            update_data: Updated vocabulary data
            
        Returns:
            Dictionary with success message
            
        Raises:
            ValueError: If vocabulary item not found or user doesn't own it
        """
        logger.info(f"Updating vocabulary {vocab_id} for user {user_id}")
        
        # Verify ownership by checking if item exists for this user
        existing = await self.db.find_one(
            self.collection,
            {"_id": ObjectId(vocab_id), "user_id": user_id}
        )
        
        if not existing:
            logger.warning(
                f"Vocabulary {vocab_id} not found or unauthorized for user {user_id}"
            )
            raise ValueError(
                "Vocabulary item not found or you don't have permission to update it"
            )
        
        # Prepare update
        update = {
            "word": update_data.word,
            "translation": update_data.translation,
            "source_language": update_data.source_language,
            "target_language": update_data.target_language,
            "notes": update_data.notes,
            "updated_at": datetime.utcnow()
        }
        
        # Update in database
        success = await self.db.update_one(
            self.collection,
            {"_id": ObjectId(vocab_id), "user_id": user_id},
            update
        )
        
        if success:
            # Record in history
            await self._record_history(
                user_id=user_id,
                action_type="vocabulary_update",
                resource_id=vocab_id,
                outcome="success",
                details={
                    "word": update_data.word,
                    "translation": update_data.translation
                }
            )
            
            logger.info(f"Vocabulary {vocab_id} updated successfully")
            return {"message": "Vocabulary updated successfully"}
        else:
            logger.warning(f"No changes made to vocabulary {vocab_id}")
            return {"message": "No changes made"}
    
    async def delete_vocabulary(
        self,
        vocab_id: str,
        user_id: str
    ) -> Dict[str, str]:
        """
        Delete a vocabulary item with user ownership verification
        
        Args:
            vocab_id: Vocabulary item ID
            user_id: User ID from JWT token
            
        Returns:
            Dictionary with success message
            
        Raises:
            ValueError: If vocabulary item not found or user doesn't own it
        """
        logger.info(f"Deleting vocabulary {vocab_id} for user {user_id}")
        
        # Get item details before deletion for history
        existing = await self.db.find_one(
            self.collection,
            {"_id": ObjectId(vocab_id), "user_id": user_id}
        )
        
        if not existing:
            logger.warning(
                f"Vocabulary {vocab_id} not found or unauthorized for user {user_id}"
            )
            raise ValueError(
                "Vocabulary item not found or you don't have permission to delete it"
            )
        
        # Delete from database
        success = await self.db.delete_one(
            self.collection,
            {"_id": ObjectId(vocab_id), "user_id": user_id}
        )
        
        if success:
            # Record in history
            await self._record_history(
                user_id=user_id,
                action_type="vocabulary_delete",
                resource_id=vocab_id,
                outcome="success",
                details={
                    "word": existing.get("word", ""),
                    "translation": existing.get("translation", "")
                }
            )
            
            logger.info(f"Vocabulary {vocab_id} deleted successfully")
            return {"message": "Vocabulary deleted successfully"}
        else:
            logger.error(f"Failed to delete vocabulary {vocab_id}")
            raise ValueError("Failed to delete vocabulary item")
    
    async def _record_history(
        self,
        user_id: str,
        action_type: str,
        resource_id: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record action in history collection
        
        Args:
            user_id: User ID performing the action
            action_type: Type of action (vocabulary_create, vocabulary_update, etc.)
            resource_id: ID of the affected resource
            outcome: Outcome of the action (success, failure)
            details: Additional details about the action
        """
        try:
            history_record = {
                "user_id": user_id,
                "action_type": action_type,
                "resource_id": resource_id,
                "resource_type": "vocabulary",
                "outcome": outcome,
                "timestamp": datetime.utcnow(),
                "details": details or {}
            }
            
            await self.db.insert_one(self.history_collection, history_record)
            logger.debug(f"History recorded: {action_type} for user {user_id}")
            
        except Exception as e:
            # Don't fail the main operation if history recording fails
            logger.error(f"Failed to record history: {str(e)}", exc_info=True)
