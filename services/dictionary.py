"""Dictionary management service for importing and querying trilingual vocabulary"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from services.database import DatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class DictionaryService:
    """Service for managing dictionary/lexicon data from dictionary.json"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize dictionary service
        
        Args:
            db_manager: Database manager instance for MongoDB operations
        """
        self.db = db_manager
        self.collection_name = "dictionary"
        
        logger.info("DictionaryService initialized")
    
    def _normalize_category_key(self, category: str) -> str:
        """
        Normalize category name to kebab-case key
        
        Extracts English name from parentheses and converts to lowercase kebab-case
        
        Examples:
            "BAHAGI NI LAWES (Parts of the Body)" -> "parts-of-body"
            "MANGAYUN NI MITTANAK (Members of the Family)" -> "members-of-family"
            "Common Objects" -> "common-objects"
        
        Args:
            category: Full category name from dictionary.json
            
        Returns:
            Normalized kebab-case category key
        """
        # Extract English name from parentheses if present
        match = re.search(r'\(([^)]+)\)', category)
        if match:
            english_name = match.group(1)
        else:
            english_name = category
        
        # Convert to lowercase and replace spaces with hyphens
        key = english_name.lower()
        key = re.sub(r'[^a-z0-9\s-]', '', key)  # Remove special chars
        key = re.sub(r'\s+', '-', key)  # Replace spaces with hyphens
        key = re.sub(r'-+', '-', key)  # Collapse multiple hyphens
        key = key.strip('-')  # Remove leading/trailing hyphens
        
        return key
    
    def _validate_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Validate dictionary entry
        
        Checks that all required fields exist and are not empty or placeholders
        
        Args:
            entry: Dictionary entry with BULOS, FILIPINO, ENGLISH fields
            
        Returns:
            True if entry is valid, False otherwise
        """
        # Check required fields exist
        required_fields = ["BULOS", "FILIPINO", "ENGLISH"]
        for field in required_fields:
            if field not in entry:
                return False
            
            value = str(entry[field]).strip()
            
            # Skip empty values
            if not value:
                return False
            
            # Skip placeholder values
            if value == "—":
                return False
        
        return True
    
    def _parse_dictionary_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load and parse dictionary.json to extract all entries
        
        Reads the JSON file, validates structure, and extracts all vocabulary entries
        with their category information
        
        Args:
            json_path: Path to dictionary.json file
            
        Returns:
            List of dictionaries with parsed entry data
            
        Raises:
            FileNotFoundError: If dictionary.json doesn't exist
            ValueError: If JSON format is invalid
        """
        # Check file exists
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Dictionary file not found: {json_path}")
        
        # Load JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        # Validate structure
        if "categories" not in dictionary:
            raise ValueError("Invalid dictionary format: 'categories' key not found")
        
        # Extract entries from all categories
        parsed_entries = []
        total_entries = 0
        skipped_entries = 0
        
        for category_data in dictionary.get("categories", []):
            category = category_data.get("category", "Unknown")
            category_key = self._normalize_category_key(category)
            
            for entry in category_data.get("entries", []):
                total_entries += 1
                
                # Validate entry
                if not self._validate_entry(entry):
                    skipped_entries += 1
                    continue
                
                # Create parsed entry
                parsed_entry = {
                    "bulos": entry.get("BULOS", "").strip().lower(),
                    "filipino": entry.get("FILIPINO", "").strip().lower(),
                    "english": entry.get("ENGLISH", "").strip().lower(),
                    "category": category,
                    "category_key": category_key,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "metadata": {
                        "source": "dictionary.json",
                        "import_version": "1.0"
                    }
                }
                
                parsed_entries.append(parsed_entry)
        
        logger.info(
            f"Parsed {len(parsed_entries)} valid entries from {total_entries} total "
            f"({skipped_entries} skipped)"
        )
        
        return parsed_entries

    
    async def _create_indexes(self) -> None:
        """
        Create MongoDB indexes for dictionary collection
        
        Creates all required indexes for optimal query performance:
        - Unique compound index on (bulos, filipino, english) to prevent duplicates
        - Text search index on all three language fields for fuzzy search
        - Individual indexes on bulos, filipino, english for exact lookups
        - Index on category_key for category filtering
        
        This method is idempotent - safe to run multiple times
        """
        try:
            logger.info("Creating indexes for dictionary collection...")
            
            # Unique compound index to prevent duplicate entries
            await self.db.db[self.collection_name].create_index(
                [("bulos", 1), ("filipino", 1), ("english", 1)],
                unique=True,
                name="unique_entry"
            )
            logger.info("Created unique compound index on (bulos, filipino, english)")
            
            # Text search index for fuzzy searching across all languages
            await self.db.db[self.collection_name].create_index(
                [("bulos", "text"), ("filipino", "text"), ("english", "text")],
                name="text_search",
                default_language="english"
            )
            logger.info("Created text search index on all language fields")
            
            # Individual field indexes for exact lookups
            await self.db.db[self.collection_name].create_index(
                [("bulos", 1)],
                name="bulos_lookup"
            )
            logger.info("Created index on bulos field")
            
            await self.db.db[self.collection_name].create_index(
                [("filipino", 1)],
                name="filipino_lookup"
            )
            logger.info("Created index on filipino field")
            
            await self.db.db[self.collection_name].create_index(
                [("english", 1)],
                name="english_lookup"
            )
            logger.info("Created index on english field")
            
            # Category index for filtering by category
            await self.db.db[self.collection_name].create_index(
                [("category_key", 1)],
                name="category_filter"
            )
            logger.info("Created index on category_key field")
            
            logger.info("All dictionary indexes created successfully")
            
        except Exception as e:
            # Log error but don't fail - indexes might already exist
            logger.warning(f"Index creation warning (may already exist): {str(e)}")

    
    async def _bulk_insert_entries(
        self,
        entries: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> tuple[int, int, int]:
        """
        Insert entries in batches with duplicate handling
        
        Args:
            entries: List of dictionary entries to insert
            batch_size: Number of entries to insert per batch (default: 100)
            
        Returns:
            Tuple of (inserted_count, skipped_count, error_count)
        """
        from pymongo.errors import BulkWriteError
        
        inserted = 0
        skipped = 0
        errors = 0
        
        total_entries = len(entries)
        
        # Process in batches
        for i in range(0, total_entries, batch_size):
            batch = entries[i:i+batch_size]
            
            try:
                # Use insert_many with ordered=False to continue on duplicates
                result = await self.db.db[self.collection_name].insert_many(
                    batch,
                    ordered=False
                )
                inserted += len(result.inserted_ids)
                
            except BulkWriteError as e:
                # Count successful inserts even with some errors
                inserted += e.details.get('nInserted', 0)
                
                # Count duplicate key errors as skipped
                for error in e.details.get('writeErrors', []):
                    if error.get('code') == 11000:  # Duplicate key error
                        skipped += 1
                    else:
                        errors += 1
                        logger.error(f"Insert error: {error}")
            
            # Log progress every 100 entries
            processed = min(i + batch_size, total_entries)
            if processed % 100 == 0 or processed == total_entries:
                logger.info(f"Processed {processed} / {total_entries} entries")
        
        return inserted, skipped, errors
    
    async def import_from_json(
        self,
        json_path: str,
        force_reimport: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Import dictionary entries from JSON file into MongoDB
        
        This is the main orchestrator method that:
        1. Parses dictionary.json
        2. Creates indexes
        3. Bulk inserts entries
        4. Returns statistics
        
        Args:
            json_path: Path to dictionary.json file
            force_reimport: If True, drops existing collection before import
            dry_run: If True, validates data without inserting
            
        Returns:
            ImportResult dictionary with statistics
            
        Raises:
            FileNotFoundError: If dictionary.json doesn't exist
            ValueError: If JSON format is invalid
        """
        import time
        
        start_time = time.time()
        
        logger.info(f"Starting dictionary import from {json_path}")
        logger.info(f"Force reimport: {force_reimport}, Dry run: {dry_run}")
        
        # Parse dictionary.json
        entries = self._parse_dictionary_json(json_path)
        total_entries = len(entries)
        
        if dry_run:
            logger.info(f"Dry run complete - would import {total_entries} entries")
            duration = time.time() - start_time
            return {
                "total_entries": total_entries,
                "inserted": 0,
                "skipped": 0,
                "errors": 0,
                "duration": duration,
                "dry_run": True,
                "error_details": []
            }
        
        # Handle force reimport
        if force_reimport:
            logger.warning("Force reimport enabled - dropping existing collection")
            try:
                await self.db.db[self.collection_name].drop()
                logger.info("Existing collection dropped")
            except Exception as e:
                logger.error(f"Failed to drop collection: {str(e)}")
        
        # Create indexes
        await self._create_indexes()
        
        # Bulk insert entries
        logger.info(f"Starting bulk insert of {total_entries} entries...")
        inserted, skipped, errors = await self._bulk_insert_entries(entries)
        
        duration = time.time() - start_time
        
        # Log final statistics
        logger.info("="*60)
        logger.info("Dictionary Import Complete")
        logger.info("="*60)
        logger.info(f"Total entries processed: {total_entries}")
        logger.info(f"Successfully inserted:   {inserted}")
        logger.info(f"Skipped (duplicates):    {skipped}")
        logger.info(f"Errors:                  {errors}")
        logger.info(f"Duration:                {duration:.2f} seconds")
        logger.info("="*60)
        
        return {
            "total_entries": total_entries,
            "inserted": inserted,
            "skipped": skipped,
            "errors": errors,
            "duration": duration,
            "error_details": []
        }

    
    async def lookup_word(
        self,
        word: str,
        source_language: str
    ) -> List[Dict[str, Any]]:
        """
        Look up a word in the dictionary
        
        Searches for exact or partial matches in the specified language field
        
        Args:
            word: Word to look up
            source_language: Language code (bul, tl, en)
            
        Returns:
            List of matching dictionary entries
        """
        try:
            word_lower = word.lower().strip()
            
            # Map language code to field name
            field_map = {
                "bul": "bulos",
                "tl": "filipino",
                "en": "english"
            }
            
            field = field_map.get(source_language, "bulos")
            
            logger.info(f"Looking up word '{word}' in {source_language} ({field})")
            
            # Search for exact match or partial match using regex
            query = {
                field: {"$regex": f"^{word_lower}", "$options": "i"}
            }
            
            results = await self.db.find_many(
                self.collection_name,
                query,
                limit=50
            )
            
            logger.info(f"Found {len(results)} matches for '{word}'")
            
            return results
            
        except Exception as e:
            logger.error(f"Lookup failed for '{word}': {str(e)}", exc_info=True)
            raise
    
    async def search_words(
        self,
        query: str,
        language: str = "all",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for words using fuzzy text search
        
        Uses MongoDB text index for searching across all languages or specific language
        
        Args:
            query: Search query string
            language: Language to search (bul, tl, en, or 'all')
            limit: Maximum number of results
            
        Returns:
            List of matching dictionary entries
        """
        try:
            logger.info(f"Searching for '{query}' in language: {language}")
            
            if language == "all":
                # Search across all languages using text index
                search_query = {
                    "$text": {"$search": query}
                }
            else:
                # Search in specific language field
                field_map = {
                    "bul": "bulos",
                    "tl": "filipino",
                    "en": "english"
                }
                
                field = field_map.get(language, "bulos")
                search_query = {
                    field: {"$regex": query, "$options": "i"}
                }
            
            results = await self.db.find_many(
                self.collection_name,
                search_query,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} search results for '{query}'")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for '{query}': {str(e)}", exc_info=True)
            raise
    
    async def get_by_category(
        self,
        category_key: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all dictionary entries in a specific category
        
        Args:
            category_key: Category key (kebab-case, e.g., "parts-of-body")
            skip: Number of entries to skip (for pagination)
            limit: Maximum number of entries to return
            
        Returns:
            List of dictionary entries in the category
        """
        try:
            logger.info(f"Getting entries for category: {category_key}")
            
            query = {"category_key": category_key}
            
            results = await self.db.find_many(
                self.collection_name,
                query,
                skip=skip,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} entries in category '{category_key}'")
            
            return results
            
        except Exception as e:
            logger.error(
                f"Failed to get category '{category_key}': {str(e)}",
                exc_info=True
            )
            raise
    
    async def get_all_categories(self) -> List[Dict[str, Any]]:
        """
        Get all dictionary categories with entry counts
        
        Returns:
            List of CategoryInfo dictionaries with category names and counts
        """
        try:
            logger.info("Getting all dictionary categories")
            
            # Use aggregation to group by category and count entries
            pipeline = [
                {
                    "$group": {
                        "_id": "$category_key",
                        "category": {"$first": "$category"},
                        "entry_count": {"$sum": 1}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "category": 1,
                        "category_key": "$_id",
                        "entry_count": 1
                    }
                },
                {
                    "$sort": {"entry_count": -1}
                }
            ]
            
            cursor = self.db.db[self.collection_name].aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            logger.info(f"Found {len(results)} categories")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get categories: {str(e)}", exc_info=True)
            raise
    
    async def count_entries(self) -> int:
        """
        Count total number of dictionary entries
        
        Returns:
            Total count of dictionary entries
        """
        try:
            count = await self.db.count_documents(self.collection_name, {})
            logger.info(f"Total dictionary entries: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to count entries: {str(e)}", exc_info=True)
            raise
