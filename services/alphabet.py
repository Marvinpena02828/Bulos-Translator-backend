"""Alphabet management service for importing and querying Dumaget Bulos alphabet"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from services.database import DatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class AlphabetService:
    """Service for managing alphabet data from alphabet.json"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize alphabet service
        
        Args:
            db_manager: Database manager instance for MongoDB operations
        """
        self.db = db_manager
        self.collection_name = "alphabet"
        self.metadata_collection = "alphabet_metadata"
        
        logger.info("AlphabetService initialized")
    
    def _validate_example(self, example: Dict[str, Any]) -> bool:
        """
        Validate alphabet example entry
        
        Args:
            example: Example with BULOS, FILIPINO, ENGLISH fields
            
        Returns:
            True if example is valid, False otherwise
        """
        if "BULOS" not in example or "FILIPINO" not in example:
            return False
        
        # Check for None values
        bulos = example.get("BULOS")
        filipino = example.get("FILIPINO")
        
        if bulos is None or filipino is None:
            return False
        
        # Convert to string and check if empty
        if not str(bulos).strip() or str(bulos).strip() == 'None':
            return False
        
        if not str(filipino).strip() or str(filipino).strip() == 'None':
            return False
        
        return True
    
    def _validate_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Validate alphabet entry
        
        Args:
            entry: Alphabet entry with letter and examples
            
        Returns:
            True if entry is valid, False otherwise
        """
        if "letter" not in entry:
            return False
        
        if not entry.get("letter", "").strip():
            return False
        
        if "examples" not in entry:
            return False
        
        examples = entry.get("examples", {})
        if not isinstance(examples, dict):
            return False
        
        # Check that examples has initial, middle, and final
        required_positions = ["initial", "middle", "final"]
        for pos in required_positions:
            if pos not in examples:
                return False
        
        return True
    
    def _parse_alphabet_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load and parse alphabet.json to extract all entries
        
        Args:
            json_path: Path to alphabet.json file
            
        Returns:
            List of parsed alphabet entries
            
        Raises:
            FileNotFoundError: If alphabet.json doesn't exist
            ValueError: If JSON format is invalid
        """
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Alphabet file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        if "letters" not in data:
            raise ValueError("Invalid alphabet format: 'letters' key not found")
        
        parsed_entries = []
        total_entries = 0
        skipped_entries = 0
        
        for entry in data.get("letters", []):
            total_entries += 1
            
            if not self._validate_entry(entry):
                skipped_entries += 1
                logger.warning(f"Skipping invalid entry: {entry.get('letter', 'unknown')}")
                continue
            
            # Parse examples by position
            examples_data = entry.get("examples", {})
            parsed_examples = {
                "initial": [],
                "middle": [],
                "final": []
            }
            
            for position in ["initial", "middle", "final"]:
                for ex in examples_data.get(position, []):
                    if self._validate_example(ex):
                        # Safely convert to string and strip
                        bulos_val = ex.get("BULOS")
                        filipino_val = ex.get("FILIPINO")
                        english_val = ex.get("ENGLISH")
                        
                        parsed_examples[position].append({
                            "bulos": str(bulos_val).strip() if bulos_val is not None else "",
                            "filipino": str(filipino_val).strip() if filipino_val is not None else "",
                            "english": str(english_val).strip() if english_val is not None and str(english_val).strip() else None
                        })
            
            parsed_entry = {
                "letter": entry.get("letter", "").strip(),
                "examples": parsed_examples,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "metadata": {
                    "source": "alphabet.json",
                    "import_version": "1.0"
                }
            }
            
            parsed_entries.append(parsed_entry)
        
        logger.info(
            f"Parsed {len(parsed_entries)} valid alphabet entries from {total_entries} total "
            f"({skipped_entries} skipped)"
        )
        
        return parsed_entries
    
    async def _create_indexes(self) -> None:
        """Create MongoDB indexes for alphabet collection"""
        try:
            logger.info("Creating indexes for alphabet collection...")
            
            # Unique index on letter
            await self.db.db[self.collection_name].create_index(
                [("letter", 1)],
                unique=True,
                name="unique_letter"
            )
            logger.info("Created unique index on letter")
            
            logger.info("All alphabet indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Index creation warning: {str(e)}")
    
    async def _bulk_insert_entries(
        self,
        entries: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> tuple[int, int, int]:
        """
        Insert entries in batches with duplicate handling
        
        Args:
            entries: List of alphabet entries to insert
            batch_size: Number of entries to insert per batch
            
        Returns:
            Tuple of (inserted_count, skipped_count, error_count)
        """
        from pymongo.errors import BulkWriteError
        
        inserted = 0
        skipped = 0
        errors = 0
        
        total_entries = len(entries)
        
        for i in range(0, total_entries, batch_size):
            batch = entries[i:i+batch_size]
            
            try:
                result = await self.db.db[self.collection_name].insert_many(
                    batch,
                    ordered=False
                )
                inserted += len(result.inserted_ids)
                
            except BulkWriteError as e:
                inserted += e.details.get('nInserted', 0)
                
                for error in e.details.get('writeErrors', []):
                    if error.get('code') == 11000:  # Duplicate key error
                        skipped += 1
                    else:
                        errors += 1
                        logger.error(f"Insert error: {error}")
            
            processed = min(i + batch_size, total_entries)
            logger.info(f"Processed {processed} / {total_entries} alphabet entries")
        
        return inserted, skipped, errors
    
    async def import_from_json(
        self,
        json_path: str,
        force_reimport: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Import alphabet entries from JSON file into MongoDB
        
        Args:
            json_path: Path to alphabet.json file
            force_reimport: If True, drops existing collection before import
            dry_run: If True, validates data without inserting
            
        Returns:
            ImportResult dictionary with statistics
        """
        start_time = time.time()
        
        logger.info(f"Starting alphabet import from {json_path}")
        logger.info(f"Force reimport: {force_reimport}, Dry run: {dry_run}")
        
        entries = self._parse_alphabet_json(json_path)
        total_entries = len(entries)
        
        if dry_run:
            logger.info(f"Dry run complete - would import {total_entries} alphabet entries")
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
        
        if force_reimport:
            logger.warning("Force reimport enabled - dropping existing collection")
            try:
                await self.db.db[self.collection_name].drop()
                logger.info("Existing collection dropped")
            except Exception as e:
                logger.error(f"Failed to drop collection: {str(e)}")
        
        await self._create_indexes()
        
        logger.info(f"Starting bulk insert of {total_entries} alphabet entries...")
        inserted, skipped, errors = await self._bulk_insert_entries(entries)
        
        duration = time.time() - start_time
        
        logger.info("="*60)
        logger.info("Alphabet Import Complete")
        logger.info("="*60)
        logger.info(f"Total alphabet entries processed: {total_entries}")
        logger.info(f"Successfully inserted:            {inserted}")
        logger.info(f"Skipped (duplicates):             {skipped}")
        logger.info(f"Errors:                           {errors}")
        logger.info(f"Duration:                         {duration:.2f} seconds")
        logger.info("="*60)
        
        return {
            "total_entries": total_entries,
            "inserted": inserted,
            "skipped": skipped,
            "errors": errors,
            "duration": duration,
            "error_details": []
        }
    
    async def get_by_letter(self, letter: str) -> Dict[str, Any]:
        """
        Get alphabet entry by letter
        
        Args:
            letter: Letter to lookup (e.g., 'Aa', 'Bb')
            
        Returns:
            Alphabet entry or None if not found
        """
        try:
            logger.info(f"Looking up alphabet entry for letter: {letter}")
            
            result = await self.db.find_one(
                self.collection_name,
                {"letter": letter}
            )
            
            if result:
                logger.info(f"Found alphabet entry for letter: {letter}")
            else:
                logger.info(f"No alphabet entry found for letter: {letter}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get letter '{letter}': {str(e)}", exc_info=True)
            raise
    
    async def get_all_letters(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all alphabet entries with pagination
        
        Args:
            skip: Number of entries to skip
            limit: Maximum number of entries to return
            
        Returns:
            List of alphabet entries
        """
        try:
            logger.info(f"Getting alphabet entries with skip={skip}, limit={limit}")
            
            results = await self.db.find_many(
                self.collection_name,
                {},
                skip=skip,
                limit=limit
            )
            
            logger.info(f"Retrieved {len(results)} alphabet entries")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get alphabet entries: {str(e)}", exc_info=True)
            raise
    
    async def count_letters(self) -> int:
        """
        Count total number of alphabet entries
        
        Returns:
            Total count of alphabet entries
        """
        try:
            count = await self.db.count_documents(self.collection_name, {})
            logger.info(f"Total alphabet entries: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to count alphabet entries: {str(e)}", exc_info=True)
            raise
