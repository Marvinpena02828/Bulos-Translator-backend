#!/usr/bin/env python3
"""
Dictionary Import Script

Imports trilingual vocabulary data from dictionary.json into MongoDB.

Usage:
    python scripts/import_dictionary.py [--force] [--dry-run]
    
Options:
    --force     Drop existing collection and reimport all data
    --dry-run   Validate data without inserting into database
    
Examples:
    python scripts/import_dictionary.py
    python scripts/import_dictionary.py --force
    python scripts/import_dictionary.py --dry-run
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.database import initialize_db_manager
from services.dictionary import DictionaryService
from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


async def main():
    """Main entry point for dictionary import script"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Import dictionary.json into MongoDB",
        epilog="Example: python scripts/import_dictionary.py --force"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop existing collection and reimport all data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without inserting into database"
    )
    args = parser.parse_args()
    
    # Display startup information
    print("="*60)
    print("Dictionary Import Script")
    print("="*60)
    print(f"Force reimport: {args.force}")
    print(f"Dry run mode:   {args.dry_run}")
    print("="*60)
    print()
    
    # Initialize database manager
    logger.info("Initializing database connection...")
    db_manager = initialize_db_manager(
        connection_string=settings.mongodb_url,
        database_name=settings.mongodb_database,
        max_pool_size=settings.mongodb_max_pool_size,
        min_pool_size=settings.mongodb_min_pool_size
    )
    
    try:
        # Connect to database
        print("Connecting to MongoDB...")
        await db_manager.connect()
        logger.info("Connected to MongoDB successfully")
        print("✓ Connected to MongoDB")
        print()
        
        # Initialize dictionary service
        logger.info("Initializing dictionary service...")
        dictionary_service = DictionaryService(db_manager)
        print("✓ Dictionary service initialized")
        print()
        
        # Path to dictionary.json
        dictionary_path = Path(__file__).parent.parent / "dictionary.json"
        
        if not dictionary_path.exists():
            print(f"✗ Error: Dictionary file not found at {dictionary_path}")
            logger.error(f"Dictionary file not found: {dictionary_path}")
            sys.exit(1)
        
        print(f"✓ Found dictionary file: {dictionary_path}")
        print()
        
        # Run import
        print("Starting dictionary import...")
        print()
        
        result = await dictionary_service.import_from_json(
            str(dictionary_path),
            force_reimport=args.force,
            dry_run=args.dry_run
        )
        
        # Display results
        print()
        print("="*60)
        print("Dictionary Import Results")
        print("="*60)
        print(f"Total entries processed: {result['total_entries']}")
        print(f"Successfully inserted:   {result['inserted']}")
        print(f"Skipped (duplicates):    {result['skipped']}")
        print(f"Errors:                  {result['errors']}")
        print(f"Duration:                {result['duration']:.2f} seconds")
        print("="*60)
        
        if result['errors'] > 0 and result['error_details']:
            print()
            print("Errors occurred during import:")
            for error in result['error_details']:
                print(f"  - {error}")
            print()
        
        if args.dry_run:
            print()
            print("[DRY RUN] No data was inserted into database")
            print()
        
        # Success message
        if result['errors'] == 0:
            print()
            print("✓ Dictionary import completed successfully!")
            logger.info("Dictionary import completed successfully")
            sys.exit(0)
        else:
            print()
            print("⚠ Dictionary import completed with errors")
            logger.warning(f"Dictionary import completed with {result['errors']} errors")
            sys.exit(1)
        
    except FileNotFoundError as e:
        print()
        print(f"✗ Error: {e}")
        logger.error(f"File not found: {e}")
        sys.exit(1)
        
    except ValueError as e:
        print()
        print(f"✗ Error: {e}")
        logger.error(f"Validation error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print()
        print(f"✗ Error: Import failed - {str(e)}")
        logger.error(f"Import failed: {str(e)}", exc_info=True)
        sys.exit(1)
        
    finally:
        # Disconnect from database
        if db_manager:
            logger.info("Disconnecting from database...")
            await db_manager.disconnect()
            print()
            print("✓ Disconnected from MongoDB")


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print()
        print("⚠ Import cancelled by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print()
        print(f"✗ Fatal error: {str(e)}")
        sys.exit(1)
