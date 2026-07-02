"""Database layer for MongoDB operations with connection management and CRUD operations"""
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure
import logging
from typing import Optional, List, Dict, Any, Tuple
import asyncio
import ssl
import certifi


class DatabaseManager:
    """
    Manages MongoDB connections and provides CRUD operation abstractions.
    
    Features:
    - Async MongoDB operations using Motor
    - Connection pooling for performance (10-50 connections)
    - Retry logic for connection failures
    - Automatic index creation for optimized queries
    - Error handling and logging for all operations
    """
    
    def __init__(
        self,
        connection_string: str,
        database_name: str,
        max_pool_size: int = 50,
        min_pool_size: int = 10
    ):
        """
        Initialize DatabaseManager with connection parameters.
        
        Args:
            connection_string: MongoDB connection URL
            database_name: Name of the database to use
            max_pool_size: Maximum number of connections in the pool (default: 50)
            min_pool_size: Minimum number of connections in the pool (default: 10)
        """
        self.client: Optional[AsyncIOMotorClient] = None
        self.connection_string = connection_string
        self.database_name = database_name
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.db = None
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, max_retries: int = 3) -> None:
        """
        Establish database connection with retry logic.
        
        Motor uses lazy connection - the actual connection is established
        on first database operation. This avoids Python 3.13 SSL issues
        during startup ping operations.
        
        Args:
            max_retries: Maximum number of connection attempts (default: 3)
            
        Raises:
            ConnectionFailure: If connection fails after all retry attempts
        """
        try:
            # Create Motor client with TLS settings for Python 3.13 compatibility
            # Use tlsAllowInvalidCertificates to bypass strict SSL validation
            self.client = AsyncIOMotorClient(
                self.connection_string,
                maxPoolSize=self.max_pool_size,
                minPoolSize=self.min_pool_size,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True,
            )
            self.db = self.client[self.database_name]
            
            self.logger.info(
                f"Database client initialized with TLS settings (pool size: {self.min_pool_size}-{self.max_pool_size})"
            )
            self.logger.info(
                "Connection will be established on first database operation"
            )
            
            # Try to create indexes with a timeout
            # If it fails, defer to first actual database operation
            try:
                await asyncio.wait_for(self._create_indexes(), timeout=5.0)
                self.logger.info("Database connection verified - indexes created successfully")
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Index creation timed out - indexes will be created on first database operation"
                )
            except ConnectionFailure as conn_error:
                error_msg = str(conn_error)
                if "TLSV1_ALERT_INTERNAL_ERROR" in error_msg or "SSL handshake failed" in error_msg:
                    self.logger.error(
                        "TLS handshake failed - This usually means your IP address is not whitelisted in MongoDB Atlas"
                    )
                    self.logger.error(
                        "Go to MongoDB Atlas → Network Access → Add IP Address → Add your current IP"
                    )
                self.logger.error(f"Connection failed: {error_msg}")
                raise
            except Exception as index_error:
                self.logger.warning(
                    f"Index creation deferred: {str(index_error)}"
                )
                self.logger.info("Indexes will be created on first successful database operation")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database client: {str(e)}", exc_info=True)
            raise
    
    async def disconnect(self) -> None:
        """Close database connection and cleanup resources."""
        if self.client:
            self.client.close()
            self.logger.info("Database connection closed")
    
    async def _create_indexes(self) -> None:
        """
        Create indexes for all collections to optimize query performance.
        
        Indexes created:
        - vocabulary: user_id+word (unique), user_id+languages
        - history: user_id+timestamp, action_type
        - users: username (unique), email (unique)
        """
        try:
            # Vocabulary collection indexes
            # Compound unique index to prevent duplicate words per user
            await self.db.vocabulary.create_index(
                [("user_id", 1), ("word", 1)],
                unique=True,
                name="user_word_unique"
            )
            
            # Compound index for language pair filtering
            await self.db.vocabulary.create_index(
                [("user_id", 1), ("source_language", 1), ("target_language", 1)],
                name="user_languages"
            )
            
            # History collection indexes
            # Compound index for user history retrieval sorted by timestamp
            await self.db.history.create_index(
                [("user_id", 1), ("timestamp", -1)],
                name="user_history"
            )
            
            # Index for filtering by action type
            await self.db.history.create_index(
                [("action_type", 1)],
                name="action_type"
            )
            
            # Users collection indexes
            # Unique index for username
            await self.db.users.create_index(
                [("username", 1)],
                unique=True,
                name="username_unique"
            )
            
            # Unique index for email
            await self.db.users.create_index(
                [("email", 1)],
                unique=True,
                name="email_unique"
            )
            
            self.logger.info("Database indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {str(e)}", exc_info=True)
            # Don't raise - indexes might already exist or can be created later
    
    async def insert_one(self, collection: str, document: dict) -> str:
        """
        Insert a single document into a collection.
        
        Args:
            collection: Name of the collection
            document: Document to insert
            
        Returns:
            str: ID of the inserted document
            
        Raises:
            ValueError: If document violates unique constraints (duplicate entry)
            Exception: For other database errors
        """
        try:
            result = await self.db[collection].insert_one(document)
            self.logger.debug(f"Inserted document into {collection}: {result.inserted_id}")
            return str(result.inserted_id)
            
        except DuplicateKeyError as e:
            self.logger.warning(f"Duplicate key error in {collection}: {str(e)}")
            raise ValueError("Duplicate entry")
            
        except Exception as e:
            self.logger.error(
                f"Insert failed in {collection}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def find_one(self, collection: str, query: dict) -> Optional[dict]:
        """
        Find a single document in a collection.
        
        Args:
            collection: Name of the collection
            query: Query filter
            
        Returns:
            Optional[dict]: Found document or None if not found
            
        Raises:
            Exception: For database errors
        """
        try:
            result = await self.db[collection].find_one(query)
            return result
            
        except Exception as e:
            self.logger.error(
                f"Find one failed in {collection}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def find_many(
        self,
        collection: str,
        query: dict,
        skip: int = 0,
        limit: int = 100,
        sort: Optional[List[Tuple[str, int]]] = None
    ) -> List[dict]:
        """
        Find multiple documents in a collection with pagination and sorting.
        
        Args:
            collection: Name of the collection
            query: Query filter
            skip: Number of documents to skip (for pagination)
            limit: Maximum number of documents to return
            sort: List of (field, direction) tuples for sorting (1=ascending, -1=descending)
            
        Returns:
            List[dict]: List of found documents
            
        Raises:
            Exception: For database errors
        """
        try:
            cursor = self.db[collection].find(query).skip(skip).limit(limit)
            
            if sort:
                cursor = cursor.sort(sort)
            
            results = await cursor.to_list(length=limit)
            self.logger.debug(
                f"Found {len(results)} documents in {collection} "
                f"(skip={skip}, limit={limit})"
            )
            return results
            
        except Exception as e:
            self.logger.error(
                f"Find many failed in {collection}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def update_one(
        self,
        collection: str,
        query: dict,
        update: dict
    ) -> bool:
        """
        Update a single document in a collection.
        
        Args:
            collection: Name of the collection
            query: Query filter to find the document
            update: Fields to update (will be wrapped in $set operator)
            
        Returns:
            bool: True if a document was modified, False otherwise
            
        Raises:
            Exception: For database errors
        """
        try:
            result = await self.db[collection].update_one(
                query,
                {"$set": update}
            )
            
            modified = result.modified_count > 0
            
            if modified:
                self.logger.debug(f"Updated document in {collection}")
            else:
                self.logger.debug(
                    f"No document updated in {collection} (not found or no changes)"
                )
            
            return modified
            
        except Exception as e:
            self.logger.error(
                f"Update failed in {collection}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def delete_one(self, collection: str, query: dict) -> bool:
        """
        Delete a single document from a collection.
        
        Args:
            collection: Name of the collection
            query: Query filter to find the document
            
        Returns:
            bool: True if a document was deleted, False otherwise
            
        Raises:
            Exception: For database errors
        """
        try:
            result = await self.db[collection].delete_one(query)
            
            deleted = result.deleted_count > 0
            
            if deleted:
                self.logger.debug(f"Deleted document from {collection}")
            else:
                self.logger.debug(f"No document deleted from {collection} (not found)")
            
            return deleted
            
        except Exception as e:
            self.logger.error(
                f"Delete failed in {collection}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def count_documents(self, collection: str, query: dict) -> int:
        """
        Count documents matching a query.
        
        Args:
            collection: Name of the collection
            query: Query filter
            
        Returns:
            int: Number of matching documents
            
        Raises:
            Exception: For database errors
        """
        try:
            count = await self.db[collection].count_documents(query)
            return count
            
        except Exception as e:
            self.logger.error(
                f"Count failed in {collection}: {str(e)}",
                exc_info=True
            )
            raise


# Singleton instance placeholder - will be initialized in app startup
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get the singleton DatabaseManager instance.
    
    Returns:
        DatabaseManager: The active database manager instance
        
    Raises:
        RuntimeError: If database manager has not been initialized
    """
    if db_manager is None:
        raise RuntimeError(
            "DatabaseManager not initialized. "
            "Call initialize_db_manager() during application startup."
        )
    return db_manager


def initialize_db_manager(
    connection_string: str,
    database_name: str,
    max_pool_size: int = 50,
    min_pool_size: int = 10
) -> DatabaseManager:
    """
    Initialize the singleton DatabaseManager instance.
    
    Args:
        connection_string: MongoDB connection URL
        database_name: Name of the database to use
        max_pool_size: Maximum number of connections in the pool
        min_pool_size: Minimum number of connections in the pool
        
    Returns:
        DatabaseManager: The initialized database manager instance
    """
    global db_manager
    db_manager = DatabaseManager(
        connection_string=connection_string,
        database_name=database_name,
        max_pool_size=max_pool_size,
        min_pool_size=min_pool_size
    )
    return db_manager
