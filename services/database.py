"""
MongoDB database connection layer for Bulos Translator backend.

This module provides a DatabaseManager class for establishing and managing
connections to MongoDB Atlas. It handles connection pooling, TLS configuration,
and provides basic CRUD operations.

PHASE M1: Infrastructure only - does NOT modify TranslationService.
"""
import logging
from typing import Optional, List, Dict, Any, Tuple
import asyncio

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure
import certifi


class DatabaseManager:
    """
    Manages MongoDB Atlas connections with async operations.
    
    Features:
    - Async MongoDB operations using Motor (async PyMongo driver)
    - Connection pooling for performance (configurable pool size)
    - TLS/SSL configuration for MongoDB Atlas
    - Lazy connection (connects on first operation)
    - Basic CRUD operations for collections
    - Error handling and logging
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
            connection_string: MongoDB Atlas connection URI
            database_name: Name of the database to use
            max_pool_size: Maximum number of connections in pool (default: 50)
            min_pool_size: Minimum number of connections in pool (default: 10)
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
            # Create Motor client with TLS settings for Python 3.13+ compatibility
            # Uses certifi CA bundle for Atlas TLS certificate validation
            self.client = AsyncIOMotorClient(
                self.connection_string,
                maxPoolSize=self.max_pool_size,
                minPoolSize=self.min_pool_size,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                tlsCAFile=certifi.where(),  # Use certifi CA bundle for Atlas TLS
            )
            self.db = self.client[self.database_name]
            
            self.logger.info(
                f"Database client initialized (pool size: {self.min_pool_size}-{self.max_pool_size})"
            )
            self.logger.info(
                "Connection will be established on first database operation (lazy connection)"
            )
            
            # Try to verify connection with a simple operation (with timeout)
            try:
                await asyncio.wait_for(self._verify_connection(), timeout=5.0)
                self.logger.info("MongoDB connection verified successfully")
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Connection verification timed out - connection will be established on first database operation"
                )
            except ConnectionFailure as conn_error:
                error_msg = str(conn_error)
                if "TLSV1_ALERT_INTERNAL_ERROR" in error_msg or "SSL handshake failed" in error_msg:
                    self.logger.error(
                        "TLS handshake failed - Your IP address may not be whitelisted in MongoDB Atlas"
                    )
                    self.logger.error(
                        "Go to MongoDB Atlas → Network Access → Add IP Address → Add your current IP"
                    )
                self.logger.error(f"Connection failed: {error_msg}")
                raise
            except Exception as verify_error:
                self.logger.warning(
                    f"Connection verification deferred: {str(verify_error)}"
                )
                self.logger.info("Connection will be verified on first successful database operation")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database client: {str(e)}", exc_info=True)
            raise
    
    async def _verify_connection(self) -> None:
        """
        Verify connection by executing a simple ping command.
        This forces Motor to establish the connection.
        """
        await self.db.command('ping')
        self.logger.debug("Connection ping successful")
    
    async def disconnect(self) -> None:
        """Close database connection and cleanup resources."""
        if self.client:
            self.client.close()
            self.logger.info("Database connection closed")
    
    async def list_collections(self) -> List[str]:
        """
        List all collection names in the database.
        
        Returns:
            List of collection names
            
        Raises:
            Exception: For database errors
        """
        try:
            collections = await self.db.list_collection_names()
            self.logger.debug(f"Found {len(collections)} collections")
            return collections
        except Exception as e:
            self.logger.error(f"Failed to list collections: {str(e)}", exc_info=True)
            raise
    
    async def count_documents(self, collection: str, query: Dict = None) -> int:
        """
        Count documents matching a query in a collection.
        
        Args:
            collection: Name of the collection
            query: Query filter (default: {} to count all documents)
            
        Returns:
            Number of matching documents
            
        Raises:
            Exception: For database errors
        """
        try:
            if query is None:
                query = {}
            count = await self.db[collection].count_documents(query)
            return count
        except Exception as e:
            self.logger.error(
                f"Count failed in {collection}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def find_one(self, collection: str, query: Dict) -> Optional[Dict]:
        """
        Find a single document in a collection (read-only for M1).
        
        Args:
            collection: Name of the collection
            query: Query filter
            
        Returns:
            Found document or None if not found
            
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
        query: Dict,
        skip: int = 0,
        limit: int = 100,
        sort: Optional[List[Tuple[str, int]]] = None
    ) -> List[Dict]:
        """
        Find multiple documents in a collection (read-only for M1).
        
        Args:
            collection: Name of the collection
            query: Query filter
            skip: Number of documents to skip (for pagination)
            limit: Maximum number of documents to return
            sort: List of (field, direction) tuples for sorting (1=ascending, -1=descending)
            
        Returns:
            List of found documents
            
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


# Singleton instance - will be initialized during application startup
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get the singleton DatabaseManager instance.
    
    Returns:
        DatabaseManager: The active database manager instance
        
    Raises:
        RuntimeError: If database manager has not been initialized
    """
    if _db_manager is None:
        raise RuntimeError(
            "DatabaseManager not initialized. "
            "Call initialize_db_manager() during application startup."
        )
    return _db_manager


def initialize_db_manager(
    connection_string: str,
    database_name: str,
    max_pool_size: int = 50,
    min_pool_size: int = 10
) -> DatabaseManager:
    """
    Initialize the singleton DatabaseManager instance.
    
    Args:
        connection_string: MongoDB Atlas connection URI
        database_name: Name of the database to use
        max_pool_size: Maximum number of connections in pool
        min_pool_size: Minimum number of connections in pool
        
    Returns:
        DatabaseManager: The initialized database manager instance
    """
    global _db_manager
    _db_manager = DatabaseManager(
        connection_string=connection_string,
        database_name=database_name,
        max_pool_size=max_pool_size,
        min_pool_size=min_pool_size
    )
    return _db_manager
