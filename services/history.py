"""History tracking service for logging user actions and events"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from services.database import DatabaseManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class HistoryService:
    """Service for tracking and retrieving user action history"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize history service
        
        Args:
            db_manager: Database manager instance for MongoDB operations
        """
        self.db = db_manager
        self.collection = "history"
    
    async def record_action(
        self,
        user_id: str,
        action_type: str,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a user action in the history collection
        
        This method stores a history record with all relevant information about
        an action performed by a user. All records include a timestamp set to
        the current UTC time.
        
        Args:
            user_id: User ID performing the action
            action_type: Type of action (e.g., 'vocabulary_create', 'translation', 'pronunciation_evaluation')
            resource_id: Optional ID of the resource affected by the action
            resource_type: Optional type of resource (e.g., 'vocabulary', 'evaluation', 'session')
            outcome: Outcome of the action (default: 'success', can be 'error', 'failure')
            details: Optional dictionary with additional details about the action
            
        Returns:
            History record ID
            
        Example action_types:
            - vocabulary_create, vocabulary_update, vocabulary_delete
            - translation
            - pronunciation_evaluation
            - session_start, session_end
        """
        logger.debug(
            f"Recording history: user={user_id}, action={action_type}, "
            f"resource={resource_type}:{resource_id}, outcome={outcome}"
        )
        
        # Create history record
        history_record = {
            "user_id": user_id,
            "action_type": action_type,
            "resource_id": resource_id,
            "resource_type": resource_type,
            "outcome": outcome,
            "timestamp": datetime.utcnow(),
            "details": details or {}
        }
        
        try:
            # Insert into database
            record_id = await self.db.insert_one(self.collection, history_record)
            
            logger.debug(f"History recorded successfully with ID: {record_id}")
            
            return record_id
            
        except Exception as e:
            # Log error but don't fail - history recording should be non-blocking
            logger.error(
                f"Failed to record history for user {user_id}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def get_history(
        self,
        user_id: str,
        action_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 0,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Retrieve user action history with filtering and pagination
        
        This method retrieves history records for a specific user with optional
        filters for action type and date range. Results are paginated and sorted
        by most recent first.
        
        Args:
            user_id: User ID to retrieve history for
            action_type: Optional filter for specific action type
            start_date: Optional filter for actions after this date
            end_date: Optional filter for actions before this date
            page: Page number (0-indexed)
            page_size: Number of records per page
            
        Returns:
            Dictionary containing:
            - records: List of history records
            - total: Total number of records matching the query
            - page: Current page number
            - page_size: Number of records per page
        """
        logger.info(
            f"Retrieving history for user {user_id} "
            f"(action_type: {action_type}, start_date: {start_date}, "
            f"end_date: {end_date}, page: {page}, size: {page_size})"
        )
        
        # Build query
        query = {"user_id": user_id}
        
        # Add action_type filter
        if action_type:
            query["action_type"] = action_type
        
        # Add date range filters
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        # Count total records matching query
        total = await self.db.count_documents(self.collection, query)
        
        # Get paginated results
        records = await self.db.find_many(
            self.collection,
            query,
            skip=page * page_size,
            limit=page_size,
            sort=[("timestamp", -1)]  # Most recent first
        )
        
        # Convert ObjectId to string for JSON serialization
        for record in records:
            if "_id" in record:
                record["_id"] = str(record["_id"])
        
        logger.info(
            f"Retrieved {len(records)} history records for user {user_id} (total: {total})"
        )
        
        return {
            "records": records,
            "total": total,
            "page": page,
            "page_size": page_size
        }
    
    async def get_action_summary(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of user actions using aggregation
        
        Provides statistics on action types and outcomes for a user within
        an optional date range.
        
        Args:
            user_id: User ID to get summary for
            start_date: Optional filter for actions after this date
            end_date: Optional filter for actions before this date
            
        Returns:
            Dictionary with action statistics:
            - total_actions: Total number of actions
            - actions_by_type: Count of each action type
            - actions_by_outcome: Count of each outcome (success/error)
        """
        logger.info(
            f"Getting action summary for user {user_id} "
            f"(start_date: {start_date}, end_date: {end_date})"
        )
        
        # Build match stage
        match_stage = {"user_id": user_id}
        
        # Add date range filters
        if start_date or end_date:
            match_stage["timestamp"] = {}
            if start_date:
                match_stage["timestamp"]["$gte"] = start_date
            if end_date:
                match_stage["timestamp"]["$lte"] = end_date
        
        # Aggregation pipeline
        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": None,
                    "total_actions": {"$sum": 1},
                    "actions_by_type": {
                        "$push": "$action_type"
                    },
                    "actions_by_outcome": {
                        "$push": "$outcome"
                    }
                }
            }
        ]
        
        # Execute aggregation
        collection = self.db.db[self.collection]
        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        
        if results:
            result = results[0]
            
            # Count occurrences of each action type
            actions_by_type = {}
            for action_type in result.get("actions_by_type", []):
                actions_by_type[action_type] = actions_by_type.get(action_type, 0) + 1
            
            # Count occurrences of each outcome
            actions_by_outcome = {}
            for outcome in result.get("actions_by_outcome", []):
                actions_by_outcome[outcome] = actions_by_outcome.get(outcome, 0) + 1
            
            return {
                "user_id": user_id,
                "total_actions": result.get("total_actions", 0),
                "actions_by_type": actions_by_type,
                "actions_by_outcome": actions_by_outcome
            }
        else:
            return {
                "user_id": user_id,
                "total_actions": 0,
                "actions_by_type": {},
                "actions_by_outcome": {}
            }
