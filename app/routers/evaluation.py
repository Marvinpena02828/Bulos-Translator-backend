"""Evaluation endpoints for ISO/IEC 25010 and TAM questionnaires"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict
from datetime import datetime
from collections import defaultdict

from models.schemas import (
    ISO25010Evaluation,
    TAMEvaluation,
    EvaluationResponse,
    EvaluationResultsResponse
)
from services.database import get_db_manager, DatabaseManager
from services.auth import get_current_user
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["Evaluation"],
    responses={
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        500: {"description": "Internal server error"}
    }
)


def interpret_score(mean_score: float) -> str:
    """
    Interpret mean score based on capstone paper scale (Chapter III, Page 62).
    
    Interpretation Scale:
    - 4.51 - 5.00: Highly Acceptable
    - 3.51 - 4.50: Acceptable
    - 2.51 - 3.50: Moderately Acceptable
    - 1.51 - 2.50: Slightly Acceptable
    - 1.00 - 1.50: Not Acceptable
    
    Args:
        mean_score: Average score from evaluations (1.0-5.0)
        
    Returns:
        str: Interpretation label
    """
    if mean_score >= 4.51:
        return "Highly Acceptable"
    elif mean_score >= 3.51:
        return "Acceptable"
    elif mean_score >= 2.51:
        return "Moderately Acceptable"
    elif mean_score >= 1.51:
        return "Slightly Acceptable"
    else:
        return "Not Acceptable"


def calculate_mean_scores(evaluations: list) -> Dict[str, float]:
    """
    Calculate mean scores for each characteristic from a list of evaluations.
    
    Args:
        evaluations: List of evaluation documents from database
        
    Returns:
        Dict[str, float]: Dictionary mapping characteristic names to mean scores
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    
    # Fields to exclude from mean calculation
    exclude_fields = {'_id', 'user_id', 'evaluation_type', 'timestamp', 'comments'}
    
    for evaluation in evaluations:
        for key, value in evaluation.items():
            if key not in exclude_fields and isinstance(value, (int, float)):
                sums[key] += value
                counts[key] += 1
    
    # Calculate means and round to 2 decimal places
    means = {
        key: round(sums[key] / counts[key], 2)
        for key in sums
        if counts[key] > 0
    }
    
    # Calculate overall mean if there are any scores
    if means:
        overall_mean = round(sum(means.values()) / len(means), 2)
        means['overall_mean'] = overall_mean
    
    return means


@router.post(
    "/iso25010",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit ISO/IEC 25010 evaluation",
    description=(
        "Submit an ISO/IEC 25010 Software Quality Model evaluation. "
        "Users can submit multiple evaluations to capture evolving opinions. "
        "Each submission updates their previous ISO/IEC 25010 evaluation."
    )
)
async def submit_iso25010_evaluation(
    evaluation: ISO25010Evaluation,
    current_user: dict = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Submit ISO/IEC 25010 Software Quality Model evaluation.
    
    Evaluates 6 characteristics:
    - Functional Suitability
    - Usability
    - Performance Efficiency
    - Reliability
    - Maintainability
    - Portability
    
    Each rated on 5-point Likert scale (1=Strongly Disagree, 5=Strongly Agree).
    """
    try:
        user_id = current_user["_id"]
        
        # Prepare evaluation document
        evaluation_doc = {
            "user_id": user_id,
            "evaluation_type": "iso25010",
            "timestamp": datetime.utcnow(),
            "functional_suitability": evaluation.functional_suitability,
            "usability": evaluation.usability,
            "performance_efficiency": evaluation.performance_efficiency,
            "reliability": evaluation.reliability,
            "maintainability": evaluation.maintainability,
            "portability": evaluation.portability,
            "comments": evaluation.comments
        }
        
        # Use update_one with upsert=True to allow re-submission
        # This replaces the user's previous ISO25010 evaluation if it exists
        filter_query = {"user_id": user_id, "evaluation_type": "iso25010"}
        
        result = await db_manager.db.evaluations.update_one(
            filter_query,
            {"$set": evaluation_doc},
            upsert=True
        )
        
        # Get the evaluation ID
        if result.upserted_id:
            evaluation_id = str(result.upserted_id)
            logger.info(f"New ISO25010 evaluation created for user {user_id}")
        else:
            # Find the existing document to get its ID
            existing = await db_manager.find_one("evaluations", filter_query)
            evaluation_id = str(existing["_id"]) if existing else "unknown"
            logger.info(f"ISO25010 evaluation updated for user {user_id}")
        
        # Log to history
        await db_manager.insert_one("history", {
            "user_id": user_id,
            "action_type": "evaluation_submission",
            "resource_type": "evaluation",
            "resource_id": evaluation_id,
            "outcome": "success",
            "timestamp": datetime.utcnow(),
            "details": {
                "evaluation_type": "iso25010",
                "ratings": {
                    "functional_suitability": evaluation.functional_suitability,
                    "usability": evaluation.usability,
                    "performance_efficiency": evaluation.performance_efficiency,
                    "reliability": evaluation.reliability,
                    "maintainability": evaluation.maintainability,
                    "portability": evaluation.portability
                }
            }
        })
        
        return EvaluationResponse(
            evaluation_id=evaluation_id,
            message="ISO/IEC 25010 evaluation submitted successfully",
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to submit ISO25010 evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit evaluation: {str(e)}"
        )


@router.post(
    "/tam",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit TAM evaluation",
    description=(
        "Submit a Technology Acceptance Model (TAM) evaluation. "
        "Users can submit multiple evaluations to capture evolving opinions. "
        "Each submission updates their previous TAM evaluation."
    )
)
async def submit_tam_evaluation(
    evaluation: TAMEvaluation,
    current_user: dict = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Submit Technology Acceptance Model (TAM) evaluation.
    
    Evaluates 3 constructs:
    - Perceived Usefulness (PU)
    - Perceived Ease of Use (PEOU)
    - Behavioral Intention to Use (BI)
    
    Each rated on 5-point Likert scale (1=Strongly Disagree, 5=Strongly Agree).
    """
    try:
        user_id = current_user["_id"]
        
        # Prepare evaluation document
        evaluation_doc = {
            "user_id": user_id,
            "evaluation_type": "tam",
            "timestamp": datetime.utcnow(),
            "perceived_usefulness": evaluation.perceived_usefulness,
            "perceived_ease_of_use": evaluation.perceived_ease_of_use,
            "behavioral_intention": evaluation.behavioral_intention,
            "comments": evaluation.comments
        }
        
        # Use update_one with upsert=True to allow re-submission
        # This replaces the user's previous TAM evaluation if it exists
        filter_query = {"user_id": user_id, "evaluation_type": "tam"}
        
        result = await db_manager.db.evaluations.update_one(
            filter_query,
            {"$set": evaluation_doc},
            upsert=True
        )
        
        # Get the evaluation ID
        if result.upserted_id:
            evaluation_id = str(result.upserted_id)
            logger.info(f"New TAM evaluation created for user {user_id}")
        else:
            # Find the existing document to get its ID
            existing = await db_manager.find_one("evaluations", filter_query)
            evaluation_id = str(existing["_id"]) if existing else "unknown"
            logger.info(f"TAM evaluation updated for user {user_id}")
        
        # Log to history
        await db_manager.insert_one("history", {
            "user_id": user_id,
            "action_type": "evaluation_submission",
            "resource_type": "evaluation",
            "resource_id": evaluation_id,
            "outcome": "success",
            "timestamp": datetime.utcnow(),
            "details": {
                "evaluation_type": "tam",
                "ratings": {
                    "perceived_usefulness": evaluation.perceived_usefulness,
                    "perceived_ease_of_use": evaluation.perceived_ease_of_use,
                    "behavioral_intention": evaluation.behavioral_intention
                }
            }
        })
        
        return EvaluationResponse(
            evaluation_id=evaluation_id,
            message="TAM evaluation submitted successfully",
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to submit TAM evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit evaluation: {str(e)}"
        )


@router.get(
    "/results",
    response_model=EvaluationResultsResponse,
    summary="Get aggregated evaluation results",
    description=(
        "Retrieve aggregated evaluation results for research analysis. "
        "Calculates mean scores for all ISO/IEC 25010 and TAM characteristics "
        "and provides interpretation based on the capstone paper scale."
    )
)
async def get_evaluation_results(
    current_user: dict = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Get aggregated evaluation results with mean scores and interpretations.
    
    Returns:
    - Total number of evaluations
    - Mean scores for ISO/IEC 25010 characteristics
    - Mean scores for TAM constructs
    - Interpretation for both evaluation types
    """
    try:
        # Fetch all ISO/IEC 25010 evaluations
        iso25010_evaluations = await db_manager.find_many(
            "evaluations",
            {"evaluation_type": "iso25010"},
            limit=10000  # High limit to get all evaluations
        )
        
        # Fetch all TAM evaluations
        tam_evaluations = await db_manager.find_many(
            "evaluations",
            {"evaluation_type": "tam"},
            limit=10000  # High limit to get all evaluations
        )
        
        # Calculate mean scores
        iso25010_results = calculate_mean_scores(iso25010_evaluations)
        tam_results = calculate_mean_scores(tam_evaluations)
        
        # Get overall means for interpretation
        iso25010_mean = iso25010_results.get('overall_mean', 0.0)
        tam_mean = tam_results.get('overall_mean', 0.0)
        
        # Interpret scores
        iso25010_interpretation = interpret_score(iso25010_mean)
        tam_interpretation = interpret_score(tam_mean)
        
        # Calculate total evaluations (unique users)
        total_evaluations = len(iso25010_evaluations) + len(tam_evaluations)
        
        logger.info(
            f"Evaluation results retrieved: {len(iso25010_evaluations)} ISO25010, "
            f"{len(tam_evaluations)} TAM evaluations"
        )
        
        return EvaluationResultsResponse(
            total_evaluations=total_evaluations,
            iso25010_results=iso25010_results,
            tam_results=tam_results,
            iso25010_interpretation=iso25010_interpretation,
            tam_interpretation=tam_interpretation
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}"
        )
