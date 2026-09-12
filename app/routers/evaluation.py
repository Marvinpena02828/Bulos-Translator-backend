"""Evaluation endpoints for ISO/IEC 25010 and TAM questionnaires"""
import uuid
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
from utils.device_id import get_device_id
from utils.logging_config import get_logger

logger = get_logger(__name__)

# In-memory store: { device_id: { "iso25010": {...}, "tam": {...} } }
_evaluations: Dict[str, Dict[str, dict]] = {}

router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["Evaluation"],
    responses={
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        500: {"description": "Internal server error"}
    }
)


# ── helpers ──────────────────────────────────────────────────────────────────

def interpret_score(mean_score: float) -> str:
    """
    Interpret mean score based on capstone paper scale (Chapter III, Page 62).

    - 4.51 – 5.00: Highly Acceptable
    - 3.51 – 4.50: Acceptable
    - 2.51 – 3.50: Moderately Acceptable
    - 1.51 – 2.50: Slightly Acceptable
    - 1.00 – 1.50: Not Acceptable
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
    """Calculate mean scores for each numeric characteristic."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    exclude = {"id", "device_id", "evaluation_type", "timestamp", "comments"}

    for ev in evaluations:
        for key, value in ev.items():
            if key not in exclude and isinstance(value, (int, float)):
                sums[key] += value
                counts[key] += 1

    means = {
        k: round(sums[k] / counts[k], 2)
        for k in sums if counts[k] > 0
    }
    if means:
        means["overall_mean"] = round(sum(means.values()) / len(means), 2)
    return means


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/iso25010",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit ISO/IEC 25010 evaluation",
    description=(
        "Submit an ISO/IEC 25010 Software Quality Model evaluation. "
        "Each submission replaces the device's previous ISO/IEC 25010 evaluation."
    )
)
async def submit_iso25010_evaluation(
    evaluation: ISO25010Evaluation,
    device_id: str = Depends(get_device_id),
):
    """
    Submit ISO/IEC 25010 Software Quality Model evaluation.

    Evaluates 6 characteristics (Functional Suitability, Usability,
    Performance Efficiency, Reliability, Maintainability, Portability)
    each on a 5-point Likert scale.
    """
    try:
        evaluation_id = str(uuid.uuid4())
        doc = {
            "id": evaluation_id,
            "device_id": device_id,
            "evaluation_type": "iso25010",
            "timestamp": datetime.utcnow(),
            "functional_suitability": evaluation.functional_suitability,
            "usability": evaluation.usability,
            "performance_efficiency": evaluation.performance_efficiency,
            "reliability": evaluation.reliability,
            "maintainability": evaluation.maintainability,
            "portability": evaluation.portability,
            "comments": evaluation.comments,
        }

        _evaluations.setdefault(device_id, {})["iso25010"] = doc
        logger.info(f"ISO25010 evaluation stored for device {device_id}")

        return EvaluationResponse(
            evaluation_id=evaluation_id,
            message="ISO/IEC 25010 evaluation submitted successfully",
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Failed to submit ISO25010 evaluation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit evaluation: {str(e)}",
        )


@router.post(
    "/tam",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit TAM evaluation",
    description=(
        "Submit a Technology Acceptance Model (TAM) evaluation. "
        "Each submission replaces the device's previous TAM evaluation."
    )
)
async def submit_tam_evaluation(
    evaluation: TAMEvaluation,
    device_id: str = Depends(get_device_id),
):
    """
    Submit TAM evaluation (Perceived Usefulness, Perceived Ease of Use,
    Behavioral Intention to Use) on a 5-point Likert scale.
    """
    try:
        evaluation_id = str(uuid.uuid4())
        doc = {
            "id": evaluation_id,
            "device_id": device_id,
            "evaluation_type": "tam",
            "timestamp": datetime.utcnow(),
            "perceived_usefulness": evaluation.perceived_usefulness,
            "perceived_ease_of_use": evaluation.perceived_ease_of_use,
            "behavioral_intention": evaluation.behavioral_intention,
            "comments": evaluation.comments,
        }

        _evaluations.setdefault(device_id, {})["tam"] = doc
        logger.info(f"TAM evaluation stored for device {device_id}")

        return EvaluationResponse(
            evaluation_id=evaluation_id,
            message="TAM evaluation submitted successfully",
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Failed to submit TAM evaluation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit evaluation: {str(e)}",
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
    device_id: str = Depends(get_device_id),
):
    """
    Get aggregated mean scores and interpretation across all submitted evaluations.
    """
    try:
        iso_evals = [
            v["iso25010"]
            for v in _evaluations.values()
            if "iso25010" in v
        ]
        tam_evals = [
            v["tam"]
            for v in _evaluations.values()
            if "tam" in v
        ]

        iso_results = calculate_mean_scores(iso_evals)
        tam_results = calculate_mean_scores(tam_evals)

        logger.info(
            f"Evaluation results: {len(iso_evals)} ISO25010, {len(tam_evals)} TAM"
        )

        return EvaluationResultsResponse(
            total_evaluations=len(iso_evals) + len(tam_evals),
            iso25010_results=iso_results,
            tam_results=tam_results,
            iso25010_interpretation=interpret_score(iso_results.get("overall_mean", 0.0)),
            tam_interpretation=interpret_score(tam_results.get("overall_mean", 0.0)),
        )

    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}",
        )
