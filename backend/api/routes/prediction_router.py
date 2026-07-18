from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database.database import get_db
from backend.services.hybrid_model_service import HybridModelService
from backend.schemas.performance_schema import (
    PredictionRequest,
    PredictionResponse,
    PredictionHistoryResponse,
    PredictionStatsResponse,
    TrainModelResponse
)

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

@router.post("/train", response_model=TrainModelResponse)
def train_model(
        request: dict,
        db: Session = Depends(get_db)
):
    """Train the hybrid performance prediction model"""
    service = HybridModelService(db)
    algorithm = request.get('algorithm', 'Hybrid')

    try:
        result = service.train_models()
        return TrainModelResponse(
            success=True,
            message=f"{algorithm} model trained successfully",
            accuracy=result.get('accuracy'),
            f1_score=result.get('f1_score'),
            top_features=result.get('top_features', [])
        )
    except Exception as e:
        return TrainModelResponse(
            success=False,
            message=f"Training failed: {str(e)}"
        )

@router.post("/predict", response_model=PredictionResponse)
def predict_employee(
        request: PredictionRequest,
        db: Session = Depends(get_db)
):
    """Predict performance for a specific employee"""
    service = HybridModelService(db)

    result = service.predict_employee(request.employee_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Employee {request.employee_id} not found or prediction failed"
        )

    return result

@router.get("/employee/{employee_id}")
def get_employee_predictions(
        employee_id: str,
        db: Session = Depends(get_db)
):
    """Get prediction history for an employee"""
    service = HybridModelService(db)
    predictions = service.get_employee_predictions(employee_id)
    return predictions

@router.get("/history/{employee_id}")
def get_employee_history(
        employee_id: str,
        db: Session = Depends(get_db)
):
    """Get historical performance data for an employee"""
    service = HybridModelService(db)
    history = service.get_employee_history(employee_id)
    return history

@router.get("/stats", response_model=PredictionStatsResponse)
def get_prediction_stats(
        db: Session = Depends(get_db)
):
    """Get prediction statistics"""
    service = HybridModelService(db)

    # Get all predictions
    predictions = db.query(PerformancePrediction).all()

    if not predictions:
        return PredictionStatsResponse(
            total_predictions=0,
            band_distribution={'High': 0, 'Medium': 0, 'Low': 0},
            average_score=0,
            average_confidence=0,
            high_performers=0,
            medium_performers=0,
            low_performers=0
        )

    # Calculate stats
    band_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    total_score = 0
    total_confidence = 0

    for p in predictions:
        band_counts[p.predicted_band] = band_counts.get(p.predicted_band, 0) + 1
        total_score += p.predicted_score
        total_confidence += p.confidence

    return PredictionStatsResponse(
        total_predictions=len(predictions),
        band_distribution=band_counts,
        average_score=total_score / len(predictions),
        average_confidence=total_confidence / len(predictions),
        high_performers=band_counts.get('High', 0),
        medium_performers=band_counts.get('Medium', 0),
        low_performers=band_counts.get('Low', 0)
    )