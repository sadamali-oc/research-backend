from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database.database import get_db
from backend.services.prediction_service import PredictionService
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
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    """Train the performance prediction model"""
    service = PredictionService(db)

    try:
        result = service.train_model()
        return TrainModelResponse(
            success=True,
            message="Model trained successfully",
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
    service = PredictionService(db)

    # Check if model is trained
    if not service.model:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please train the model first."
        )

    result = service.predict_employee(request.employee_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Employee {request.employee_id} not found or prediction failed"
        )

    return result

@router.get("/history/{employee_id}", response_model=List[PredictionHistoryResponse])
def get_prediction_history(
        employee_id: str,
        db: Session = Depends(get_db)
):
    """Get prediction history for an employee"""
    service = PredictionService(db)
    history = service.get_employee_history(employee_id)
    return history

@router.get("/stats", response_model=PredictionStatsResponse)
def get_prediction_stats(
        db: Session = Depends(get_db)
):
    """Get prediction statistics"""
    service = PredictionService(db)
    stats = service.get_prediction_stats()
    return stats

@router.get("/all", response_model=List[PredictionHistoryResponse])
def get_all_predictions(
        db: Session = Depends(get_db)
):
    """Get all predictions"""
    service = PredictionService(db)
    predictions = service.get_all_predictions()
    return predictions