from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio

from backend.database.database import get_db
from backend.services.prediction_service import PredictionService
from backend.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
    PredictionHistoryResponse,
    PredictionStatsResponse,
    TrainModelResponse
)

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

@router.post("/train", response_model=TrainModelResponse)
async def train_model(
        request: dict,
        db: Session = Depends(get_db)
):
    """Train the hybrid performance prediction model and auto-predict for all employees"""
    service = PredictionService(db)
    algorithm = request.get('algorithm', 'Hybrid')

    try:
        # Step 1: Train the model
        print("="*60)
        print("🚀 STEP 1: Training Model")
        print("="*60)
        result = service.train_models()

        # Step 2: Auto-predict for all employees
        print("\n" + "="*60)
        print("🚀 STEP 2: Auto-Predicting for All Employees")
        print("="*60)

        # Get all employees from history
        from backend.models.history_model import PerformanceHistory
        employees = db.query(PerformanceHistory.employee_id).distinct().all()
        total_employees = len(employees)
        print(f"📊 Found {total_employees} employees to predict")

        # Make predictions for each employee
        predictions_made = 0
        for emp in employees:
            try:
                emp_id = emp[0]
                service.predict_employee(emp_id)
                predictions_made += 1
                if predictions_made % 10 == 0:
                    print(f"  Progress: {predictions_made}/{total_employees}")
            except Exception as e:
                print(f"  ❌ Error for {emp[0]}: {e}")

        print(f"\n✅ Auto-prediction complete: {predictions_made} employees predicted")

        # Step 3: Get stats
        print("\n" + "="*60)
        print("🚀 STEP 3: Getting Statistics")
        print("="*60)
        stats = service.get_prediction_stats()

        print(f"📊 Total Predictions: {stats.get('total_predictions', 0)}")
        print(f"  High: {stats.get('high_performers', 0)}")
        print(f"  Medium: {stats.get('medium_performers', 0)}")
        print(f"  Low: {stats.get('low_performers', 0)}")
        print("="*60)
        print("✅ Training and Auto-Prediction Complete!")
        print("="*60)

        return TrainModelResponse(
            success=True,
            message=f"{algorithm} model trained and {predictions_made} employees predicted successfully",
            accuracy=result.get('accuracy'),
            f1_score=result.get('f1_score'),
            top_features=result.get('top_features', []),
            predictions_count=predictions_made,
            stats=stats
        )
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()
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

    result = service.predict_employee(request.employee_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Employee {request.employee_id} not found or prediction failed"
        )

    return result

@router.get("/employee/{employee_id}", response_model=List[PredictionHistoryResponse])
def get_employee_predictions(
        employee_id: str,
        db: Session = Depends(get_db)
):
    """Get prediction history for an employee"""
    service = PredictionService(db)
    predictions = service.get_employee_predictions(employee_id)
    return predictions

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

@router.post("/predict-all", response_model=dict)
def predict_all_employees(
        db: Session = Depends(get_db)
):
    """Predict performance for all employees"""
    service = PredictionService(db)

    try:
        # Check if models are loaded
        if service.rf_model is None or service.gb_model is None:
            return {
                "success": False,
                "message": "Models not trained. Please train the model first."
            }

        results = service.predict_all_employees()
        stats = service.get_prediction_stats()

        return {
            "success": True,
            "message": f"Predicted for {len(results)} employees",
            "total": len(results),
            "stats": stats,
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Prediction failed: {str(e)}"
        }