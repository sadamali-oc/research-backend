from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.services.pdi_risk_model_service import PDIRiskModelService
from backend.schemas.pdi_risk_schema import (
    TrainPDIRiskResponse, PDIRiskPredictRequest, PDIRiskPredictResponse
)

router = APIRouter(prefix="/api/pdi-risk", tags=["pdi-risk"])


@router.post("/train", response_model=TrainPDIRiskResponse)
async def train_pdi_risk_model(db: Session = Depends(get_db)):
    """Train the supervised PDI-risk classifier (RandomForest) on current data."""
    service = PDIRiskModelService(db)
    result = service.train()
    if result['status'] == 'error':
        raise HTTPException(status_code=404, detail=result['message'])
    return TrainPDIRiskResponse(**result)


@router.post("/predict", response_model=PDIRiskPredictResponse)
async def predict_pdi_risk(request: PDIRiskPredictRequest, db: Session = Depends(get_db)):
    """Predict PDI-risk category for a hypothetical/new employee profile."""
    service = PDIRiskModelService(db)
    try:
        result = service.predict(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PDIRiskPredictResponse(**result)