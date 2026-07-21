# backend/api/routes/feedback_router.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import Optional, List
import tempfile
import os

from backend.database.database import get_db
from backend.services.feedback_service import FeedbackService
from backend.schemas.feedback_schema import (
    PreprocessRequest, PreprocessResponse,
    AggregatedPerformanceResponse,
    DivergenceAnalysisResponse,
    FeedbackStatsResponse,
    Feedback360Create, Feedback360Response
)
from backend.models.feedback_model import Feedback360

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

# -------- Data Import --------
@router.post("/import-csv")
async def import_csv(
        file: UploadFile = File(...),
        institution_id: Optional[str] = None,
        db: Session = Depends(get_db)
):
    """
    Import feedback data from CSV file
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        service = FeedbackService(db)
        result = service.import_csv_data(tmp_path, institution_id)
        return result
    finally:
        os.unlink(tmp_path)

# -------- Preprocessing --------
@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(
        request: PreprocessRequest,
        db: Session = Depends(get_db)
):
    """
    Preprocess feedback data:
    - Aggregate 360-degree feedback
    - Calculate performance scores
    - Compute divergence and PDI proxies
    """
    service = FeedbackService(db)
    result = service.aggregate_performance(
        institution_id=request.institution_id,
        period_year=request.period_year,
        period_quarter=request.period_quarter
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))

    return PreprocessResponse(
        status=result.get("status", "success"),
        message=result.get("message", "Preprocessing completed successfully"),
        records_processed=result.get("records_processed", 0),
        employees_processed=result.get("employees_processed", 0),
        quarters_processed=result.get("quarters_processed", 0),
        institutions_processed=result.get("institutions_processed", 0),
        aggregated_records=result.get("aggregated_records", 0)
    )

# -------- Query Methods --------
@router.get("/aggregated", response_model=List[AggregatedPerformanceResponse])
async def get_aggregated_data(
        employee_id: Optional[str] = None,
        institution_id: Optional[str] = None,
        period_year: Optional[int] = None,
        period_quarter: Optional[int] = None,
        db: Session = Depends(get_db)
):
    """Get aggregated performance data"""
    service = FeedbackService(db)
    results = service.get_aggregated_data(
        employee_id=employee_id,
        institution_id=institution_id,
        period_year=period_year,
        period_quarter=period_quarter
    )
    return results

@router.get("/divergence", response_model=List[DivergenceAnalysisResponse])
async def get_divergence_analysis(
        employee_id: Optional[str] = None,
        institution_id: Optional[str] = None,
        db: Session = Depends(get_db)
):
    """Get divergence analysis (Step 1 output)"""
    service = FeedbackService(db)
    results = service.get_divergence_analysis(
        employee_id=employee_id,
        institution_id=institution_id
    )
    return results

@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
        db: Session = Depends(get_db)
):
    """Get statistics about feedback data"""
    service = FeedbackService(db)
    stats = service.get_stats()
    return stats

# -------- Raw Feedback --------
@router.get("/raw", response_model=List[Feedback360Response])
async def get_raw_feedback(
        evaluatee_id: Optional[str] = None,
        evaluator_id: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        period_year: Optional[int] = None,
        period_quarter: Optional[int] = None,
        limit: int = 100,
        db: Session = Depends(get_db)
):
    """Get raw feedback records"""
    query = db.query(Feedback360)
    if evaluatee_id:
        query = query.filter(Feedback360.evaluatee_id == evaluatee_id)
    if evaluator_id:
        query = query.filter(Feedback360.evaluator_id == evaluator_id)
    if evaluation_type:
        query = query.filter(Feedback360.evaluation_type == evaluation_type)
    if period_year:
        query = query.filter(Feedback360.period_year == period_year)
    if period_quarter:
        query = query.filter(Feedback360.period_quarter == period_quarter)

    results = query.limit(limit).all()
    return results