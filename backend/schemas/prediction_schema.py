from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class PredictionRequest(BaseModel):
    employee_id: str

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class PredictionResponse(BaseModel):
    employee_id: str
    predicted_band: str
    predicted_score: float
    confidence: float
    next_quarter: str
    rf_prediction: Optional[str] = None
    rf_confidence: Optional[float] = None
    gb_prediction: Optional[str] = None
    gb_confidence: Optional[float] = None
    feature_importance: Dict[str, float]

class PredictionHistoryResponse(BaseModel):
    employee_id: str
    period: str
    year: int
    quarter: str
    predicted_band: str
    predicted_score: float
    confidence: float
    algorithm: str
    actual_score: Optional[float] = None
    actual_band: Optional[str] = None
    predicted_at: datetime
    # Remove 'evaluated' field - it's not in the data

class PredictionStatsResponse(BaseModel):
    total_predictions: int
    band_distribution: Dict[str, int]
    average_score: float
    average_confidence: float
    high_performers: int
    medium_performers: int
    low_performers: int

class TrainModelResponse(BaseModel):
    success: bool
    message: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    top_features: Optional[List[FeatureImportance]] = None
    predictions_count: Optional[int] = None
    stats: Optional[PredictionStatsResponse] = None