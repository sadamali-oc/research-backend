from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Union, Tuple
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for making a prediction"""
    employee_id: str

class FeatureImportance(BaseModel):
    """Feature importance model"""
    feature: str
    importance: float

class PredictionResponse(BaseModel):
    """Response model for a single prediction"""
    employee_id: str
    performance_band: str  # High / Medium / Low
    performance_score: float
    confidence: float
    top_features: List[Tuple[str, float]]  # List of (feature_name, importance)
    predicted_at: datetime

class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history"""
    employee_id: str
    performance_band: str
    performance_score: float
    confidence: float
    predicted_at: datetime

class PredictionStatsResponse(BaseModel):
    """Response model for prediction statistics"""
    total_predictions: int
    band_distribution: Dict[str, int]
    average_score: float
    high_performers: int
    medium_performers: int
    low_performers: int

class TrainModelResponse(BaseModel):
    """Response model for model training"""
    success: bool
    message: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    top_features: Optional[List[FeatureImportance]] = None