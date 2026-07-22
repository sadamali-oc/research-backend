from pydantic import BaseModel
from typing import Optional, Dict, List, Any


class TrainPDIRiskResponse(BaseModel):
    status: str
    n_samples: Optional[int] = None
    label_distribution: Optional[Dict[str, int]] = None
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    classification_report: Optional[Dict[str, Any]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None


class PDIRiskPredictRequest(BaseModel):
    years_of_experience: float
    gender: str
    age_group: str
    department: str
    job_role: str
    ethnicity: str


class PDIRiskPredictResponse(BaseModel):
    predicted_risk: str
    probabilities: Dict[str, float]