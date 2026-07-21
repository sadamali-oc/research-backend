# backend/schemas/feedback_schema.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# -------- Feedback 360 Schemas --------
class Feedback360Base(BaseModel):
    evaluation_id: str
    evaluatee_id: str
    evaluator_id: str
    evaluation_type: str
    relationship: Optional[str] = None
    institution_id: str
    period_year: int
    period_quarter: int
    hierarchy_distance: Optional[int] = 0
    punctuality: float = Field(ge=1, le=5)
    problem_solving: float = Field(ge=1, le=5)
    leadership: float = Field(ge=1, le=5)
    collaboration: float = Field(ge=1, le=5)
    communication: float = Field(ge=1, le=5)
    overall_rating: float = Field(ge=1, le=5)
    feedback_text: Optional[str] = None
    created_at: Optional[datetime] = None

class Feedback360Create(Feedback360Base):
    pass

class Feedback360Response(Feedback360Base):
    id: int

    class Config:
        from_attributes = True

# -------- Aggregated Performance Schemas --------
class AggregatedPerformanceBase(BaseModel):
    employee_id: str
    institution_id: str
    period_year: int
    period_quarter: int
    self_punctuality: Optional[float] = None
    self_problem_solving: Optional[float] = None
    self_leadership: Optional[float] = None
    self_collaboration: Optional[float] = None
    self_communication: Optional[float] = None
    self_overall: Optional[float] = None
    manager_punctuality: Optional[float] = None
    manager_problem_solving: Optional[float] = None
    manager_leadership: Optional[float] = None
    manager_collaboration: Optional[float] = None
    manager_communication: Optional[float] = None
    manager_overall: Optional[float] = None
    peer_punctuality: Optional[float] = None
    peer_problem_solving: Optional[float] = None
    peer_leadership: Optional[float] = None
    peer_collaboration: Optional[float] = None
    peer_communication: Optional[float] = None
    peer_overall: Optional[float] = None
    sub_punctuality: Optional[float] = None
    sub_problem_solving: Optional[float] = None
    sub_leadership: Optional[float] = None
    sub_collaboration: Optional[float] = None
    sub_communication: Optional[float] = None
    sub_overall: Optional[float] = None
    performance_score: Optional[float] = None
    relational_score: Optional[float] = None
    divergence_score: Optional[float] = None
    self_count: Optional[int] = 0
    manager_count: Optional[int] = 0
    peer_count: Optional[int] = 0
    sub_count: Optional[int] = 0
    pdi_institution: Optional[float] = None

class AggregatedPerformanceResponse(AggregatedPerformanceBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# -------- Preprocessing Request/Response --------
class PreprocessRequest(BaseModel):
    """Request to trigger preprocessing"""
    institution_id: Optional[str] = None
    period_year: Optional[int] = None
    period_quarter: Optional[int] = None

class PreprocessResponse(BaseModel):
    status: str
    message: str = Field(default="Preprocessing completed successfully")
    records_processed: int
    employees_processed: int
    quarters_processed: int
    institutions_processed: int
    aggregated_records: Optional[int] = 0

# -------- Divergence Analysis --------
class DivergenceAnalysisResponse(BaseModel):
    employee_id: str
    period_year: int
    period_quarter: int
    divergence_score: float
    manager_peer_correlation: Optional[float] = None
    manager_sub_correlation: Optional[float] = None
    pdi_institution: Optional[float] = None
    interpretation: str  # "High Power Distance Influence" or "Low Power Distance Influence"

# -------- Statistics --------
class FeedbackStatsResponse(BaseModel):
    total_evaluations: int
    total_employees: int
    total_institutions: int
    periods_available: List[dict]  # [{"year": 2024, "quarter": 1}, ...]
    evaluation_type_distribution: dict
    competency_stats: dict  # {"punctuality": {"min": 1.0, "max": 5.0, "mean": 3.5}, ...}