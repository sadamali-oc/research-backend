from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class DiplomatRunRequest(BaseModel):
    team_size: int = Field(..., ge=2, le=20)
    project_complexity: float = Field(..., description="Raw project_complexity value, e.g. 1.09-9.86 in current dataset")
    n_clusters: int = Field(default=4, ge=2, le=10)
    institution_id: Optional[str] = None
    beta_min: Optional[float] = None  # if omitted, derived from EmployeePerformanceView min/max
    beta_max: Optional[float] = None
    n_restarts: int = Field(default=20, ge=5, le=100)


class DiplomatUtilityDetail(BaseModel):
    utility: float
    weight: float
    rationale: str


class DiplomatReasoning(BaseModel):
    A_performance: DiplomatUtilityDetail
    B_diversity: DiplomatUtilityDetail
    C_conflict: DiplomatUtilityDetail


class RejectedAlternative(BaseModel):
    team: List[Dict[str, Any]]
    team_utility: float
    utility_gap_vs_best: float


class DiplomatRunResponse(BaseModel):
    status: str
    team_size: Optional[int] = None
    project_complexity_raw: Optional[float] = None
    project_complexity_normalized: Optional[float] = None
    arbitration_weights: Optional[Dict[str, float]] = None
    team: Optional[List[Dict[str, Any]]] = None
    team_utility: Optional[float] = None
    diplomat_reasoning: Optional[DiplomatReasoning] = None
    growth_headroom: Optional[Dict[str, float]] = None
    surplus_capacity: Optional[float] = None
    rejected_alternative: Optional[RejectedAlternative] = None
    message: Optional[str] = None