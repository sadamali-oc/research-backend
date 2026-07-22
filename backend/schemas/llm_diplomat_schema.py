from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class LLMCompareRequest(BaseModel):
    team_size: int = Field(..., ge=2, le=20)
    project_complexity: float = Field(..., description="Raw project_complexity value, e.g. 1.0-9.99")
    n_clusters: int = Field(default=4, ge=2, le=10)
    institution_id: Optional[str] = None
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    n_llm_runs: int = Field(default=3, ge=1, le=10)
    candidate_pool_size: int = Field(default=40, ge=10, le=200,
                                     description="Both formal optimizer and LLM search within this sampled sub-pool, for a fair comparison")
    model: str = Field(default="claude-haiku-4-5-20251001")


class LLMRunResult(BaseModel):
    llm_team: List[str]
    jaccard_overlap_with_formal: float
    llm_team_scored_by_formal_utility: Optional[float] = None
    reasoning: str


class LLMCompareResponse(BaseModel):
    status: str
    sampled_pool_size: Optional[int] = None
    formal_team: Optional[List[str]] = None
    formal_team_utility: Optional[float] = None
    llm_runs: Optional[List[LLMRunResult]] = None
    llm_run_to_run_stability_mean_jaccard: Optional[float] = None
    message: Optional[str] = None