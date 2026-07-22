from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ClusterRunRequest(BaseModel):
    institution_id: Optional[str] = None
    n_clusters: int = Field(default=4, ge=2, le=10)
    include_employees: bool = Field(default=True)


class ClusterAlignmentSummary(BaseModel):
    cluster_id: int
    n: int
    pct_aligned: float
    mean_ability: float
    mean_relational: float


class DemographicProfileItem(BaseModel):
    cluster_id: int
    size: int
    gender: Dict[str, int] = {}
    age_group: Dict[str, int] = {}
    department: Dict[str, int] = {}
    ethnicity: Dict[str, int] = {}
    language: Dict[str, int] = {}
    mean_years_experience: Optional[float] = None


class OpinionDynamicsProfileItem(BaseModel):
    cluster_id: int
    n: int
    mean_divergence: float
    median_divergence: Optional[float] = None
    std_divergence: Optional[float] = None
    pct_high_divergence: Optional[float] = None


class HaloProfileItem(BaseModel):
    cluster_id: int
    n: int
    mean_ability_composite: float
    mean_performance_score_norm: float
    mean_halo_gap: float


class AlignmentSignificance(BaseModel):
    chi2: Optional[float] = None
    dof: Optional[int] = None
    p_value: float
    significant: bool
    test: str = "chi-square"


class OpinionDynamicsSignificance(BaseModel):
    h_statistic: Optional[float] = None
    p_value: float
    significant: bool
    test: str = "kruskal-wallis"


class HaloSignificance(BaseModel):
    h_statistic: Optional[float] = None
    p_value: float
    significant: bool
    test: str = "kruskal-wallis"


class ClusterRunResponse(BaseModel):
    status: str
    n_employees: Optional[int] = None
    n_clusters: Optional[int] = None
    demographic_profile: Optional[List[DemographicProfileItem]] = None
    alignment_summary: Optional[List[ClusterAlignmentSummary]] = None
    alignment_significance: Optional[AlignmentSignificance] = None
    opinion_dynamics_profile: Optional[List[OpinionDynamicsProfileItem]] = None
    opinion_dynamics_significance: Optional[OpinionDynamicsSignificance] = None
    halo_profile: Optional[List[HaloProfileItem]] = None
    halo_significance: Optional[HaloSignificance] = None
    employees: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None