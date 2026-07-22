import pandas as pd
from sqlalchemy.orm import Session
from typing import Optional

from backend.services.culture_service import CultureClusterService


def build_diplomat_pool(db: Session, n_clusters: int = 4, institution_id: Optional[str] = None) -> pd.DataFrame:
    """
    Runs Step 2 clustering and returns the flat pool DialogueDiplomatService needs:
    employee_id, cluster_id, ability_composite, relational_score, divergence_score.
    Restricted to the 432 employees with Feedback_360 coverage, per agreed scope.
    """
    service = CultureClusterService(db, n_clusters=n_clusters)
    result = service.run(institution_id=institution_id)
    if result['status'] == 'error':
        raise ValueError(result['message'])

    df = pd.DataFrame(result['employees'])
    return df[['employee_id', 'cluster_id', 'ability_composite', 'relational_score', 'divergence_score']]


def get_project_complexity_range(db: Session) -> tuple:
    """Pulls the observed min/max of project_complexity from the panel data, for beta normalization defaults."""
    from backend.models.employee_model import EmployeePerformanceView
    from sqlalchemy import func
    result = db.query(
        func.min(EmployeePerformanceView.project_complexity),
        func.max(EmployeePerformanceView.project_complexity)
    ).first()
    beta_min, beta_max = result[0], result[1]
    if beta_min is None or beta_max is None:
        return 0.0, 1.0
    return float(beta_min), float(beta_max)