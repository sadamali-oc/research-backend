from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.services.diplomat_service import DialogueDiplomatService
from backend.services.pool_builder import build_diplomat_pool, get_project_complexity_range
from backend.schemas.diplomat_schema import DiplomatRunRequest, DiplomatRunResponse

router = APIRouter(prefix="/api/diplomats", tags=["diplomats"])


@router.post("/optimize-team", response_model=DiplomatRunResponse)
async def optimize_team(request: DiplomatRunRequest, db: Session = Depends(get_db)):
    try:
        pool = build_diplomat_pool(db, n_clusters=request.n_clusters, institution_id=request.institution_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    beta_min, beta_max = request.beta_min, request.beta_max
    if beta_min is None or beta_max is None:
        derived_min, derived_max = get_project_complexity_range(db)
        beta_min = beta_min if beta_min is not None else derived_min
        beta_max = beta_max if beta_max is not None else derived_max

    try:
        service = DialogueDiplomatService(pool)
        result = service.optimize_team(
            team_size=request.team_size,
            project_complexity=request.project_complexity,
            beta_min=beta_min,
            beta_max=beta_max,
            n_restarts=request.n_restarts
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return DiplomatRunResponse(**result)