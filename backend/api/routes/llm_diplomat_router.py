from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json

from backend.database.database import get_db
from backend.services.llm_diplomat_service import compare_formal_vs_llm
from backend.services.pool_builder import build_diplomat_pool, get_project_complexity_range
from backend.schemas.llm_diplomat_schema import LLMCompareRequest, LLMCompareResponse

router = APIRouter(prefix="/api/llm-diplomats", tags=["llm-diplomats"])


@router.post("/compare", response_model=LLMCompareResponse)
async def compare_formal_vs_llm_endpoint(request: LLMCompareRequest, db: Session = Depends(get_db)):
    """
    Runs both the formal optimizer and the LLM-negotiated approach on identical
    inputs, restricted to the same sampled candidate pool for a fair comparison.
    Makes real Anthropic API calls (n_llm_runs x 4 calls each) — costs apply.
    """
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
        result = compare_formal_vs_llm(
            pool=pool,
            team_size=request.team_size,
            project_complexity=request.project_complexity,
            beta_min=beta_min,
            beta_max=beta_max,
            n_llm_runs=request.n_llm_runs,
            candidate_pool_size=request.candidate_pool_size
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"LLM response could not be parsed: {str(e)}")
    except Exception as e:
        # covers anthropic API errors (auth, rate limit, etc.) without leaking internals
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(e)}")

    return LLMCompareResponse(status='success', **result)