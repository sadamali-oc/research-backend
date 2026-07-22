from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.services.culture_service import CultureClusterService
from backend.schemas.culture_schema import ClusterRunRequest, ClusterRunResponse

router = APIRouter(prefix="/api/culture", tags=["culture"])


@router.post("/cluster", response_model=ClusterRunResponse)
async def run_clustering(request: ClusterRunRequest, db: Session = Depends(get_db)):
    service = CultureClusterService(db, n_clusters=request.n_clusters)
    result = service.run(institution_id=request.institution_id)
    if result['status'] == 'error':
        raise HTTPException(status_code=404, detail=result['message'])

    response = ClusterRunResponse(
        status=result['status'],
        n_employees=result['n_employees'],
        n_clusters=result['n_clusters'],
        demographic_profile=result['demographic_profile'],
        alignment_summary=result['alignment_summary'],
        alignment_significance=result['alignment_significance'],
        opinion_dynamics_profile=result['opinion_dynamics_profile'],
        opinion_dynamics_significance=result['opinion_dynamics_significance'],
        halo_profile=result['halo_profile'],
        halo_significance=result['halo_significance']
    )
    if request.include_employees:
        response.employees = [
            {
                'employee_id': e['employee_id'],
                'cluster_id': e['cluster_id'],
                'ability_composite': e.get('ability_composite'),
                'relational_score': e.get('relational_score'),
                'divergence_score': e.get('divergence_score'),
                'ability_rank': e.get('ability_rank'),
                'relational_rank': e.get('relational_rank'),
                'aligned': e.get('aligned'),
                'performance_score_norm': e.get('performance_score_norm'),
                'halo_gap': e.get('halo_gap'),
                'gender': e.get('gender'),
                'age_group': e.get('age_group'),
                'department': e.get('department'),
                'ethnicity': e.get('ethnicity'),
                'years_of_experience': e.get('years_of_experience'),
            }
            for e in result['employees']
        ]
    return response