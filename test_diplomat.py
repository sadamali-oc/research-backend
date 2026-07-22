# test_low_complexity.py
from dotenv import load_dotenv
load_dotenv()

from backend.database.database import SessionLocal
from backend.services.pool_builder import build_diplomat_pool, get_project_complexity_range
from backend.services.llm_diplomat_service import compare_formal_vs_llm

db = SessionLocal()
pool = build_diplomat_pool(db, n_clusters=4)
beta_min, beta_max = get_project_complexity_range(db)

result = compare_formal_vs_llm(
    pool, team_size=5, project_complexity=beta_min,  # low complexity this time
    beta_min=beta_min, beta_max=beta_max, n_llm_runs=3
)

print("Formal team:", result['formal_team'], "utility:", result['formal_team_utility'])
for i, run in enumerate(result['llm_runs']):
    print(f"\nRun {i+1}: overlap={run['jaccard_overlap_with_formal']}, "
          f"llm_utility={run['llm_team_scored_by_formal_utility']}")
print("\nStability:", result['llm_run_to_run_stability_mean_jaccard'])
db.close()