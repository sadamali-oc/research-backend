import os
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List
import anthropic

from backend.services.diplomat_service import DialogueDiplomatService

PERSONAS = {
    'A_performance': "You argue for maximizing team ability. You want the highest-ability candidates on the team, regardless of diversity or conflict risk.",
    'B_diversity':   "You argue for maximizing sub-culture diversity and Proposition-2 alignment (higher-ability members should also have higher relational scores; avoid mismatched pairs). You are willing to trade some raw ability for a well-composed, aligned team.",
    'C_conflict':    "You argue for minimizing conflict. You prioritize low divergence_score (low historical evaluator disagreement) above raw ability or diversity."
}


class LLMDiplomatService:
    """
    Parallel implementation to DialogueDiplomatService, for direct comparison.
    Three LLM personas each propose a team; a fourth LLM call arbitrates using
    the SAME weight formula as the formal optimizer.
    """

    def __init__(self, pool: pd.DataFrame, model: str = "claude-haiku-4-5-20251001",
                 candidate_pool_size: int = 40):
        """
        model: defaults to Haiku — this is a repeated, simple selection task,
        not one that benefits much from a larger model, and it's called 4x per run.
        candidate_pool_size: instead of sending all 432 employees to the LLM every
        call (expensive, and unnecessary — no persona needs to see everyone), we
        sample a fixed, reproducible subset. Set to len(pool) if you want the full set.
        """
        self.full_pool = pool.reset_index(drop=True)
        self.candidate_pool_size = min(candidate_pool_size, len(self.full_pool))
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _sample_pool(self, team_size: int, seed: int = 42) -> pd.DataFrame:
        """Fixed-seed sample so formal-vs-LLM comparisons stay apples-to-apples
        across repeated calls in the same run."""
        n = max(self.candidate_pool_size, team_size * 4)  # always leave real choice
        return self.full_pool.sample(n=min(n, len(self.full_pool)), random_state=seed)

    @staticmethod
    def _extract_text(response) -> str:
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        raise ValueError(f"No text block found. Block types: {[b.type for b in response.content]}")

    @staticmethod
    def _check_truncation(response):
        if response.stop_reason == "max_tokens":
            raise ValueError(
                "Response was truncated (hit max_tokens before finishing). "
                "Increase max_tokens in the API call, or reduce candidate_pool_size."
            )

    @staticmethod
    def _parse_json_response(text: str) -> Dict:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse JSON from response:\n{text}")

    def _candidate_summary(self, sample_df: pd.DataFrame) -> str:
        cols = ['employee_id', 'cluster_id', 'ability_composite', 'relational_score', 'divergence_score']
        return sample_df[cols].round(3).to_json(orient='records')

    def _call_persona(self, persona_key: str, team_size: int, sample_df: pd.DataFrame) -> Dict:
        prompt = f"""{PERSONAS[persona_key]}

Candidate pool ({len(sample_df)} employees, JSON records):
{self._candidate_summary(sample_df)}

Select exactly {team_size} employee_ids for your proposed team. Respond ONLY with JSON, no other text, no markdown:
{{"team": ["E001", "E002", ...], "rationale": "2-3 sentence argument for this team from your perspective"}}"""

        response = self.client.messages.create(
            model=self.model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        self._check_truncation(response)
        text = self._extract_text(response)
        return self._parse_json_response(text)

    def _call_arbitrator(self, proposals: Dict, weights: Dict, team_size: int) -> Dict:
        prompt = f"""Three diplomats have each proposed a team. Their arbitration weights (based on project complexity) are: {json.dumps(weights)}.

Diplomat A (performance): {json.dumps(proposals['A_performance'])}
Diplomat B (diversity): {json.dumps(proposals['B_diversity'])}
Diplomat C (conflict): {json.dumps(proposals['C_conflict'])}

Synthesize a FINAL team of exactly {team_size} employees, weighing each diplomat's proposal according to their weight. Respond ONLY with JSON, no other text, no markdown:
{{"final_team": ["E001", ...], "reasoning": "explain how you weighed the three proposals to reach this team"}}"""

        response = self.client.messages.create(
            model=self.model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        self._check_truncation(response)
        text = self._extract_text(response)
        return self._parse_json_response(text)

    def form_team(self, team_size: int, project_complexity: float, beta_min: float, beta_max: float) -> Dict:
        beta_norm = float(np.clip((project_complexity - beta_min) / (beta_max - beta_min), 0, 1))
        weights = {'A': round(1 - beta_norm, 3), 'B': round(beta_norm * 0.6, 3), 'C': round(beta_norm * 0.4, 3)}

        sample_df = self._sample_pool(team_size)
        proposals = {key: self._call_persona(key, team_size, sample_df) for key in PERSONAS}
        final = self._call_arbitrator(proposals, weights, team_size)

        return {
            'status': 'success',
            'weights': weights,
            'sampled_pool_ids': sample_df['employee_id'].tolist(),
            'proposals': proposals,
            'final_team': final['final_team'],
            'reasoning': final['reasoning']
        }


def compare_formal_vs_llm(pool: pd.DataFrame, team_size: int, project_complexity: float,
                          beta_min: float, beta_max: float, n_llm_runs: int = 3,
                          candidate_pool_size: int = 40) -> Dict:
    """
    Runs both pipelines on identical inputs. IMPORTANT: to keep the comparison
    fair, the formal optimizer is restricted to the SAME sampled sub-pool the
    LLM sees, not the full 432 — otherwise the formal optimizer has an unfair
    search-space advantage.
    """
    llm_service = LLMDiplomatService(pool, candidate_pool_size=candidate_pool_size)
    sample_df = llm_service._sample_pool(team_size)

    formal_service = DialogueDiplomatService(sample_df)
    formal_result = formal_service.optimize_team(team_size, project_complexity, beta_min, beta_max)
    formal_team_ids = set(m['employee_id'] for m in formal_result['team'])

    llm_runs = []
    for _ in range(n_llm_runs):
        proposals = {key: llm_service._call_persona(key, team_size, sample_df) for key in PERSONAS}
        beta_norm = float(np.clip((project_complexity - beta_min) / (beta_max - beta_min), 0, 1))
        weights = {'A': round(1 - beta_norm, 3), 'B': round(beta_norm * 0.6, 3), 'C': round(beta_norm * 0.4, 3)}
        final = llm_service._call_arbitrator(proposals, weights, team_size)
        llm_team_ids = set(final['final_team'])

        overlap = len(formal_team_ids & llm_team_ids) / len(formal_team_ids | llm_team_ids)

        llm_idx = sample_df.reset_index(drop=True)
        llm_idx = llm_idx[llm_idx['employee_id'].isin(llm_team_ids)].index.to_numpy()
        llm_utility = formal_service._team_utility(llm_idx, weights) if len(llm_idx) == team_size else None

        llm_runs.append({
            'llm_team': sorted(llm_team_ids),
            'jaccard_overlap_with_formal': round(overlap, 3),
            'llm_team_scored_by_formal_utility': round(llm_utility, 4) if llm_utility is not None else None,
            'reasoning': final['reasoning']
        })

    stability_scores = []
    for i in range(len(llm_runs)):
        for j in range(i + 1, len(llm_runs)):
            a, b = set(llm_runs[i]['llm_team']), set(llm_runs[j]['llm_team'])
            stability_scores.append(len(a & b) / len(a | b))

    return {
        'sampled_pool_size': len(sample_df),
        'formal_team': sorted(formal_team_ids),
        'formal_team_utility': formal_result['team_utility'],
        'llm_runs': llm_runs,
        'llm_run_to_run_stability_mean_jaccard': round(float(np.mean(stability_scores)), 3) if stability_scores else None
    }