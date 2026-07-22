# backend/services/diplomat_service.py
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DialogueDiplomatService:
    """Step 3: Multi-agent team formation via weighted-utility arbitration (CAPAF core module)"""

    def __init__(self, employee_pool: pd.DataFrame):
        """
        employee_pool must contain: employee_id, performance_score, relational_score,
        cluster_id, divergence_score
        """
        self.pool = employee_pool.reset_index(drop=True)
        self.max_perf = self.pool['performance_score'].max()

    # ---------- Diplomat A: Performance ----------
    def utility_a(self, team: pd.DataFrame) -> float:
        return team['performance_score'].mean()

    # ---------- Diplomat B: Diversity + alignment (Prop 1 & 2) ----------
    def utility_b(self, team: pd.DataFrame, lam: float = 0.5) -> float:
        # Blau index over cluster membership (higher = more diverse)
        proportions = team['cluster_id'].value_counts(normalize=True)
        blau = 1 - (proportions ** 2).sum()

        # Faultline / alignment penalty: for every pair, mismatch between
        # ability ordering and relational ordering is an anti-aligned pair (Prop 2)
        pairs = list(itertools.combinations(team.index, 2))
        if not pairs:
            return blau
        mismatches = 0
        for i, j in pairs:
            a_i, a_j = team.loc[i, 'performance_score'], team.loc[j, 'performance_score']
            r_i, r_j = team.loc[i, 'relational_score'], team.loc[j, 'relational_score']
            if np.sign(a_i - a_j) != np.sign(r_i - r_j) and a_i != a_j and r_i != r_j:
                mismatches += 1
        anti_alignment_rate = mismatches / len(pairs)

        # We want diversity (blau) HIGH but anti-alignment LOW
        return blau - lam * anti_alignment_rate

    # ---------- Diplomat C: Conflict minimization ----------
    def utility_c(self, team: pd.DataFrame) -> float:
        # lower mean divergence_score = lower conflict; return negative so "higher is better"
        return -team['divergence_score'].mean()

    def _normalized_beta(self, project_complexity: float, beta_min: float, beta_max: float) -> float:
        if beta_max == beta_min:
            return 0.5
        return float(np.clip((project_complexity - beta_min) / (beta_max - beta_min), 0, 1))

    def arbitration_weights(self, beta_norm: float) -> Dict[str, float]:
        w_a = 1 - beta_norm
        w_b = beta_norm * 0.6
        w_c = beta_norm * 0.4
        return {'A': w_a, 'B': w_b, 'C': w_c}

    def team_utility(self, team: pd.DataFrame, weights: Dict[str, float]) -> float:
        return (weights['A'] * self.utility_a(team)
                + weights['B'] * self.utility_b(team)
                + weights['C'] * self.utility_c(team))

    def optimize_team(self, team_size: int, project_complexity: float,
                      beta_min: float = 0.0, beta_max: float = 1.0,
                      n_restarts: int = 20) -> Dict:
        """
        Greedy local-search over the combinatorial team space.
        Multiple random restarts + hill-climbing swaps to approximate the optimum.
        """
        beta_norm = self._normalized_beta(project_complexity, beta_min, beta_max)
        weights = self.arbitration_weights(beta_norm)

        best_team, best_score = None, -np.inf

        for _ in range(n_restarts):
            candidate_idx = list(np.random.choice(self.pool.index, size=team_size, replace=False))
            candidate = self.pool.loc[candidate_idx]
            score = self.team_utility(candidate, weights)

            improved = True
            while improved:
                improved = False
                outside = self.pool.index.difference(candidate_idx)
                for out_member in list(candidate_idx):
                    for in_member in outside:
                        trial_idx = [in_member if m == out_member else m for m in candidate_idx]
                        trial = self.pool.loc[trial_idx]
                        trial_score = self.team_utility(trial, weights)
                        if trial_score > score:
                            candidate_idx, score = trial_idx, trial_score
                            improved = True
                            break
                    if improved:
                        break

            if score > best_score:
                best_score, best_team = score, self.pool.loc[candidate_idx].copy()

        growth_headroom = (self.max_perf - best_team['performance_score']) / self.max_perf
        surplus_capacity = float(growth_headroom.mean())

        return {
            'status': 'success',
            'project_complexity_normalized': beta_norm,
            'weights': weights,
            'team': best_team.to_dict('records'),
            'team_utility': best_score,
            'growth_headroom': growth_headroom.to_dict(),
            'surplus_capacity': surplus_capacity
        }