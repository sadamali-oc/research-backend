# backend/services/diplomat_service.py
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DialogueDiplomatService:
    """
    Step 3: Multi-agent team formation via weighted-utility arbitration.
    Hot-path optimization uses raw numpy arrays (not pandas .loc) — the greedy
    swap search evaluates ~10^5-10^6 candidate teams and pandas row-indexing
    overhead there was the bottleneck.
    """

    REQUIRED_COLS = ['employee_id', 'ability_composite', 'relational_score', 'cluster_id', 'divergence_score']

    def __init__(self, pool: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLS if c not in pool.columns]
        if missing:
            raise ValueError(f"Pool missing required columns: {missing}")

        pool = pool.dropna(subset=self.REQUIRED_COLS).reset_index(drop=True)
        if pool.empty:
            raise ValueError("Pool is empty after dropping rows with missing required fields")

        self.pool = pool
        self.n = len(pool)
        self.max_ability = float(pool['ability_composite'].max())

        # raw numpy views for the hot loop — indexed by row position, not employee_id
        self._employee_id = pool['employee_id'].to_numpy()
        self._ability = pool['ability_composite'].to_numpy(dtype=float)
        self._relational = pool['relational_score'].to_numpy(dtype=float)
        self._cluster = pool['cluster_id'].to_numpy()
        self._divergence = pool['divergence_score'].to_numpy(dtype=float)
        self._n_clusters_total = pool['cluster_id'].nunique()

    # ---------- Utilities operating on integer index arrays (numpy, no pandas) ----------
    def _utility_a(self, idx: np.ndarray) -> float:
        return float(self._ability[idx].mean())

    def _utility_b(self, idx: np.ndarray, lam: float = 0.5) -> float:
        clusters = self._cluster[idx]
        _, counts = np.unique(clusters, return_counts=True)
        proportions = counts / counts.sum()
        blau = 1 - np.sum(proportions ** 2)

        k = len(idx)
        if k < 2:
            return float(blau)
        ability = self._ability[idx]
        relational = self._relational[idx]
        mismatches, total_pairs = 0, 0
        for i in range(k):
            for j in range(i + 1, k):
                total_pairs += 1
                if ability[i] != ability[j] and relational[i] != relational[j]:
                    if np.sign(ability[i] - ability[j]) != np.sign(relational[i] - relational[j]):
                        mismatches += 1
        anti_alignment_rate = mismatches / total_pairs if total_pairs else 0.0
        return float(blau - lam * anti_alignment_rate)

    def _utility_c(self, idx: np.ndarray) -> float:
        return float(-self._divergence[idx].mean())

    def _team_utility(self, idx: np.ndarray, weights: Dict[str, float]) -> float:
        return (weights['A'] * self._utility_a(idx)
                + weights['B'] * self._utility_b(idx)
                + weights['C'] * self._utility_c(idx))

    # ---------- Arbitration ----------
    def _normalized_beta(self, project_complexity: float, beta_min: float, beta_max: float) -> float:
        if beta_max == beta_min:
            return 0.5
        return float(np.clip((project_complexity - beta_min) / (beta_max - beta_min), 0, 1))

    def arbitration_weights(self, beta_norm: float) -> Dict[str, float]:
        return {'A': 1 - beta_norm, 'B': beta_norm * 0.6, 'C': beta_norm * 0.4}

    # ---------- Explainability (only computed ONCE on the final team, not in the hot loop) ----------
    def _rationale_a(self, idx: np.ndarray) -> str:
        mean_ab = self._ability[idx].mean()
        percentile = (self._ability < mean_ab).mean()
        return (f"Team mean ability_composite = {mean_ab:.3f} "
                f"(~{percentile*100:.0f}th percentile of available pool of {self.n})")

    def _rationale_b(self, idx: np.ndarray) -> str:
        clusters = self._cluster[idx]
        n_spanned = len(np.unique(clusters))
        ability, relational = self._ability[idx], self._relational[idx]
        k = len(idx)
        mismatches, total_pairs = 0, 0
        for i in range(k):
            for j in range(i + 1, k):
                total_pairs += 1
                if ability[i] != ability[j] and relational[i] != relational[j]:
                    if np.sign(ability[i] - ability[j]) != np.sign(relational[i] - relational[j]):
                        mismatches += 1
        return (f"Spans {n_spanned} of {self._n_clusters_total} sub-cultures; "
                f"{mismatches}/{total_pairs} pairs are Proposition-2 anti-aligned")

    def _rationale_c(self, idx: np.ndarray) -> str:
        mean_div = self._divergence[idx].mean()
        pool_mean = self._divergence.mean()
        comparison = "below" if mean_div < pool_mean else "above"
        level = "low" if mean_div < 0.3 else "moderate" if mean_div < 0.5 else "high"
        return (f"Team mean divergence_score = {mean_div:.3f} "
                f"({comparison} pool average of {pool_mean:.3f}) — {level} evaluator disagreement risk")

    def _team_reasoning(self, idx: np.ndarray, weights: Dict[str, float]) -> Dict:
        return {
            'A_performance': {'utility': round(self._utility_a(idx), 4), 'weight': round(weights['A'], 3),
                              'rationale': self._rationale_a(idx)},
            'B_diversity':   {'utility': round(self._utility_b(idx), 4), 'weight': round(weights['B'], 3),
                              'rationale': self._rationale_b(idx)},
            'C_conflict':    {'utility': round(self._utility_c(idx), 4), 'weight': round(weights['C'], 3),
                              'rationale': self._rationale_c(idx)}
        }

    def _team_records(self, idx: np.ndarray) -> List[Dict]:
        return [{
            'employee_id': self._employee_id[i],
            'cluster_id': int(self._cluster[i]),
            'ability_composite': round(float(self._ability[i]), 4),
            'relational_score': round(float(self._relational[i]), 4),
            'divergence_score': round(float(self._divergence[i]), 4),
        } for i in idx]

    def _growth_metrics(self, idx: np.ndarray) -> Dict:
        headroom = (self.max_ability - self._ability[idx]) / self.max_ability
        return {
            'growth_headroom': {self._employee_id[i]: round(float(h), 3) for i, h in zip(idx, headroom)},
            'surplus_capacity': round(float(headroom.mean()), 3)
        }

    # ---------- Optimization: greedy local search, numpy hot path ----------
    def optimize_team(self, team_size: int, project_complexity: float,
                      beta_min: float, beta_max: float,
                      n_restarts: int = 20, max_swap_iters: int = 30) -> Dict:

        if team_size > self.n:
            raise ValueError(f"team_size ({team_size}) exceeds available pool ({self.n})")

        beta_norm = self._normalized_beta(project_complexity, beta_min, beta_max)
        weights = self.arbitration_weights(beta_norm)

        rng = np.random.default_rng(42)
        all_idx = np.arange(self.n)

        seen_teams = []
        best_idx, best_score = None, -np.inf

        for _ in range(n_restarts):
            candidate = rng.choice(all_idx, size=team_size, replace=False)
            score = self._team_utility(candidate, weights)

            for _ in range(max_swap_iters):
                improved = False
                outside = np.setdiff1d(all_idx, candidate, assume_unique=False)
                for pos in range(len(candidate)):
                    out_member = candidate[pos]
                    for in_member in outside:
                        trial = candidate.copy()
                        trial[pos] = in_member
                        trial_score = self._team_utility(trial, weights)
                        if trial_score > score:
                            candidate, score = trial, trial_score
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break

            seen_teams.append((frozenset(candidate.tolist()), score))
            if score > best_score:
                best_score, best_idx = score, candidate.copy()

        # rejected alternative: best distinct runner-up
        runner_up = None
        for idx_set, score in sorted(seen_teams, key=lambda x: -x[1]):
            if idx_set != frozenset(best_idx.tolist()) and score < best_score:
                ru_idx = np.array(list(idx_set))
                runner_up = {
                    'team': self._team_records(ru_idx),
                    'team_utility': round(float(score), 4),
                    'utility_gap_vs_best': round(float(best_score - score), 4)
                }
                break

        growth = self._growth_metrics(best_idx)

        return {
            'status': 'success',
            'team_size': team_size,
            'project_complexity_raw': project_complexity,
            'project_complexity_normalized': round(beta_norm, 3),
            'arbitration_weights': {k: round(v, 3) for k, v in weights.items()},
            'team': self._team_records(best_idx),
            'team_utility': round(float(best_score), 4),
            'diplomat_reasoning': self._team_reasoning(best_idx, weights),
            'growth_headroom': growth['growth_headroom'],
            'surplus_capacity': growth['surplus_capacity'],
            'rejected_alternative': runner_up
        }