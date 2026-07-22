import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency, kruskal
from sqlalchemy.orm import Session
from typing import Dict, Optional, List
import logging

from backend.models.employee_model import EmployeePerformanceView
from backend.models.feedback_model import AggregatedPerformance
from backend.models.prediction_model import PerformancePrediction

logger = logging.getLogger(__name__)


class CultureClusterService:
    """
    Step 2: Sub-culture identification.
    Clustering features = demographics ONLY. KPI + prediction + opinion-dynamics
    scores are attached AFTER clustering as an outcome/profiling layer.
    Scope: restricted to the 432 employees with Feedback_360 data (relational_score
    and divergence_score don't exist for the other 568 — documented limitation).
    """

    DEMOGRAPHIC_NUMERIC = ['years_of_experience']
    DEMOGRAPHIC_CATEGORICAL = ['gender', 'age_group', 'department', 'ethnicity', 'language_proficiency']
    ABILITY_WEIGHTS = {'task_score_norm': 0.35, 'kpi_score': 0.35, 'predicted_score_norm': 0.30}

    def __init__(self, db: Session, n_clusters: int = 4):
        self.db = db
        self.n_clusters = n_clusters

    # ---------- Clustering (demographics only) ----------
    def _build_feature_frame(self, institution_id: Optional[str] = None) -> pd.DataFrame:
        employees = (
            self.db.query(EmployeePerformanceView)
            .filter(EmployeePerformanceView.employee_id.isnot(None))
            .all()
        )
        emp_df = pd.DataFrame([{
            'employee_id': e.employee_id,
            **{c: getattr(e, c, None) for c in self.DEMOGRAPHIC_NUMERIC + self.DEMOGRAPHIC_CATEGORICAL}
        } for e in employees]).drop_duplicates(subset='employee_id')

        perf_q = self.db.query(AggregatedPerformance)
        if institution_id:
            perf_q = perf_q.filter(AggregatedPerformance.institution_id == institution_id)
        perf_df = pd.DataFrame([{
            'employee_id': r.employee_id,
            'relational_score': r.relational_score
        } for r in perf_q.all()])

        if perf_df.empty:
            return perf_df
        perf_df = perf_df.groupby('employee_id', as_index=False).mean(numeric_only=True)

        df = emp_df.merge(perf_df, on='employee_id', how='inner')  # restricts scope to the 432, by design
        return df.dropna(subset=self.DEMOGRAPHIC_NUMERIC + ['relational_score'])

    def fit_clusters(self, institution_id: Optional[str] = None) -> pd.DataFrame:
        df = self._build_feature_frame(institution_id)
        if df.empty or len(df) < self.n_clusters:
            logger.warning("Insufficient employees with complete data to cluster")
            return pd.DataFrame()

        num = df[self.DEMOGRAPHIC_NUMERIC]
        cat = pd.get_dummies(df[self.DEMOGRAPHIC_CATEGORICAL], drop_first=True)
        features = pd.concat([num, cat], axis=1)
        scaled = StandardScaler().fit_transform(features)

        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        df = df.copy()
        df['cluster_id'] = km.fit_predict(scaled)
        return df

    def evaluate_k_range(self, institution_id: Optional[str] = None, k_range=range(2, 8)) -> List[Dict]:
        df = self._build_feature_frame(institution_id)
        num = df[self.DEMOGRAPHIC_NUMERIC]
        cat = pd.get_dummies(df[self.DEMOGRAPHIC_CATEGORICAL], drop_first=True)
        features = pd.concat([num, cat], axis=1)
        scaled = StandardScaler().fit_transform(features)

        results = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
            sil = silhouette_score(scaled, km.labels_)
            results.append({'k': k, 'inertia': float(km.inertia_), 'silhouette': float(sil)})
        return results

    # ---------- Ability composite (post-clustering) ----------
    def _build_ability_composite(self, employee_ids: List[str]) -> pd.DataFrame:
        agg_rows = self.db.query(AggregatedPerformance).filter(
            AggregatedPerformance.employee_id.in_(employee_ids)
        ).all()
        task_records = []
        for r in agg_rows:
            vals = []
            for prefix in ['self', 'manager', 'peer', 'sub']:
                for comp in ['punctuality', 'problem_solving', 'leadership']:
                    v = getattr(r, f'{prefix}_{comp}', None)
                    if v is not None:
                        vals.append(v)
            task_records.append({'employee_id': r.employee_id, 'task_score': np.mean(vals) if vals else None})
        task_df = pd.DataFrame(task_records).groupby('employee_id', as_index=False).mean(numeric_only=True)

        kpi_rows = self.db.query(EmployeePerformanceView).filter(
            EmployeePerformanceView.employee_id.in_(employee_ids)
        ).all()
        kpi_df = pd.DataFrame([{
            'employee_id': r.employee_id,
            'deadline_adherence_rate': r.deadline_adherence_rate,
            'completed_storypoint_ratio': r.completed_storypoint_ratio
        } for r in kpi_rows]).groupby('employee_id', as_index=False).mean(numeric_only=True)
        kpi_df['kpi_score'] = kpi_df.apply(
            lambda row: np.nanmean([
                row['deadline_adherence_rate'] / 100.0 if pd.notna(row['deadline_adherence_rate']) else np.nan,
                row['completed_storypoint_ratio'] if pd.notna(row['completed_storypoint_ratio']) else np.nan
            ]), axis=1
        )

        pred_rows = self.db.query(PerformancePrediction).filter(
            PerformancePrediction.employee_id.in_(employee_ids)
        ).order_by(PerformancePrediction.employee_id, PerformancePrediction.predicted_at.desc()).all()
        seen, pred_records = set(), []
        for p in pred_rows:
            if p.employee_id not in seen:
                pred_records.append({'employee_id': p.employee_id, 'predicted_score': p.predicted_score})
                seen.add(p.employee_id)
        pred_df = pd.DataFrame(pred_records) if pred_records else pd.DataFrame(columns=['employee_id', 'predicted_score'])
        if not pred_df.empty:
            pred_df['predicted_score_norm'] = pred_df['predicted_score'] / 100.0
        else:
            pred_df['predicted_score_norm'] = pd.Series(dtype=float)

        composite = task_df.merge(kpi_df[['employee_id', 'kpi_score']], on='employee_id', how='outer')
        composite = composite.merge(pred_df[['employee_id', 'predicted_score_norm']], on='employee_id', how='outer')
        composite['task_score_norm'] = (composite['task_score'] - 1) / 4.0

        def blend(row):
            parts, weights = [], []
            for col, w in self.ABILITY_WEIGHTS.items():
                if pd.notna(row.get(col)):
                    parts.append(row[col])
                    weights.append(w)
            if not parts:
                return np.nan
            weights = np.array(weights) / np.sum(weights)
            return float(np.dot(parts, weights))

        composite['ability_composite'] = composite.apply(blend, axis=1)
        return composite[['employee_id', 'task_score', 'kpi_score', 'predicted_score_norm', 'ability_composite']]

    # ---------- Alignment (Prop 2) ----------
    def compute_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ability_rank'] = df['ability_composite'].rank(pct=True)
        df['relational_rank'] = df['relational_score'].rank(pct=True)
        df['aligned'] = (df['ability_rank'] > 0.5) == (df['relational_rank'] > 0.5)
        return df

    def cluster_alignment_crosstab(self, df: pd.DataFrame) -> List[Dict]:
        summary = df.groupby('cluster_id').agg(
            n=('employee_id', 'count'),
            pct_aligned=('aligned', 'mean'),
            mean_ability=('ability_composite', 'mean'),
            mean_relational=('relational_score', 'mean')
        ).reset_index()
        return summary.to_dict('records')

    def alignment_significance_test(self, df: pd.DataFrame) -> Dict:
        contingency = pd.crosstab(df['cluster_id'], df['aligned'])
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        return {'chi2': float(chi2), 'p_value': float(p_value), 'dof': int(dof), 'significant': bool(p_value < 0.05)}

    # ---------- Opinion dynamics link (Step 1 <-> Step 2) ----------
    def _attach_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        divergence_rows = self.db.query(AggregatedPerformance).filter(
            AggregatedPerformance.employee_id.in_(df['employee_id'].tolist())
        ).all()
        div_df = pd.DataFrame([{
            'employee_id': r.employee_id,
            'divergence_score': r.divergence_score
        } for r in divergence_rows]).groupby('employee_id', as_index=False).mean(numeric_only=True)
        return df.merge(div_df, on='employee_id', how='left')

    def cluster_opinion_dynamics_profile(self, merged_df: pd.DataFrame) -> List[Dict]:
        profile = []
        for cluster_id, group in merged_df.groupby('cluster_id'):
            profile.append({
                'cluster_id': int(cluster_id),
                'n': int(len(group)),
                'mean_divergence': round(float(group['divergence_score'].mean()), 3),
                'median_divergence': round(float(group['divergence_score'].median()), 3),
                'std_divergence': round(float(group['divergence_score'].std(ddof=0) or 0), 3),
                'pct_high_divergence': round(float((group['divergence_score'] > 0.5).mean()), 2)
            })
        return profile

    def divergence_by_cluster_significance(self, merged_df: pd.DataFrame) -> Dict:
        groups = [g['divergence_score'].dropna().values for _, g in merged_df.groupby('cluster_id')]
        groups = [g for g in groups if len(g) > 0]
        stat, p_value = kruskal(*groups)
        return {'h_statistic': float(stat), 'p_value': float(p_value), 'significant': bool(p_value < 0.05)}

    # ---------- Profiling ----------
    def cluster_demographic_profile(self, df: pd.DataFrame) -> List[Dict]:
        profile = []
        for cluster_id, group in df.groupby('cluster_id'):
            profile.append({
                'cluster_id': int(cluster_id),
                'size': int(len(group)),
                'gender': group['gender'].value_counts().to_dict(),
                'age_group': group['age_group'].value_counts().to_dict(),
                'department': group['department'].value_counts().to_dict(),
                'ethnicity': group['ethnicity'].value_counts().to_dict(),
                'language': group['language_proficiency'].value_counts().to_dict(),
                'mean_years_experience': round(float(group['years_of_experience'].mean()), 1)
            })
        return profile

    # ---------- Orchestration ----------
    def run(self, institution_id: Optional[str] = None) -> Dict:
        df = self.fit_clusters(institution_id)
        if df.empty:
            return {'status': 'error', 'message': 'Insufficient data to cluster'}

        ability_df = self._build_ability_composite(df['employee_id'].tolist())
        df = df.merge(ability_df, on='employee_id', how='left')
        df = self.compute_alignment(df)
        df = self._attach_divergence(df)

        return {
            'status': 'success',
            'n_employees': int(len(df)),
            'n_clusters': self.n_clusters,
            'demographic_profile': self.cluster_demographic_profile(df),
            'alignment_summary': self.cluster_alignment_crosstab(df),
            'alignment_significance': self.alignment_significance_test(df),
            'opinion_dynamics_profile': self.cluster_opinion_dynamics_profile(df),
            'opinion_dynamics_significance': self.divergence_by_cluster_significance(df),
            'employees': df.to_dict('records')
        }