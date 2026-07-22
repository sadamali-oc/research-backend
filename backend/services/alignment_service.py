# backend/services/alignment_service.py
"""
Alignment Analysis Service - Step 2 of the Dialogue Diplomat Framework
Connects to database and computes alignment classification (Proposition 2)
"""
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, List, Optional
import logging

from backend.database.database import SessionLocal
from backend.models.feedback_model import AggregatedPerformance

logger = logging.getLogger(__name__)

class AlignmentService:
    """Service for Step 2: Sub-culture Identification & Performance Mapping"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_employee_data(self, institution_id: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch employee performance and relational data from database
        """
        query = self.db.query(AggregatedPerformance)

        if institution_id:
            query = query.filter(AggregatedPerformance.institution_id == institution_id)

        results = query.all()

        # Convert to DataFrame
        data = []
        for r in results:
            if r.performance_score is not None and r.relational_score is not None:
                data.append({
                    'employee_id': r.employee_id,
                    'institution_id': r.institution_id,
                    'period_year': r.period_year,
                    'period_quarter': r.period_quarter,
                    'performance_score': r.performance_score,
                    'relational_score': r.relational_score,
                    'divergence_score': r.divergence_score,
                    'self_count': r.self_count,
                    'manager_count': r.manager_count,
                    'peer_count': r.peer_count,
                    'sub_count': r.sub_count,
                    'self_overall': r.self_overall,
                    'manager_overall': r.manager_overall,
                    'peer_overall': r.peer_overall,
                    'sub_overall': r.sub_overall,
                    'pdi_institution': r.pdi_institution
                })

        df = pd.DataFrame(data)
        logger.info(f"Fetched {len(df)} employee records")
        return df

    def compute_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute alignment classification based on Proposition 2:
        - Aligned: High performance AND high relational (or both low)
        - Anti-Aligned: High performance BUT low relational (or vice versa)
        """
        if df.empty:
            logger.warning("No data to classify")
            return df

        # Compute means
        perf_mean = df['performance_score'].mean()
        rel_mean = df['relational_score'].mean()

        logger.info(f"Mean Performance: {perf_mean:.3f}")
        logger.info(f"Mean Relational: {rel_mean:.3f}")

        # Classify
        df['is_high_perf'] = df['performance_score'] > perf_mean
        df['is_high_rel'] = df['relational_score'] > rel_mean

        df['alignment'] = np.where(
            df['is_high_perf'] == df['is_high_rel'],
            'Aligned',
            'Anti-Aligned'
        )

        # Add alignment score (0 = anti-aligned, 1 = aligned)
        df['alignment_score'] = (df['is_high_perf'] == df['is_high_rel']).astype(int)

        return df

    def compute_alignment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Compute summary statistics for alignment analysis
        """
        if df.empty:
            return {}

        # Overall summary
        summary = {
            'total_employees': len(df),
            'aligned_count': int(df[df['alignment'] == 'Aligned'].shape[0]),
            'anti_aligned_count': int(df[df['alignment'] == 'Anti-Aligned'].shape[0]),
            'mean_performance': float(df['performance_score'].mean()),
            'mean_relational': float(df['relational_score'].mean()),
            'alignment_distribution': df['alignment'].value_counts().to_dict(),
            'institutions': df['institution_id'].unique().tolist()
        }

        # Performance by alignment
        perf_by_alignment = df.groupby('alignment')['performance_score'].agg(['mean', 'count']).to_dict()
        summary['perf_by_alignment'] = {
            'Aligned': {
                'mean': float(perf_by_alignment['mean'].get('Aligned', 0)),
                'count': int(perf_by_alignment['count'].get('Aligned', 0))
            },
            'Anti-Aligned': {
                'mean': float(perf_by_alignment['mean'].get('Anti-Aligned', 0)),
                'count': int(perf_by_alignment['count'].get('Anti-Aligned', 0))
            }
        }

        # Performance gain calculation (Proposition 2)
        aligned_perf = summary['perf_by_alignment']['Aligned']['mean']
        anti_aligned_perf = summary['perf_by_alignment']['Anti-Aligned']['mean']
        summary['performance_gain'] = aligned_perf - anti_aligned_perf
        summary['performance_gain_percentage'] = (aligned_perf / anti_aligned_perf - 1) * 100 if anti_aligned_perf > 0 else 0

        return summary

    def get_high_divergence_employees(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Get employees with high divergence scores for case study analysis
        """
        high_div = df[df['divergence_score'] > threshold].copy()
        high_div = high_div.sort_values('divergence_score', ascending=False)
        return high_div

    def get_case_study_employees(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Get four case study employees for qualitative analysis:
        1. High divergence, high performance (Aligned)
        2. High divergence, low performance (Anti-Aligned)
        3. Low divergence, high performance (Aligned)
        4. Low divergence, low performance (Aligned)
        """
        case_studies = {}

        # 1. High divergence, high performance (Aligned)
        case1 = df[(df['divergence_score'] > 0.5) & (df['performance_score'] > df['performance_score'].mean())]
        if not case1.empty:
            case_studies['high_div_high_perf'] = case1.head(3)

        # 2. High divergence, low performance (Anti-Aligned)
        case2 = df[(df['divergence_score'] > 0.5) & (df['performance_score'] < df['performance_score'].mean())]
        if not case2.empty:
            case_studies['high_div_low_perf'] = case2.head(3)

        # 3. Low divergence, high performance (Aligned)
        case3 = df[(df['divergence_score'] < 0.15) & (df['performance_score'] > df['performance_score'].mean())]
        if not case3.empty:
            case_studies['low_div_high_perf'] = case3.head(3)

        # 4. Low divergence, low performance (Aligned)
        case4 = df[(df['divergence_score'] < 0.15) & (df['performance_score'] < df['performance_score'].mean())]
        if not case4.empty:
            case_studies['low_div_low_perf'] = case4.head(3)

        return case_studies

    def run_full_analysis(self, institution_id: Optional[str] = None) -> Dict:
        """
        Run the complete Step 2 analysis
        """
        logger.info(f"Running Step 2 analysis for institution: {institution_id or 'All'}")

        # Fetch data
        df = self.get_employee_data(institution_id)

        if df.empty:
            return {'status': 'error', 'message': 'No data found'}

        # Compute alignment
        df = self.compute_alignment(df)

        # Compute summary
        summary = self.compute_alignment_summary(df)

        # Get case studies
        case_studies = self.get_case_study_employees(df)

        # Get high divergence employees
        high_div = self.get_high_divergence_employees(df)

        return {
            'status': 'success',
            'summary': summary,
            'data': df.to_dict('records'),
            'high_divergence_employees': high_div.to_dict('records'),
            'case_studies': {
                key: case_studies[key].to_dict('records')
                for key in case_studies.keys()
            },
            'means': {
                'performance_mean': float(df['performance_score'].mean()),
                'relational_mean': float(df['relational_score'].mean())
            }
        }


# -------- Usage Script --------
# Run this directly to see results

def run_alignment_analysis():
    """
    Main function to run alignment analysis and print results
    """
    try:
        # Create database session
        db = SessionLocal()

        # Initialize service
        service = AlignmentService(db)

        print("=" * 70)
        print("STEP 2: ALIGNMENT ANALYSIS (Proposition 2)")
        print("=" * 70)

        # Run analysis for all institutions
        result = service.run_full_analysis()

        if result['status'] == 'error':
            print(f"Error: {result['message']}")
            return

        # Print Summary
        summary = result['summary']
        print("\n📊 SUMMARY STATISTICS")
        print("-" * 50)
        print(f"Total Employees: {summary['total_employees']}")
        print(f"Aligned Employees: {summary['aligned_count']} ({summary['aligned_count']/summary['total_employees']*100:.1f}%)")
        print(f"Anti-Aligned Employees: {summary['anti_aligned_count']} ({summary['anti_aligned_count']/summary['total_employees']*100:.1f}%)")
        print(f"\nMean Performance: {summary['mean_performance']:.3f}")
        print(f"Mean Relational Score: {summary['mean_relational']:.3f}")

        print("\n📈 PERFORMANCE BY ALIGNMENT")
        print("-" * 50)
        for alignment, stats in summary['perf_by_alignment'].items():
            print(f"{alignment}: {stats['count']} employees, Avg Performance: {stats['mean']:.3f}")

        print(f"\n🏆 PERFORMANCE GAIN (Proposition 2)")
        print("-" * 50)
        print(f"Aligned teams outperform by: {summary['performance_gain']:.3f} points")
        print(f"Percentage gain: {summary['performance_gain_percentage']:.1f}%")

        # Print Means
        means = result['means']
        print(f"\n📐 MEANS FOR CLASSIFICATION")
        print("-" * 50)
        print(f"Performance Mean: {means['performance_mean']:.3f}")
        print(f"Relational Mean: {means['relational_mean']:.3f}")

        # Print High Divergence Employees
        high_div = result['high_divergence_employees']
        if high_div:
            print(f"\n🔴 HIGH DIVERGENCE EMPLOYEES (> 0.5)")
            print("-" * 50)
            print(f"Count: {len(high_div)}")
            df_high = pd.DataFrame(high_div)
            print(df_high[['employee_id', 'performance_score', 'relational_score', 'divergence_score', 'alignment']].to_string(index=False))

        # Print Case Studies
        print(f"\n📋 CASE STUDIES")
        print("-" * 50)
        for case_name, case_data in result['case_studies'].items():
            if case_data:
                print(f"\n{case_name.replace('_', ' ').upper()}:")
                df_case = pd.DataFrame(case_data)
                print(df_case[['employee_id', 'performance_score', 'relational_score', 'divergence_score', 'alignment']].to_string(index=False))
            else:
                print(f"\n{case_name.replace('_', ' ').upper()}: No data found")

        # Save results to CSV
        df_all = pd.DataFrame(result['data'])
        df_all.to_csv('alignment_analysis_results.csv', index=False)
        print(f"\n✅ Full results saved to alignment_analysis_results.csv")

        return result

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    run_alignment_analysis()