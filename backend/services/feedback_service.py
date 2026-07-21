# backend/services/feedback_service.py
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, text
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from backend.models.feedback_model import Feedback360, AggregatedPerformance
from backend.schemas.feedback_schema import AggregatedPerformanceBase

logger = logging.getLogger(__name__)

class FeedbackService:
    """Service for 360-degree feedback data processing with hierarchy distance awareness"""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.competency_cols = ['punctuality', 'problem_solving', 'leadership', 'collaboration', 'communication']

        # Role to level mapping for hierarchy distance
        self.ROLE_LEVELS = {
            'PM': 4,      # Project Manager (highest)
            'BA': 3,      # Business Analyst
            'SE': 3,      # Software Engineer
            'DevOps': 2,  # DevOps Engineer
            'QA': 2,      # Quality Assurance
        }

        # Competency weights for divergence calculation
        self.COMPETENCY_WEIGHTS = {
            'punctuality': 0.15,
            'problem_solving': 0.25,
            'leadership': 0.25,
            'collaboration': 0.20,
            'communication': 0.15
        }

        # Divergence thresholds
        self.DIVERGENCE_THRESHOLDS = {
            'very_low': 0.15,
            'low': 0.30,
            'moderate': 0.50,
            'high': 0.75,
            'very_high': 1.0
        }

    def _convert_bytes_to_int(self, value):
        """Helper to convert bytes to int if needed"""
        if isinstance(value, bytes):
            try:
                return int.from_bytes(value, 'little')
            except:
                return int(value) if value else 0
        return value

    def _safe_int(self, value):
        """Safely convert any value to int"""
        if value is None:
            return 0
        if isinstance(value, bytes):
            try:
                return int.from_bytes(value, 'little')
            except:
                return 0
        try:
            return int(value)
        except:
            return 0

    def _safe_float(self, value):
        """Safely convert any value to float"""
        if value is None:
            return 0.0
        if isinstance(value, bytes):
            try:
                return float(int.from_bytes(value, 'little'))
            except:
                return 0.0
        try:
            return float(value)
        except:
            return 0.0

    # -------- Data Ingestion --------
    def import_csv_data(self, csv_path: str, institution_id: str = None) -> Dict:
        """
        Import feedback data from CSV file into database
        Returns stats about the import
        """
        logger.info(f"Importing CSV data from {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Read {len(df)} records from CSV")

        # Optional filter by institution
        if institution_id:
            df = df[df['institution_id'] == institution_id]
            logger.info(f"Filtered to {len(df)} records for institution {institution_id}")

        # Convert to list of dicts for bulk insert
        records = df.to_dict('records')

        # Insert in batches
        batch_size = 500
        total_inserted = 0

        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            # Use SQLAlchemy bulk insert
            self.db.bulk_insert_mappings(Feedback360, batch)
            self.db.commit()
            total_inserted += len(batch)
            logger.info(f"Inserted batch {i//batch_size + 1} ({len(batch)} records)")

        return {
            "total_records": len(df),
            "inserted": total_inserted,
            "institutions": df['institution_id'].unique().tolist(),
            "periods": df.groupby(['period_year', 'period_quarter']).size().to_dict()
        }

    # -------- Hierarchy Distance Calculation --------
    def calculate_hierarchy_distance(self, evaluator_role: str, evaluatee_role: str) -> int:
        """
        Calculate hierarchy distance between evaluator and evaluatee
        hierarchy_distance = evaluator_level - evaluatee_level
        Positive: evaluator is higher level (manager)
        Zero: same level (peer)
        Negative: evaluator is lower level (subordinate)
        """
        evaluator_level = self.ROLE_LEVELS.get(evaluator_role, 2)
        evaluatee_level = self.ROLE_LEVELS.get(evaluatee_role, 2)
        return evaluator_level - evaluatee_level

    def update_hierarchy_distances(self, institution_id: Optional[str] = None) -> Dict:
        """
        Update hierarchy_distance for all feedback records based on role mapping
        """
        logger.info("Updating hierarchy distances...")

        # Get all feedback records
        query = self.db.query(Feedback360)
        if institution_id:
            query = query.filter(Feedback360.institution_id == institution_id)

        records = query.all()
        updated_count = 0

        for record in records:
            # Extract role from evaluator_id (assuming format like E001, PM001, etc.)
            # This is a simplified example - you'd need actual role data
            # In practice, you'd join with employee table to get roles
            evaluator_role = self._extract_role(record.evaluator_id)
            evaluatee_role = self._extract_role(record.evaluatee_id)

            hierarchy_distance = self.calculate_hierarchy_distance(evaluator_role, evaluatee_role)

            if record.hierarchy_distance != hierarchy_distance:
                record.hierarchy_distance = hierarchy_distance
                updated_count += 1

        self.db.commit()
        logger.info(f"Updated hierarchy distances for {updated_count} records")

        return {
            "total_records": len(records),
            "updated": updated_count
        }

    def _extract_role(self, employee_id: str) -> str:
        """
        Extract role from employee ID
        This is a simplified example - in reality, you'd query the employee table
        """
        # Example: E001 -> SE, PM001 -> PM, etc.
        if employee_id and len(employee_id) > 0:
            # This is just a placeholder - you'd implement actual role lookup
            return 'SE'  # Default role
        return 'SE'

    # -------- Data Aggregation (Preprocessing) --------
    def aggregate_performance(self, institution_id: Optional[str] = None,
                              period_year: Optional[int] = None,
                              period_quarter: Optional[int] = None) -> Dict:
        """
        Aggregate 360-degree feedback into per-employee, per-quarter metrics
        This is the core preprocessing step for your research
        """
        logger.info("Starting performance aggregation...")

        # Build query filter
        filters = []
        if institution_id:
            filters.append(Feedback360.institution_id == institution_id)
        if period_year:
            filters.append(Feedback360.period_year == period_year)
        if period_quarter:
            filters.append(Feedback360.period_quarter == period_quarter)

        # Get all feedback records
        query = self.db.query(Feedback360).filter(*filters) if filters else self.db.query(Feedback360)
        records = query.all()
        logger.info(f"Processing {len(records)} feedback records")

        if not records:
            return {"status": "error", "message": "No records found matching filters"}

        # Convert to DataFrame
        df = pd.DataFrame([{
            'evaluatee_id': r.evaluatee_id,
            'evaluator_id': r.evaluator_id,
            'evaluation_type': r.evaluation_type,
            'institution_id': r.institution_id,
            'period_year': self._safe_int(r.period_year),
            'period_quarter': self._safe_int(r.period_quarter),
            'hierarchy_distance': r.hierarchy_distance,
            'punctuality': self._safe_float(r.punctuality),
            'problem_solving': self._safe_float(r.problem_solving),
            'leadership': self._safe_float(r.leadership),
            'collaboration': self._safe_float(r.collaboration),
            'communication': self._safe_float(r.communication),
            'overall_rating': self._safe_float(r.overall_rating)
        } for r in records])

        # Group by employee, period, and evaluation type
        aggregated = []

        for (employee_id, year, quarter, institution), group in df.groupby(['evaluatee_id', 'period_year', 'period_quarter', 'institution_id']):
            # Separate by evaluation type using hierarchy_distance
            self_df = group[group['evaluation_type'] == 'Self']
            manager_df = group[group['hierarchy_distance'] > 0]  # Positive = manager
            peer_df = group[group['hierarchy_distance'] == 0]     # Zero = peer
            sub_df = group[group['hierarchy_distance'] < 0]      # Negative = subordinate

            # Calculate averages for each competency
            def get_avg(df, col):
                return df[col].mean() if not df.empty else None

            def get_count(df):
                return len(df)

            # Build aggregated record
            agg_record = {
                'employee_id': employee_id,
                'institution_id': institution,
                'period_year': year,
                'period_quarter': quarter,
                'self_count': get_count(self_df),
                'manager_count': get_count(manager_df),
                'peer_count': get_count(peer_df),
                'sub_count': get_count(sub_df),
            }

            # Self scores
            for col in self.competency_cols + ['overall_rating']:
                agg_record[f'self_{col}'] = get_avg(self_df, col)

            # Manager scores
            for col in self.competency_cols + ['overall_rating']:
                agg_record[f'manager_{col}'] = get_avg(manager_df, col)

            # Peer scores
            for col in self.competency_cols + ['overall_rating']:
                agg_record[f'peer_{col}'] = get_avg(peer_df, col)

            # Subordinate scores
            for col in self.competency_cols + ['overall_rating']:
                agg_record[f'sub_{col}'] = get_avg(sub_df, col)

            # Calculate overall performance score (all sources)
            all_scores = []
            for col in self.competency_cols + ['overall_rating']:
                for prefix in ['self', 'manager', 'peer', 'sub']:
                    val = agg_record.get(f'{prefix}_{col}')
                    if val is not None:
                        all_scores.append(val)

            agg_record['performance_score'] = np.mean(all_scores) if all_scores else None

            # Calculate relational score (average of collaboration + communication)
            rel_scores = []
            for prefix in ['self', 'manager', 'peer', 'sub']:
                collab = agg_record.get(f'{prefix}_collaboration')
                comm = agg_record.get(f'{prefix}_communication')
                if collab is not None and comm is not None:
                    rel_scores.append((collab + comm) / 2)

            agg_record['relational_score'] = np.mean(rel_scores) if rel_scores else None

            # ========== DIVERGENCE CALCULATION WITH HIERARCHY ==========
            # Collect ALL individual overall_rating values by evaluation type
            all_ratings = []

            # Self ratings
            for _, row in self_df.iterrows():
                if pd.notna(row['overall_rating']):
                    all_ratings.append({
                        'type': 'Self',
                        'rating': row['overall_rating'],
                        'hierarchy_distance': 0
                    })

            # Manager ratings
            for _, row in manager_df.iterrows():
                if pd.notna(row['overall_rating']):
                    all_ratings.append({
                        'type': 'Manager',
                        'rating': row['overall_rating'],
                        'hierarchy_distance': row['hierarchy_distance']
                    })

            # Peer ratings
            for _, row in peer_df.iterrows():
                if pd.notna(row['overall_rating']):
                    all_ratings.append({
                        'type': 'Peer',
                        'rating': row['overall_rating'],
                        'hierarchy_distance': 0
                    })

            # Subordinate ratings
            for _, row in sub_df.iterrows():
                if pd.notna(row['overall_rating']):
                    all_ratings.append({
                        'type': 'Subordinate',
                        'rating': row['overall_rating'],
                        'hierarchy_distance': row['hierarchy_distance']
                    })

            # Calculate divergence as weighted standard deviation
            # with hierarchy distance weighting
            if len(all_ratings) > 1:
                # Weight by hierarchy distance - larger distance = more weight
                weights = []
                ratings_list = []

                for r in all_ratings:
                    ratings_list.append(r['rating'])
                    # Weight: higher hierarchy distance gets more weight
                    # This captures power distance effect
                    weight = 1 + abs(r['hierarchy_distance']) * 0.2
                    weights.append(weight)

                # Normalize weights
                weights = np.array(weights) / np.sum(weights)

                # Calculate weighted standard deviation
                weighted_mean = np.average(ratings_list, weights=weights)
                weighted_variance = np.average(
                    [(r - weighted_mean)**2 for r in ratings_list],
                    weights=weights
                )
                weighted_std = np.sqrt(weighted_variance)

                # Normalize to 0-1 scale (max expected std is ~1.5 for 1-3 scale)
                max_std = 1.5
                agg_record['divergence_score'] = min(weighted_std / max_std, 1.0)

                # Add hierarchy distance penalty for extreme distances
                hierarchy_penalty = 0
                for r in all_ratings:
                    if abs(r['hierarchy_distance']) > 1:
                        hierarchy_penalty += 0.05 * abs(r['hierarchy_distance'])

                agg_record['divergence_score'] = min(
                    agg_record['divergence_score'] + hierarchy_penalty, 1.0
                )
            else:
                agg_record['divergence_score'] = 0

            # Calculate manager vs non-manager divergence (for PDI proxy)
            manager_ratings = [r['rating'] for r in all_ratings if r['type'] == 'Manager']
            non_manager_ratings = [r['rating'] for r in all_ratings if r['type'] != 'Manager']

            if len(manager_ratings) > 0 and len(non_manager_ratings) > 0:
                manager_mean = np.mean(manager_ratings)
                non_manager_mean = np.mean(non_manager_ratings)
                agg_record['manager_divergence'] = abs(manager_mean - non_manager_mean)
            else:
                agg_record['manager_divergence'] = 0

            # Calculate hierarchy distance correlation
            agg_record['hierarchy_correlation'] = self._calculate_hierarchy_correlation(all_ratings)

            aggregated.append(agg_record)

        logger.info(f"Aggregated {len(aggregated)} employee-period records")

        # Save to database
        self._clear_aggregated_data(institution_id, period_year, period_quarter)

        # Bulk insert
        batch_size = 100
        total_inserted = 0

        for i in range(0, len(aggregated), batch_size):
            batch = aggregated[i:i+batch_size]
            self.db.bulk_insert_mappings(AggregatedPerformance, batch)
            self.db.commit()
            total_inserted += len(batch)

        logger.info(f"Saved {total_inserted} aggregated records")

        # Calculate PDI proxies
        self._calculate_pdi_proxies(aggregated)

        return {
            "status": "success",
            "message": "Preprocessing completed successfully",
            "records_processed": len(records),
            "employees_processed": len(df['evaluatee_id'].unique()),
            "quarters_processed": len(df.groupby(['period_year', 'period_quarter'])),
            "institutions_processed": len(df['institution_id'].unique()),
            "aggregated_records": total_inserted
        }

    def _calculate_hierarchy_correlation(self, ratings: List[Dict]) -> float:
        """
        Calculate correlation between hierarchy distance and ratings
        """
        if len(ratings) < 2:
            return 0.0

        distances = [r['hierarchy_distance'] for r in ratings]
        rating_values = [r['rating'] for r in ratings]

        # Calculate correlation
        corr = np.corrcoef(distances, rating_values)[0, 1] if len(distances) > 1 else 0

        # Return absolute value (magnitude of effect)
        return abs(corr) if not np.isnan(corr) else 0.0

    def _clear_aggregated_data(self, institution_id: Optional[str] = None,
                               period_year: Optional[int] = None,
                               period_quarter: Optional[int] = None):
        """Clear existing aggregated data for specific periods"""
        query = self.db.query(AggregatedPerformance)
        if institution_id:
            query = query.filter(AggregatedPerformance.institution_id == institution_id)
        if period_year:
            query = query.filter(AggregatedPerformance.period_year == period_year)
        if period_quarter:
            query = query.filter(AggregatedPerformance.period_quarter == period_quarter)
        deleted = query.delete()
        self.db.commit()
        logger.info(f"Cleared {deleted} existing aggregated records")

    def _calculate_pdi_proxies(self, aggregated_records: List[Dict]):
        """Calculate PDI proxy for each institution based on divergence scores"""
        # Group by institution
        df = pd.DataFrame(aggregated_records)

        pdi_by_institution = {}
        for institution, group in df.groupby('institution_id'):
            # PDI proxy = median divergence score
            pdi_by_institution[institution] = group['divergence_score'].median()

        # Update records with PDI values
        for record in aggregated_records:
            record['pdi_institution'] = pdi_by_institution.get(record['institution_id'], 0)

        # Update database
        for record in aggregated_records:
            self.db.query(AggregatedPerformance).filter(
                AggregatedPerformance.employee_id == record['employee_id'],
                AggregatedPerformance.period_year == record['period_year'],
                AggregatedPerformance.period_quarter == record['period_quarter']
            ).update({'pdi_institution': record['pdi_institution']})

        self.db.commit()
        logger.info(f"Calculated PDI proxies for {len(pdi_by_institution)} institutions")

    # -------- Query Methods --------
    def get_aggregated_data(self, employee_id: Optional[str] = None,
                            institution_id: Optional[str] = None,
                            period_year: Optional[int] = None,
                            period_quarter: Optional[int] = None) -> List[Dict]:
        """Get aggregated performance data with filters - returns dicts with proper type conversion"""
        query = self.db.query(AggregatedPerformance)
        if employee_id:
            query = query.filter(AggregatedPerformance.employee_id == employee_id)
        if institution_id:
            query = query.filter(AggregatedPerformance.institution_id == institution_id)
        if period_year:
            query = query.filter(AggregatedPerformance.period_year == period_year)
        if period_quarter:
            query = query.filter(AggregatedPerformance.period_quarter == period_quarter)

        results = query.all()

        # Convert to dict with proper type conversion
        output = []
        for r in results:
            output.append({
                'id': r.id,
                'employee_id': r.employee_id,
                'institution_id': r.institution_id,
                'period_year': self._safe_int(r.period_year),
                'period_quarter': self._safe_int(r.period_quarter),
                'performance_score': self._safe_float(r.performance_score),
                'relational_score': self._safe_float(r.relational_score),
                'divergence_score': self._safe_float(r.divergence_score),
                'self_count': self._safe_int(r.self_count),
                'manager_count': self._safe_int(r.manager_count),
                'peer_count': self._safe_int(r.peer_count),
                'sub_count': self._safe_int(r.sub_count),
                'self_overall': self._safe_float(r.self_overall),
                'manager_overall': self._safe_float(r.manager_overall),
                'peer_overall': self._safe_float(r.peer_overall),
                'sub_overall': self._safe_float(r.sub_overall),
                'pdi_institution': self._safe_float(r.pdi_institution),
                'created_at': r.created_at,
                'updated_at': r.updated_at
            })

        return output

    def get_divergence_analysis(self, employee_id: Optional[str] = None,
                                institution_id: Optional[str] = None,
                                period_year: Optional[int] = None,
                                period_quarter: Optional[int] = None) -> List[Dict]:
        """
        Get divergence analysis with hierarchy-aware interpretation
        """
        query = self.db.query(AggregatedPerformance)
        if employee_id:
            query = query.filter(AggregatedPerformance.employee_id == employee_id)
        if institution_id:
            query = query.filter(AggregatedPerformance.institution_id == institution_id)
        if period_year:
            query = query.filter(AggregatedPerformance.period_year == period_year)
        if period_quarter:
            query = query.filter(AggregatedPerformance.period_quarter == period_quarter)

        results = query.all()

        analysis = []
        for r in results:
            # Convert bytes to int if needed
            p_year = self._safe_int(r.period_year)
            p_quarter = self._safe_int(r.period_quarter)

            # Get hierarchy-aware interpretation
            interpretation = self._generate_interpretation(
                self._safe_float(r.divergence_score),
                self._safe_float(r.pdi_institution)
            )

            analysis.append({
                'employee_id': r.employee_id,
                'period_year': p_year,
                'period_quarter': p_quarter,
                'divergence_score': self._safe_float(r.divergence_score),
                'pdi_institution': self._safe_float(r.pdi_institution),
                'interpretation': interpretation,
                'self_count': self._safe_int(r.self_count),
                'manager_count': self._safe_int(r.manager_count),
                'peer_count': self._safe_int(r.peer_count),
                'sub_count': self._safe_int(r.sub_count),
                'self_overall': self._safe_float(r.self_overall),
                'manager_overall': self._safe_float(r.manager_overall),
                'peer_overall': self._safe_float(r.peer_overall),
                'sub_overall': self._safe_float(r.sub_overall)
            })

        return analysis

    def _generate_interpretation(self, divergence_score: float, pdi: float) -> str:
        """
        Generate interpretation based on divergence score and PDI
        """
        # Determine divergence level
        if divergence_score < self.DIVERGENCE_THRESHOLDS['very_low']:
            level = "Very Low Divergence - Consensus"
        elif divergence_score < self.DIVERGENCE_THRESHOLDS['low']:
            level = "Low Divergence"
        elif divergence_score < self.DIVERGENCE_THRESHOLDS['moderate']:
            level = "Moderate Divergence"
        elif divergence_score < self.DIVERGENCE_THRESHOLDS['high']:
            level = "High Divergence"
        else:
            level = "Very High Divergence"

        # Add power distance context
        if pdi < 0.2:
            power_context = "Low Power Distance"
        elif pdi < 0.4:
            power_context = "Some Power Distance Influence"
        elif pdi < 0.6:
            power_context = "Moderate Power Distance Influence"
        elif pdi < 0.8:
            power_context = "Significant Power Distance Influence"
        else:
            power_context = "Strong Power Distance Influence"

        return f"{level} ({power_context})"

    def get_stats(self) -> Dict:
        """Get statistics about the feedback data - using raw SQL to avoid type issues"""
        try:
            # Use raw SQL to get stats from Feedback360
            total_feedback = self.db.execute(text("SELECT COUNT(*) FROM feedback_360")).scalar() or 0

            # Get unique employees from feedback_360
            total_employees = self.db.execute(text("SELECT COUNT(DISTINCT evaluatee_id) FROM feedback_360")).scalar() or 0

            # Get unique institutions from feedback_360
            total_institutions = self.db.execute(text("SELECT COUNT(DISTINCT institution_id) FROM feedback_360")).scalar() or 0

            # Get periods from feedback_360
            periods_result = self.db.execute(text(
                "SELECT DISTINCT period_year, period_quarter FROM feedback_360 ORDER BY period_year, period_quarter"
            )).fetchall()

            periods_list = []
            for row in periods_result:
                year = row[0]
                quarter = row[1]
                # Convert if bytes
                if isinstance(year, bytes):
                    year = int.from_bytes(year, 'little')
                if isinstance(quarter, bytes):
                    quarter = int.from_bytes(quarter, 'little')
                periods_list.append({"year": year, "quarter": quarter})

            # Get evaluation type distribution
            type_dist_result = self.db.execute(text(
                "SELECT evaluation_type, COUNT(*) FROM feedback_360 GROUP BY evaluation_type"
            )).fetchall()

            type_dist_dict = {}
            for row in type_dist_result:
                type_dist_dict[row[0]] = row[1]

            # Get hierarchy distance distribution
            h_dist_result = self.db.execute(text(
                "SELECT hierarchy_distance, COUNT(*) FROM feedback_360 GROUP BY hierarchy_distance ORDER BY hierarchy_distance"
            )).fetchall()

            hierarchy_distribution = {}
            for row in h_dist_result:
                hierarchy_distribution[row[0]] = row[1]

            # Get competency stats
            comp_stats = {}
            for col in self.competency_cols + ['overall_rating']:
                min_val = self.db.execute(text(f"SELECT MIN({col}) FROM feedback_360")).scalar() or 0
                max_val = self.db.execute(text(f"SELECT MAX({col}) FROM feedback_360")).scalar() or 0
                avg_val = self.db.execute(text(f"SELECT AVG({col}) FROM feedback_360")).scalar() or 0

                comp_stats[col] = {
                    'min': float(min_val),
                    'max': float(max_val),
                    'mean': float(avg_val)
                }

            return {
                'total_evaluations': total_feedback,
                'total_employees': total_employees,
                'total_institutions': total_institutions,
                'periods_available': periods_list,
                'evaluation_type_distribution': type_dist_dict,
                'hierarchy_distance_distribution': hierarchy_distribution,
                'competency_stats': comp_stats
            }

        except Exception as e:
            logger.error(f"Error in get_stats: {e}")
            # Return empty stats
            return {
                'total_evaluations': 0,
                'total_employees': 0,
                'total_institutions': 0,
                'periods_available': [],
                'evaluation_type_distribution': {},
                'hierarchy_distance_distribution': {},
                'competency_stats': {}
            }

    # -------- Export Methods --------
    def export_divergence_analysis(self, institution_id: Optional[str] = None,
                                   period_year: Optional[int] = None,
                                   period_quarter: Optional[int] = None) -> pd.DataFrame:
        """
        Export divergence analysis as DataFrame with hierarchy-aware metrics
        """
        analysis = self.get_divergence_analysis(
            institution_id=institution_id,
            period_year=period_year,
            period_quarter=period_quarter
        )

        df = pd.DataFrame(analysis)

        # Add hierarchy-aware columns
        if not df.empty:
            # Add divergence level
            df['divergence_level'] = df['divergence_score'].apply(
                lambda x: self._get_divergence_level(x)
            )

            # Add PDI level
            df['pdi_level'] = df['pdi_institution'].apply(
                lambda x: self._get_pdi_level(x)
            )

        return df

    def _get_divergence_level(self, score: float) -> str:
        """Get divergence level label"""
        if score < self.DIVERGENCE_THRESHOLDS['very_low']:
            return 'Very Low'
        elif score < self.DIVERGENCE_THRESHOLDS['low']:
            return 'Low'
        elif score < self.DIVERGENCE_THRESHOLDS['moderate']:
            return 'Moderate'
        elif score < self.DIVERGENCE_THRESHOLDS['high']:
            return 'High'
        else:
            return 'Very High'

    def _get_pdi_level(self, pdi: float) -> str:
        """Get PDI level label"""
        if pdi < 0.2:
            return 'Very Low'
        elif pdi < 0.4:
            return 'Low'
        elif pdi < 0.6:
            return 'Moderate'
        elif pdi < 0.8:
            return 'High'
        else:
            return 'Very High'

    # -------- Batch Processing --------
    def process_batch(self, institution_id: Optional[str] = None,
                      period_year: Optional[int] = None,
                      period_quarter: Optional[int] = None) -> Dict:
        """
        Complete batch processing pipeline:
        1. Update hierarchy distances
        2. Aggregate performance data
        3. Calculate divergence scores
        """
        results = {
            'hierarchy_update': None,
            'aggregation': None
        }

        # Step 1: Update hierarchy distances
        if institution_id:
            results['hierarchy_update'] = self.update_hierarchy_distances(institution_id)
        else:
            results['hierarchy_update'] = self.update_hierarchy_distances()

        # Step 2: Aggregate and calculate divergence
        results['aggregation'] = self.aggregate_performance(
            institution_id=institution_id,
            period_year=period_year,
            period_quarter=period_quarter
        )

        return results