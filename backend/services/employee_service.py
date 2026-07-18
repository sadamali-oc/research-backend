from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from typing import List, Optional, Tuple, Dict
from datetime import datetime

from backend.models.employee_model import EmployeePerformanceView
from backend.models.history_model import PerformanceHistory
from backend.schemas.employee_schema import EmployeeHistoryResponse

class EmployeeService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def get_all_employees(self, skip: int = 0, limit: int = 100) -> Tuple[List[dict], int]:
        """Get all employees with pagination"""
        try:
            query = self.db.query(EmployeePerformanceView)
            total = query.count()
            employees = query.offset(skip).limit(limit).all()

            result = []
            for emp in employees:
                result.append({
                    'employee_id': emp.employee_id,
                    'period_year': emp.period_year,
                    'period_quarter': emp.period_quarter,
                    'institution': emp.institution,
                    'job_role': emp.job_role,
                    'years_of_experience': emp.years_of_experience,
                    'department': emp.department,
                    'punctuality': emp.punctuality,
                    'problem_solving': emp.problem_solving,
                    'leadership': emp.leadership,
                    'collaboration': emp.collaboration,
                    'communication': emp.communication,
                    'deadline_adherence_rate': emp.deadline_adherence_rate,
                    'adherence_level': emp.adherence_level,
                    'avg_response_time': emp.avg_response_time,
                    'response_time_level': emp.response_time_level,
                    'no_of_meetings_attended': emp.no_of_meetings_attended,
                    'no_of_subordinates': emp.no_of_subordinates,
                    'decision_contribution': emp.decision_contribution,
                    'learning_hours_per_month': emp.learning_hours_per_month,
                    'type_of_learning': emp.type_of_learning,
                    'team_engagement_frequency': emp.team_engagement_frequency,
                    'completed_storypoint_ratio': emp.completed_storypoint_ratio,
                    'completed_story_points': emp.completed_story_points,
                    'assigned_story_points': emp.assigned_story_points,
                    'project_id': emp.project_id,
                    'project_name': emp.project_name,
                    'duration_weeks': emp.duration_weeks,
                    'relative_effort': emp.relative_effort,
                    'team_size': emp.team_size,
                    'project_complexity': emp.project_complexity,
                    'rework_count': emp.rework_count,
                    'no_pay_leave': emp.no_pay_leave,
                    'blockers': emp.blockers,
                    'metric_1_name': emp.metric_1_name,
                    'metric_1_value': emp.metric_1_value,
                    'metric_2_name': emp.metric_2_name,
                    'metric_2_value': emp.metric_2_value,
                    'metric_3_name': emp.metric_3_name,
                    'metric_3_value': emp.metric_3_value,
                    'metric_4_name': emp.metric_4_name,
                    'metric_4_value': emp.metric_4_value,
                    'metric_5_name': emp.metric_5_name,
                    'metric_5_value': emp.metric_5_value,
                    'metric_6_name': emp.metric_6_name,
                    'metric_6_value': emp.metric_6_value,
                    'metric_7_name': emp.metric_7_name,
                    'metric_7_value': emp.metric_7_value
                })

            return result, total
        except Exception as e:
            print(f"Error in get_all_employees: {e}")
            return [], 0

    def get_employee_by_id(self, employee_id: str) -> Optional[dict]:
        """Get employee by ID"""
        try:
            emp = self.db.query(EmployeePerformanceView).filter_by(
                employee_id=employee_id
            ).first()

            if emp:
                return {
                    'employee_id': emp.employee_id,
                    'period_year': emp.period_year,
                    'period_quarter': emp.period_quarter,
                    'job_role': emp.job_role,
                    'department': emp.department,
                    'years_of_experience': emp.years_of_experience,
                    'punctuality': emp.punctuality,
                    'problem_solving': emp.problem_solving,
                    'leadership': emp.leadership,
                    'collaboration': emp.collaboration,
                    'communication': emp.communication,
                    'deadline_adherence_rate': emp.deadline_adherence_rate,
                    'avg_response_time': emp.avg_response_time,
                    'completed_storypoint_ratio': emp.completed_storypoint_ratio
                }
            return None
        except Exception as e:
            print(f"Error in get_employee_by_id: {e}")
            return None

    def search_employees(self, query: str) -> List[dict]:
        """Search employees by ID"""
        try:
            results = self.db.query(EmployeePerformanceView).filter(
                EmployeePerformanceView.employee_id.ilike(f"%{query}%")
            ).limit(20).all()

            return [{
                'employee_id': r.employee_id,
                'name': r.employee_id,
                'job_role': r.job_role or 'N/A',
                'department': r.department or 'N/A'
            } for r in results]
        except Exception as e:
            print(f"Error in search_employees: {e}")
            return []

    def get_employee_history(self, employee_id: str) -> List[dict]:
        """Get historical performance data for an employee"""
        try:
            records = self.db.query(PerformanceHistory).filter_by(
                employee_id=employee_id
            ).order_by(
                PerformanceHistory.period_year,
                PerformanceHistory.period_quarter
            ).all()

            history = []
            for r in records:
                score = (
                        r.deadline_adherence_rate * 0.25 +
                        r.completed_storypoint_ratio * 100 * 0.25 +
                        r.punctuality * 5 * 0.10 +
                        r.problem_solving * 5 * 0.10 +
                        r.leadership * 5 * 0.10 +
                        r.collaboration * 5 * 0.10 +
                        r.communication * 5 * 0.10
                )

                history.append({
                    'period': f"{r.period_year} {r.period_quarter}",
                    'year': r.period_year,
                    'quarter': r.period_quarter,
                    'score': round(score, 2),
                    'band': 'High' if score >= 70 else 'Medium' if score >= 40 else 'Low',
                    'deadline_adherence': r.deadline_adherence_rate,
                    'punctuality': r.punctuality,
                    'problem_solving': r.problem_solving,
                    'leadership': r.leadership,
                    'collaboration': r.collaboration,
                    'communication': r.communication
                })

            return history
        except Exception as e:
            print(f"Error in get_employee_history: {e}")
            return []

    def get_quarters_info(self) -> dict:
        """Get information about available quarters"""
        try:
            # Get distinct quarters
            quarters = self.db.query(
                PerformanceHistory.period_year,
                PerformanceHistory.period_quarter
            ).distinct().order_by(
                PerformanceHistory.period_year,
                PerformanceHistory.period_quarter
            ).all()

            total_quarters = len(quarters)

            return {
                'total_quarters': total_quarters,
                'quarters': [f"{q.period_year} {q.period_quarter}" for q in quarters],
                'years': list(set(q.period_year for q in quarters))
            }
        except Exception as e:
            print(f"Error in get_quarters_info: {e}")
            return {'total_quarters': 0, 'quarters': [], 'years': []}