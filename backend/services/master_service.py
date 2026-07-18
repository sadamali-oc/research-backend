from sqlalchemy.orm import Session
from typing import List, Optional, Tuple
from backend.models.master import EmployeePerformanceView

class MasterService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def get_all_employees(self, skip: int = 0, limit: int = 100) -> Tuple[List[dict], int]:
        """Get all employees with pagination"""
        try:
            query = self.db.query(EmployeePerformanceView)
            total = query.count()
            employees = query.offset(skip).limit(limit).all()

            # Convert to list of dicts
            result = []
            for emp in employees:
                result.append({
                    'employee_id': emp.employee_id,
                    'institution': emp.institution,
                    'date_of_birth': emp.date_of_birth,
                    'gender': emp.gender,
                    'age_group': emp.age_group,
                    'job_role': emp.job_role,
                    'years_of_experience': emp.years_of_experience,
                    'department': emp.department,
                    'language_proficiency': emp.language_proficiency,
                    'ethnicity': emp.ethnicity,
                    'educational_institute': emp.educational_institute,
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
            employee = self.db.query(EmployeePerformanceView).filter_by(
                employee_id=employee_id
            ).first()

            if employee:
                return {
                    'employee_id': employee.employee_id,
                    'institution': employee.institution,
                    'job_role': employee.job_role,
                    'department': employee.department,
                    'years_of_experience': employee.years_of_experience,
                    'punctuality': employee.punctuality,
                    'problem_solving': employee.problem_solving,
                    'leadership': employee.leadership,
                    'collaboration': employee.collaboration,
                    'communication': employee.communication,
                    'deadline_adherence_rate': employee.deadline_adherence_rate,
                    'adherence_level': employee.adherence_level,
                    'avg_response_time': employee.avg_response_time,
                    'response_time_level': employee.response_time_level,
                    'no_of_meetings_attended': employee.no_of_meetings_attended,
                    'no_of_subordinates': employee.no_of_subordinates,
                    'decision_contribution': employee.decision_contribution,
                    'learning_hours_per_month': employee.learning_hours_per_month,
                    'type_of_learning': employee.type_of_learning,
                    'team_engagement_frequency': employee.team_engagement_frequency,
                    'completed_storypoint_ratio': employee.completed_storypoint_ratio,
                    'completed_story_points': employee.completed_story_points,
                    'assigned_story_points': employee.assigned_story_points,
                    'project_id': employee.project_id,
                    'project_name': employee.project_name,
                    'duration_weeks': employee.duration_weeks,
                    'relative_effort': employee.relative_effort,
                    'team_size': employee.team_size,
                    'project_complexity': employee.project_complexity,
                    'rework_count': employee.rework_count,
                    'no_pay_leave': employee.no_pay_leave,
                    'blockers': employee.blockers,
                    'metric_1_name': employee.metric_1_name,
                    'metric_1_value': employee.metric_1_value,
                    'metric_2_name': employee.metric_2_name,
                    'metric_2_value': employee.metric_2_value,
                    'metric_3_name': employee.metric_3_name,
                    'metric_3_value': employee.metric_3_value,
                    'metric_4_name': employee.metric_4_name,
                    'metric_4_value': employee.metric_4_value,
                    'metric_5_name': employee.metric_5_name,
                    'metric_5_value': employee.metric_5_value,
                    'metric_6_name': employee.metric_6_name,
                    'metric_6_value': employee.metric_6_value,
                    'metric_7_name': employee.metric_7_name,
                    'metric_7_value': employee.metric_7_value
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

    def get_employees_by_department(self, department: str) -> List[dict]:
        """Get employees by department"""
        try:
            employees = self.db.query(EmployeePerformanceView).filter_by(
                department=department
            ).all()

            return [{
                'employee_id': e.employee_id,
                'job_role': e.job_role,
                'department': e.department
            } for e in employees]
        except Exception as e:
            print(f"Error in get_employees_by_department: {e}")
            return []