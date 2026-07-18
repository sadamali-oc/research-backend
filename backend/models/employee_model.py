from sqlalchemy import Column, String, Integer, Float, DateTime
from datetime import datetime
from backend.database.database import Base

class EmployeePerformanceView(Base):
    __tablename__ = 'employee_performance_view'

    employee_id = Column(String(50), primary_key=True, index=True)
    period_year = Column(Integer)
    period_quarter = Column(String(2))

    institution = Column(String(100))
    date_of_birth = Column(String(20))
    gender = Column(String(20))
    age_group = Column(String(20))
    job_role = Column(String(100))
    years_of_experience = Column(Float)
    department = Column(String(100))
    language_proficiency = Column(String(50))
    ethnicity = Column(String(50))
    educational_institute = Column(String(200))

    punctuality = Column(Integer)
    problem_solving = Column(Integer)
    leadership = Column(Integer)
    collaboration = Column(Integer)
    communication = Column(Integer)

    deadline_adherence_rate = Column(Float)
    adherence_level = Column(String(20))
    avg_response_time = Column(Float)
    response_time_level = Column(String(20))
    no_of_meetings_attended = Column(Integer)
    no_of_subordinates = Column(Integer)
    decision_contribution = Column(Integer)
    learning_hours_per_month = Column(Float)
    type_of_learning = Column(String(100))
    team_engagement_frequency = Column(Integer)

    completed_storypoint_ratio = Column(Float)
    completed_story_points = Column(Float)
    assigned_story_points = Column(Float)

    project_id = Column(String(50))
    project_name = Column(String(200))
    duration_weeks = Column(Float)
    relative_effort = Column(Float)
    team_size = Column(Integer)
    project_complexity = Column(String(50))
    rework_count = Column(Integer)
    no_pay_leave = Column(Integer)
    blockers = Column(Integer)

    metric_1_name = Column(String(100))
    metric_1_value = Column(Float)
    metric_2_name = Column(String(100))
    metric_2_value = Column(Float)
    metric_3_name = Column(String(100))
    metric_3_value = Column(Float)
    metric_4_name = Column(String(100))
    metric_4_value = Column(Float)
    metric_5_name = Column(String(100))
    metric_5_value = Column(Float)
    metric_6_name = Column(String(100))
    metric_6_value = Column(Float)
    metric_7_name = Column(String(100))
    metric_7_value = Column(Float)

    def __repr__(self):
        return f"<Employee {self.employee_id}>"