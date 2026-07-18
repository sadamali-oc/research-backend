from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class EmployeeBase(BaseModel):
    employee_id: str
    period_year: Optional[int] = None
    period_quarter: Optional[str] = None
    institution: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    job_role: Optional[str] = None
    years_of_experience: Optional[float] = 0.0
    department: Optional[str] = None
    language_proficiency: Optional[str] = None
    ethnicity: Optional[str] = None
    educational_institute: Optional[str] = None
    punctuality: Optional[int] = 0
    problem_solving: Optional[int] = 0
    leadership: Optional[int] = 0
    collaboration: Optional[int] = 0
    communication: Optional[int] = 0
    deadline_adherence_rate: Optional[float] = 0.0
    adherence_level: Optional[str] = None
    avg_response_time: Optional[float] = 0.0
    response_time_level: Optional[str] = None
    no_of_meetings_attended: Optional[int] = 0
    no_of_subordinates: Optional[int] = 0
    decision_contribution: Optional[int] = 0
    learning_hours_per_month: Optional[float] = 0.0
    type_of_learning: Optional[str] = None
    team_engagement_frequency: Optional[int] = 0
    completed_storypoint_ratio: Optional[float] = 0.0
    completed_story_points: Optional[float] = 0.0
    assigned_story_points: Optional[float] = 0.0
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    duration_weeks: Optional[float] = 0.0
    relative_effort: Optional[float] = 0.0
    team_size: Optional[int] = 0
    project_complexity: Optional[str] = None
    rework_count: Optional[int] = 0
    no_pay_leave: Optional[int] = 0
    blockers: Optional[int] = 0
    metric_1_name: Optional[str] = None
    metric_1_value: Optional[float] = 0.0
    metric_2_name: Optional[str] = None
    metric_2_value: Optional[float] = 0.0
    metric_3_name: Optional[str] = None
    metric_3_value: Optional[float] = 0.0
    metric_4_name: Optional[str] = None
    metric_4_value: Optional[float] = 0.0
    metric_5_name: Optional[str] = None
    metric_5_value: Optional[float] = 0.0
    metric_6_name: Optional[str] = None
    metric_6_value: Optional[float] = 0.0
    metric_7_name: Optional[str] = None
    metric_7_value: Optional[float] = 0.0

class EmployeeResponse(EmployeeBase):
    class Config:
        from_attributes = True

class EmployeeListResponse(BaseModel):
    employees: List[EmployeeResponse]
    total: int

class EmployeeSearchResponse(BaseModel):
    employee_id: str
    name: str
    job_role: str
    department: str

class EmployeeHistoryResponse(BaseModel):
    period: str
    year: int
    quarter: str
    score: float
    band: str
    deadline_adherence: float
    punctuality: int
    problem_solving: int
    leadership: int
    collaboration: int
    communication: int