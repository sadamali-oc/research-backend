from pydantic import BaseModel, field_validator
from typing import Optional, List, Union
from datetime import date, datetime

class EmployeeBase(BaseModel):
    employee_id: str
    institution: Optional[str] = None
    date_of_birth: Optional[datetime] = None
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
    adherence_level: Optional[Union[str, int]] = None  # Accept both
    avg_response_time: Optional[float] = 0.0
    response_time_level: Optional[Union[str, int]] = None  # Accept both
    no_of_meetings_attended: Optional[int] = 0
    no_of_subordinates: Optional[int] = 0
    decision_contribution: Optional[int] = 0
    learning_hours_per_month: Optional[float] = 0.0
    type_of_learning: Optional[str] = None
    team_engagement_frequency: Optional[int] = 0
    completed_storypoint_ratio: Optional[float] = 0.0
    completed_story_points: Optional[float] = 0.0
    assigned_story_points: Optional[float] = 0.0
    project_id: Optional[Union[str, int]] = None  # Accept both
    project_name: Optional[str] = None
    duration_weeks: Optional[float] = 0.0
    relative_effort: Optional[float] = 0.0
    team_size: Optional[int] = 0
    project_complexity: Optional[Union[str, float]] = None  # Accept both
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

    @field_validator('adherence_level', mode='before')
    def validate_adherence_level(cls, v):
        if v is None:
            return None
        # Map integer values to strings
        if isinstance(v, int):
            mapping = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Medium', 5: 'High'}
            return mapping.get(v, str(v))
        return v

    @field_validator('response_time_level', mode='before')
    def validate_response_time_level(cls, v):
        if v is None:
            return None
        if isinstance(v, int):
            mapping = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Medium', 5: 'High'}
            return mapping.get(v, str(v))
        return v

    @field_validator('project_complexity', mode='before')
    def validate_project_complexity(cls, v):
        if v is None:
            return None
        if isinstance(v, float):
            # Map float to string
            if v <= 2.0:
                return 'Low'
            elif v <= 5.0:
                return 'Medium'
            elif v <= 8.0:
                return 'High'
            else:
                return 'Very High'
        return v

    @field_validator('project_id', mode='before')
    def validate_project_id(cls, v):
        if v is None:
            return None
        return str(v)

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