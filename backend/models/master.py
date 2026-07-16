from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.database import Base
import enum

class GenderEnum(enum.Enum):
    Male = "Male"
    Female = "Female"
    Other = "Other"

class JobRoleEnum(enum.Enum):
    Software_Engineer = "Software Engineer"
    Senior_Software_Engineer = "Senior Software Engineer"
    Team_Lead = "Team Lead"
    Project_Manager = "Project Manager"
    QA_Engineer = "QA Engineer"
    DevOps_Engineer = "DevOps Engineer"
    Data_Scientist = "Data Scientist"
    Business_Analyst = "Business Analyst"
    UI_UX_Designer = "UI/UX Designer"
    Technical_Architect = "Technical Architect"

class DepartmentEnum(enum.Enum):
    Development = "Development"
    QA = "QA"
    DevOps = "DevOps"
    Data_Science = "Data Science"
    Product = "Product"
    Design = "Design"
    Project_Management = "Project Management"
    Business_Analysis = "Business Analysis"

class LanguageEnum(enum.Enum):
    Sinhala = "Sinhala"
    Tamil = "Tamil"
    English = "English"

class EthnicityEnum(enum.Enum):
    Sinhalese = "Sinhalese"
    Tamil = "Tamil"
    Muslim = "Muslim"
    Burgher = "Burgher"
    Malay = "Malay"

class AdherenceLevelEnum(enum.Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"

class ResponseTimeLevelEnum(enum.Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"

class LearningTypeEnum(enum.Enum):
    Online_Courses = "Online courses / certifications"
    Workshops = "Workshops"
    Self_Study = "Self-study"
    Mentorship = "Mentorship"
    University_Courses = "University courses"

class ProjectComplexityEnum(enum.Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"
    Very_High = "Very High"

class Employee(Base):
    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True)
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    institution = Column(String(100))
    date_of_birth = Column(Date)
    gender = Column(Enum(GenderEnum))
    age_group = Column(String(20))
    job_role = Column(Enum(JobRoleEnum))
    years_of_experience = Column(Float)
    department = Column(Enum(DepartmentEnum))
    language_proficiency = Column(Enum(LanguageEnum))
    ethnicity = Column(Enum(EthnicityEnum))
    educational_institute = Column(String(200))

    # Behavioral metrics
    punctuality = Column(Integer)  # 1-5
    problem_solving = Column(Integer)  # 1-5
    leadership = Column(Integer)  # 1-5
    collaboration = Column(Integer)  # 1-5
    communication = Column(Integer)  # 1-5

    # Performance metrics
    deadline_adherence_rate = Column(Float)  # percentage
    adherence_level = Column(Enum(AdherenceLevelEnum))
    avg_response_time = Column(Float)  # hours
    response_time_level = Column(Enum(ResponseTimeLevelEnum))
    no_of_meetings_attended = Column(Integer)
    no_of_subordinates = Column(Integer)
    decision_contribution = Column(Integer)  # 1-5
    learning_hours_per_month = Column(Float)
    type_of_learning = Column(Enum(LearningTypeEnum))
    team_engagement_frequency = Column(Integer)  # 1-5 scale

    # Story point metrics
    completed_storypoint_ratio = Column(Float)
    completed_story_points = Column(Float)
    assigned_story_points = Column(Float)

    # Additional metrics (stored as JSON for flexibility)
    additional_metrics = Column(JSON)  # Stores Metric 1-7 name-value pairs

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    projects = relationship("EmployeeProject", back_populates="employee")
    learning_activities = relationship("LearningActivity", back_populates="employee")
    meetings = relationship("MeetingAttendance", back_populates="employee")
    performance_metrics = relationship("PerformanceMetric", back_populates="employee")

    def __repr__(self):
        return f"<Employee {self.employee_id}: {self.job_role}>"

class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    project_id = Column(String(50), unique=True, nullable=False, index=True)
    project_name = Column(String(200), nullable=False)
    duration_weeks = Column(Float)
    relative_effort = Column(Float)  # Story points
    team_size = Column(Integer)
    project_complexity = Column(Enum(ProjectComplexityEnum))
    rework_count = Column(Integer)
    no_pay_leave = Column(Integer)
    blockers = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    employees = relationship("EmployeeProject", back_populates="project")

    def __repr__(self):
        return f"<Project {self.project_id}: {self.project_name}>"

class EmployeeProject(Base):
    __tablename__ = 'employee_projects'

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    project_id = Column(Integer, ForeignKey('projects.id'))
    role_in_project = Column(String(100))
    start_date = Column(Date)
    end_date = Column(Date)

    # Project-specific metrics
    tasks_assigned = Column(Integer)
    tasks_completed = Column(Integer)
    tasks_on_time = Column(Integer)
    bug_count = Column(Integer)
    bugs_fixed_count = Column(Integer)
    code_quality = Column(Integer)  # 1-10
    rework_count = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    employee = relationship("Employee", back_populates="projects")
    project = relationship("Project", back_populates="employees")

class LearningActivity(Base):
    __tablename__ = 'learning_activities'

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    activity_date = Column(Date)
    activity_type = Column(String(100))
    hours_spent = Column(Float)
    certification_earned = Column(String(200))
    provider = Column(String(200))

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    employee = relationship("Employee", back_populates="learning_activities")

class MeetingAttendance(Base):
    __tablename__ = 'meeting_attendances'

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    meeting_date = Column(DateTime)
    meeting_type = Column(String(100))
    duration_minutes = Column(Integer)
    contribution_score = Column(Integer)  # 1-5

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    employee = relationship("Employee", back_populates="meetings")

class PerformanceMetric(Base):
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    metric_date = Column(Date)

    # We'll store all metrics as key-value pairs for flexibility
    metric_name = Column(String(100))
    metric_value = Column(Float)
    metric_scale = Column(String(20))  # e.g., "1-10", "1-5", "percentage"
    category = Column(String(50))  # e.g., "Quality", "Productivity", "Leadership"

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    employee = relationship("Employee", back_populates="performance_metrics")

# We'll also create a view-friendly table that mirrors your CSV structure
class EmployeePerformanceView(Base):
    __tablename__ = 'employee_performance_view'

    id = Column(Integer, primary_key=True)
    employee_id = Column(String(50))
    institution = Column(String(100))
    date_of_birth = Column(Date)
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

    __table_args__ = {'extend_existing': True}