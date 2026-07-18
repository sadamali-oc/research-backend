from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.database import Base
import enum

# Enums
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

# Main View - This is what we're using
class EmployeePerformanceView(Base):
    __tablename__ = 'employee_performance_view'

    # Use employee_id as primary key since it's unique
    employee_id = Column(String(50), primary_key=True, index=True)

    institution = Column(String(100))
    date_of_birth = Column(DateTime)
    gender = Column(String(20))
    age_group = Column(String(20))
    job_role = Column(String(100))
    years_of_experience = Column(Float)
    department = Column(String(100))
    language_proficiency = Column(String(50))
    ethnicity = Column(String(50))
    educational_institute = Column(String(200))

    # Behavioral metrics
    punctuality = Column(Integer)
    problem_solving = Column(Integer)
    leadership = Column(Integer)
    collaboration = Column(Integer)
    communication = Column(Integer)

    # Performance metrics
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

    # Story point metrics
    completed_storypoint_ratio = Column(Float)
    completed_story_points = Column(Float)
    assigned_story_points = Column(Float)

    # Project Information
    project_id = Column(String(50))
    project_name = Column(String(200))
    duration_weeks = Column(Float)
    relative_effort = Column(Float)
    team_size = Column(Integer)
    project_complexity = Column(String(50))
    rework_count = Column(Integer)
    no_pay_leave = Column(Integer)
    blockers = Column(Integer)

    # Metrics 1-7
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
        return f"<EmployeePerformanceView {self.employee_id}>"

# If you need other models, uncomment and define them
# class Employee(Base):
#     __tablename__ = 'employees'
#     # ... fields ...
#
# class Project(Base):
#     __tablename__ = 'projects'
#     # ... fields ...