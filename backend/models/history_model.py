from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.database import Base

class PerformanceHistory(Base):
    """Store historical quarterly performance data"""
    __tablename__ = 'performance_history'
    __table_args__ = (
        UniqueConstraint('employee_id', 'period_year', 'period_quarter', name='uq_employee_quarter'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), nullable=False, index=True)
    period_year = Column(Integer, nullable=False)
    period_quarter = Column(String(2), nullable=False)  # Q1, Q2, Q3, Q4

    # Employee demographics (could be normalized later)
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

    # Project info
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

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PerformanceHistory {self.employee_id} {self.period_year} {self.period_quarter}>"


class PerformancePrediction(Base):
    """Store model predictions"""
    __tablename__ = 'performance_predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), nullable=False, index=True)
    period_year = Column(Integer, nullable=False)
    period_quarter = Column(String(2), nullable=False)

    # Prediction results
    predicted_score = Column(Float, nullable=False)
    predicted_band = Column(String(20), nullable=False)  # High/Medium/Low
    confidence = Column(Float, nullable=False)

    # Individual algorithm predictions
    rf_score = Column(Float, nullable=True)
    rf_band = Column(String(20), nullable=True)
    rf_confidence = Column(Float, nullable=True)
    gb_score = Column(Float, nullable=True)
    gb_band = Column(String(20), nullable=True)
    gb_confidence = Column(Float, nullable=True)

    # Model metadata
    algorithm_used = Column(String(50), nullable=False)  # RF/GB/Hybrid
    model_version = Column(String(50), nullable=True)
    feature_importance = Column(JSON, nullable=True)

    # Actual data (filled later)
    actual_score = Column(Float, nullable=True)
    actual_band = Column(String(20), nullable=True)

    predicted_at = Column(DateTime, default=datetime.utcnow)
    evaluated_at = Column(DateTime, nullable=True)  # When actual data becomes available

    def __repr__(self):
        return f"<PerformancePrediction {self.employee_id} {self.period_year}{self.period_quarter}: {self.predicted_band}>"


class ModelMetadata(Base):
    """Track trained models"""
    __tablename__ = 'model_metadata'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    algorithm_type = Column(String(20), nullable=False)  # RF/GB/Hybrid
    training_date = Column(DateTime, default=datetime.utcnow)

    # Performance metrics
    accuracy = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    cv_mean = Column(Float)

    # Model details
    feature_importance = Column(JSON)
    feature_names = Column(JSON)
    hyperparameters = Column(JSON)

    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive

    def __repr__(self):
        return f"<ModelMetadata {self.model_name} {self.algorithm_type}>"