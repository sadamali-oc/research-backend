# backend/models/feedback_model.py
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Feedback360(Base):
    """360-degree feedback evaluation data"""
    __tablename__ = "feedback_360"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(String(50), unique=True, nullable=False, index=True)
    evaluatee_id = Column(String(50), nullable=False, index=True)
    evaluator_id = Column(String(50), nullable=False, index=True)
    evaluation_type = Column(String(50), nullable=False)  # Self, Manager, Peer, Subordinate
    relationship = Column(String(100))  # Self Appraisal, Direct Manager, Team Member, Project Peer, etc.
    institution_id = Column(String(50), nullable=False, index=True)
    period_year = Column(Integer, nullable=False)
    period_quarter = Column(Integer, nullable=False)
    hierarchy_distance = Column(Integer, default=0)

    # Competency scores (1-5)
    punctuality = Column(Float, nullable=False)
    problem_solving = Column(Float, nullable=False)
    leadership = Column(Float, nullable=False)
    collaboration = Column(Float, nullable=False)
    communication = Column(Float, nullable=False)
    overall_rating = Column(Float, nullable=False)

    # Text feedback
    feedback_text = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.now)

    # Composite index for fast querying
    __table_args__ = (
        Index('idx_feedback_evaluatee_period', 'evaluatee_id', 'period_year', 'period_quarter'),
        Index('idx_feedback_evaluator_type', 'evaluator_id', 'evaluation_type'),
        Index('idx_feedback_institution_period', 'institution_id', 'period_year', 'period_quarter'),
    )

    def __repr__(self):
        return f"<Feedback360(evaluation_id={self.evaluation_id}, evaluatee={self.evaluatee_id}, type={self.evaluation_type})>"


class AggregatedPerformance(Base):
    """Aggregated performance data per employee per quarter"""
    __tablename__ = "aggregated_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), nullable=False, index=True)
    institution_id = Column(String(50), nullable=False, index=True)
    period_year = Column(Integer, nullable=False)
    period_quarter = Column(Integer, nullable=False)

    # Self scores (average)
    self_punctuality = Column(Float)
    self_problem_solving = Column(Float)
    self_leadership = Column(Float)
    self_collaboration = Column(Float)
    self_communication = Column(Float)
    self_overall = Column(Float)

    # Manager scores (average)
    manager_punctuality = Column(Float)
    manager_problem_solving = Column(Float)
    manager_leadership = Column(Float)
    manager_collaboration = Column(Float)
    manager_communication = Column(Float)
    manager_overall = Column(Float)

    # Peer scores (average)
    peer_punctuality = Column(Float)
    peer_problem_solving = Column(Float)
    peer_leadership = Column(Float)
    peer_collaboration = Column(Float)
    peer_communication = Column(Float)
    peer_overall = Column(Float)

    # Subordinate scores (average)
    sub_punctuality = Column(Float)
    sub_problem_solving = Column(Float)
    sub_leadership = Column(Float)
    sub_collaboration = Column(Float)
    sub_communication = Column(Float)
    sub_overall = Column(Float)

    # Aggregated metrics
    performance_score = Column(Float)  # Overall performance (all sources)
    relational_score = Column(Float)   # Average of collaboration + communication
    divergence_score = Column(Float)   # Standard deviation across evaluation types

    # Counts
    self_count = Column(Integer, default=0)
    manager_count = Column(Integer, default=0)
    peer_count = Column(Integer, default=0)
    sub_count = Column(Integer, default=0)

    # PDI proxy (institution level)
    pdi_institution = Column(Float)

    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        Index('idx_agg_employee_period', 'employee_id', 'period_year', 'period_quarter'),
        Index('idx_agg_institution_period', 'institution_id', 'period_year', 'period_quarter'),
    )

    def __repr__(self):
        return f"<AggregatedPerformance(employee={self.employee_id}, period={self.period_year}Q{self.period_quarter}, score={self.performance_score})>"