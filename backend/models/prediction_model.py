from sqlalchemy import Column, String, Integer, Float, DateTime
from datetime import datetime
from backend.database.database import Base

class PerformancePrediction(Base):
    __tablename__ = 'performance_predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), nullable=False, index=True)
    period_year = Column(Integer, nullable=False)
    period_quarter = Column(String(2), nullable=False)

    predicted_score = Column(Float, nullable=False)
    predicted_band = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)

    rf_score = Column(Float)
    rf_band = Column(String(20))
    rf_confidence = Column(Float)
    gb_score = Column(Float)
    gb_band = Column(String(20))
    gb_confidence = Column(Float)

    algorithm_used = Column(String(50), nullable=False)
    model_version = Column(String(50))
    feature_importance = Column(String(500))  # JSON string

    actual_score = Column(Float)
    actual_band = Column(String(20))

    predicted_at = Column(DateTime, default=datetime.utcnow)
    evaluated_at = Column(DateTime)

    def __repr__(self):
        return f"<Prediction {self.employee_id} {self.period_year}{self.period_quarter}: {self.predicted_band}>"