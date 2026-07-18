from sqlalchemy import Column, String, Integer, Float, DateTime
from datetime import datetime
from backend.database.database import Base

class PerformanceResult(Base):
    __tablename__ = 'performance_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), nullable=False, index=True)
    performance_score = Column(Float, nullable=False)
    performance_band = Column(String(20), nullable=False)
    confidence = Column(Float)
    feature_snapshot = Column(String(500))
    model_version = Column(String(50))
    predicted_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Result {self.employee_id}: {self.performance_band} ({self.performance_score:.1f}%)>"