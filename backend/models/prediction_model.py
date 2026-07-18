from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey
from datetime import datetime
from backend.database.database import Base

class PerformanceResult(Base):
    __tablename__ = "performance_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), nullable=False, index=True)
    performance_score = Column(Float, nullable=False)
    performance_band = Column(String(20), nullable=False)  # High / Medium / Low
    confidence = Column(Float, nullable=True)
    feature_snapshot = Column(JSON, nullable=True)  # Store feature importance
    model_version = Column(String(50), nullable=True)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PerformanceResult {self.employee_id}: {self.performance_band} ({self.performance_score:.2f})>"