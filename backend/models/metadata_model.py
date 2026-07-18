from sqlalchemy import Column, String, Integer, Float, DateTime
from datetime import datetime
from backend.database.database import Base

class ModelMetadata(Base):
    __tablename__ = 'model_metadata'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    algorithm_type = Column(String(20), nullable=False)
    training_date = Column(DateTime, default=datetime.utcnow)

    accuracy = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    cv_mean = Column(Float)

    feature_importance = Column(String(1000))  # JSON
    feature_names = Column(String(1000))  # JSON
    hyperparameters = Column(String(1000))  # JSON

    is_active = Column(Integer, default=1)

    def __repr__(self):
        return f"<Model {self.model_name} ({self.algorithm_type})>"