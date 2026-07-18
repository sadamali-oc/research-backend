from fastapi import Depends
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.services.prediction_service import PredictionService
from backend.services.employee_service import EmployeeService

def get_prediction_service(db: Session = Depends(get_db)):
    return PredictionService(db)

def get_employee_service(db: Session = Depends(get_db)):
    return EmployeeService(db)