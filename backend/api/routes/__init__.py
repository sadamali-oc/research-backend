# backend/api/routes/__init__.py
from backend.api.routes.employee_router import router as employee_router
from backend.api.routes.prediction_router import router as prediction_router
from backend.api.routes.feedback_router import router as feedback_router
from backend.api.routes.culture_router import router as culture_router

__all__ = ['employee_router', 'prediction_router', 'feedback_router', 'culture_router']