# backend/api/routes/__init__.py
from backend.api.routes.employee_router import router as employee_router
from backend.api.routes.prediction_router import router as prediction_router
from backend.api.routes.feedback_router import router as feedback_router
from backend.api.routes.culture_router import router as culture_router
from backend.api.routes.diplomat_router import router as diplomat_router
from backend.api.routes.pdi_risk_router import router as pdi_risk_router
from backend.api.routes.llm_diplomat_router import router as llm_diplomat_router

__all__ = ['employee_router', 'prediction_router', 'feedback_router', 'culture_router',
           'diplomat_router', 'pdi_risk_router', 'llm_diplomat_router']