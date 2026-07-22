# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from backend.database.database import init_db
from backend.api.routes.employee_router import router as employee_router
from backend.api.routes.prediction_router import router as prediction_router
from backend.api.routes.feedback_router import router as feedback_router  # ✅ Already imported
from backend.api.routes.culture_router import router as culture_router  # ✅ Already imported

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting up...")
    init_db()
    print("✅ Database initialized")
    yield
    # Shutdown
    print("👋 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Employee Performance Appraisal API",
    description="AI-based employee performance appraisal system",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "message": "API is running"}
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Employee Performance Appraisal API",
        "version": "2.0.0",
        "endpoints": {
            "employees": "/api/employees",
            "predictions": "/api/predictions",
            "feedback": "/api/feedback",  # ✅ Added feedback endpoint
            "docs": "/docs"
        }
    }

# Include routers
app.include_router(employee_router)
app.include_router(prediction_router)
app.include_router(feedback_router)  # ✅ Already registered
app.include_router(culture_router)  # ✅ Already registered

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )