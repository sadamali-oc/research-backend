from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from backend.database.database import init_db
from backend.api.routes import employees_router, predictions_router

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
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
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
        "version": "1.0.0",
        "endpoints": {
            "employees": "/api/employees",
            "predictions": "/api/predictions",
            "docs": "/docs"
        }
    }

# Include routers
app.include_router(employees_router)
app.include_router(predictions_router)

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )