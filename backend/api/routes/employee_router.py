from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database.database import get_db
from backend.services.employee_service import EmployeeService
from backend.schemas.employee_schema import (
    EmployeeResponse,
    EmployeeListResponse,
    EmployeeSearchResponse,
    EmployeeHistoryResponse
)

# Create the router instance
router = APIRouter(prefix="/api/employees", tags=["employees"])

@router.get("/", response_model=EmployeeListResponse)
def get_all_employees(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        db: Session = Depends(get_db)
):
    """Get all employees with pagination"""
    service = EmployeeService(db)
    employees, total = service.get_all_employees(skip, limit)
    return EmployeeListResponse(employees=employees, total=total)

@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(
        employee_id: str,
        db: Session = Depends(get_db)
):
    """Get employee by ID"""
    service = EmployeeService(db)
    employee = service.get_employee_by_id(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee

@router.get("/search/", response_model=List[EmployeeSearchResponse])
def search_employees(
        q: str = Query(..., min_length=1),
        db: Session = Depends(get_db)
):
    """Search employees by ID"""
    service = EmployeeService(db)
    results = service.search_employees(q)
    return results

@router.get("/history/{employee_id}", response_model=List[EmployeeHistoryResponse])
def get_employee_history(
        employee_id: str,
        db: Session = Depends(get_db)
):
    """Get historical performance data for an employee"""
    service = EmployeeService(db)
    history = service.get_employee_history(employee_id)
    return history

@router.get("/quarters/", response_model=dict)
def get_quarters_info(
        db: Session = Depends(get_db)
):
    """Get information about available quarters"""
    service = EmployeeService(db)
    quarters = service.get_quarters_info()
    return quarters