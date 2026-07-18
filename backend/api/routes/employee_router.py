from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database.database import get_db
from backend.services.master_service import MasterService
from backend.schemas.master_schema import (
    EmployeeResponse,
    EmployeeListResponse,
    EmployeeSearchResponse
)

router = APIRouter(prefix="/api/employees", tags=["employees"])

@router.get("/", response_model=EmployeeListResponse)  # Added trailing slash
def get_all_employees(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        db: Session = Depends(get_db)
):
    """Get all employees with pagination"""
    service = MasterService(db)
    employees, total = service.get_all_employees(skip, limit)
    return EmployeeListResponse(employees=employees, total=total)

@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(
        employee_id: str,
        db: Session = Depends(get_db)
):
    """Get employee by ID"""
    service = MasterService(db)
    employee = service.get_employee_by_id(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee

@router.get("/search", response_model=List[EmployeeSearchResponse])
def search_employees(
        q: str = Query(..., min_length=1),
        db: Session = Depends(get_db)
):
    """Search employees by ID"""
    service = MasterService(db)
    results = service.search_employees(q)
    return results

@router.get("/department/{department}", response_model=List[EmployeeResponse])
def get_employees_by_department(
        department: str,
        db: Session = Depends(get_db)
):
    """Get employees by department"""
    service = MasterService(db)
    employees = service.get_employees_by_department(department)
    return employees