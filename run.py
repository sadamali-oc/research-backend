import subprocess
import sys
import os
import time
import threading

def run_backend():
    """Run the FastAPI backend using Uvicorn"""
    # Points to backend/main.py -> app object
    command = f'"{sys.executable}" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload'
    os.system(command)

def run_frontend():
    """Run the Streamlit frontend"""
    time.sleep(3)  # Wait briefly for backend to spin up
    # Points to your frontend entry script
    command = f'"{sys.executable}" -m streamlit run frontend/app.py'
    os.system(command)

if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting Performance Appraisal System")
    print("="*60)

    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()

    # Start frontend in the main thread
    run_frontend()