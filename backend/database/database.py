from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import os
import time

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

DATABASE_URL = "sqlite:///./data/research_system.db"

# Create engine with proper settings for SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "timeout": 60,  # Increased timeout for lock wait
        "check_same_thread": False
    },
    pool_size=1,  # SQLite works best with single connection
    max_overflow=0,
    pool_timeout=30
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise e
    finally:
        SessionLocal.remove()

def init_db():
    retries = 3
    for i in range(retries):
        try:
            # First, try to enable WAL mode with a separate connection
            try:
                import sqlite3
                conn = sqlite3.connect('data/research_system.db', timeout=30)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.close()
                print("⚡ WAL Mode enabled")
            except Exception as e:
                print(f"⚠️ Could not set WAL mode: {e}")

            # Then create tables
            Base.metadata.create_all(bind=engine)
            print("✅ Database initialized at ./data/research_system.db")
            break
        except Exception as e:
            if i < retries - 1:
                print(f"⚠️ Database init failed, retrying... ({i+1}/{retries})")
                time.sleep(2)
            else:
                print(f"❌ Database init failed: {e}")
                raise