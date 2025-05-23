from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Add echo=True to see SQL queries
engine = create_engine(
    settings.DATABASE_URL,
    echo=True,  # This will log all SQL operations
    pool_pre_ping=True  # This will enable automatic reconnection
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
