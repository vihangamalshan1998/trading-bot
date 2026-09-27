from sqlalchemy import create_engine
from core.database.base import Base
from core.database.models.ai import Experience, ModelVersion
from core.config.settings import settings

print("🚨 WARNING: THIS WILL WIPE ALL AI MEMORY AND TRAINING DATA 🚨")
print("Connecting to MySQL...")
engine = create_engine(settings.database_url)

print("Dropping all existing V2 tables...")
Base.metadata.drop_all(bind=engine)

print("Creating fresh V3 tables with Multi-Horizon columns...")
Base.metadata.create_all(bind=engine)

print("✅ V3 Database Schema successfully built!")
