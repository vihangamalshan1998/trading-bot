import os
import torch
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.database.models.ai import ModelVersion
from core.config.settings import settings
from core.logging.logger import logger

class ModelRegistry:
    """
    Manages saving/loading PyTorch weights and logging version metadata to MySQL.
    """
    def __init__(self, storage_dir: str = "data/models/weights"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Connect to MySQL synchronously for simple registry operations
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def save_model(self, model: torch.nn.Module, version_id: str, architecture: str, metrics: dict) -> str:
        """
        Saves the PyTorch state dict to disk and logs the version in MySQL.
        """
        file_path = os.path.join(self.storage_dir, f"{version_id}.pth")
        
        # Save weights
        torch.save(model.state_dict(), file_path)
        logger.info(f"Saved PyTorch weights to {file_path}")
        
        # Log to MySQL
        try:
            with self.SessionLocal() as session:
                new_version = ModelVersion(
                    version_id=version_id,
                    architecture=architecture,
                    file_path=file_path,
                    metrics=metrics,
                    is_active=False # Must be manually promoted to active
                )
                session.add(new_version)
                session.commit()
                logger.info(f"Registered model version {version_id} in MySQL.")
        except Exception as e:
            logger.error(f"Failed to register model in MySQL: {e}")
            
        return file_path
        
    def load_model(self, model: torch.nn.Module, version_id: str) -> torch.nn.Module:
        """
        Loads weights from disk for a specific version.
        """
        file_path = os.path.join(self.storage_dir, f"{version_id}.pth")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model weights not found at {file_path}")
            
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model
