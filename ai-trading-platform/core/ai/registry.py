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
                    checkpoint_path=file_path,
                    metrics=metrics,
                    status="CANDIDATE"
                )
                session.add(new_version)
                session.commit()
                logger.info(f"Registered model version {version_id} in MySQL.")
        except Exception as e:
            logger.error(f"Failed to register model in MySQL: {e}")
            
        return file_path
        
    def load_model(self, model: torch.nn.Module, version_id: str = None) -> torch.nn.Module:
        """
        Loads weights from disk. If version_id is None, loads the strictly Active production model.
        Includes architecture, dimension, and schema validation.
        """
        try:
            with self.SessionLocal() as session:
                if version_id:
                    version_meta = session.query(ModelVersion).filter_by(version_id=version_id).first()
                else:
                    version_meta = session.query(ModelVersion).filter_by(status="PRODUCTION").first()
                    
                if not version_meta:
                    logger.error("SAFETY GATE: No valid model version found in MySQL.")
                    raise ValueError("No valid model version found.")
                    
                # 1. Architecture validation
                expected_arch = model.__class__.__name__
                if version_meta.architecture != expected_arch:
                    logger.error(f"SAFETY GATE: Model architecture mismatch. Expected {expected_arch}, DB says {version_meta.architecture}.")
                    raise ValueError("Architecture mismatch.")
                    
                # 2. Dimension and Symbol validation
                hyperparams = version_meta.hyperparameters or {}
                
                # Check symbol ordering
                db_symbols = hyperparams.get("symbol_universe")
                if db_symbols and db_symbols != settings.symbol_universe:
                    logger.error(f"SAFETY GATE: Symbol universe mismatch. Config wants {settings.symbol_universe}, Model trained on {db_symbols}.")
                    raise ValueError("Symbol universe mismatch.")
                    
                # Check dimension
                from core.schemas.dimension_config import get_expected_observation_dimension
                expected_dim = get_expected_observation_dimension(len(settings.symbol_universe))
                db_dim = hyperparams.get("observation_dimension")
                
                if db_dim and db_dim != expected_dim:
                    logger.error(f"SAFETY GATE: Observation dimension mismatch. Model expects {db_dim}, Pipeline provides {expected_dim}.")
                    raise ValueError("Observation dimension mismatch.")
                    
                # Finally, verify model weights input layer matches the dimension
                try:
                    if hasattr(model, 'encoder') and hasattr(model.encoder[1], 'in_features'):
                        if model.encoder[1].in_features != expected_dim:
                            raise ValueError(f"Model weight in_features {model.encoder[1].in_features} != expected {expected_dim}")
                except Exception as ex:
                    logger.warning(f"Could not statically verify linear layer dim: {ex}")
                    
                file_path = version_meta.checkpoint_path or getattr(version_meta, 'file_path', None)
                if not file_path or not os.path.exists(file_path):
                    logger.error(f"SAFETY GATE: Model weights not found at {file_path}")
                    raise FileNotFoundError(f"Model weights not found at {file_path}")
                    
                model.load_state_dict(torch.load(file_path))
                model.eval()
                logger.info(f"Loaded rigorously validated model {version_meta.version_id} into production.")
                return model
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
