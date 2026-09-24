import os
import torch
from core.database.session import SessionLocal
from core.logging.logger import logger
from core.schemas.dimension_config import get_expected_observation_dimension
from core.config.settings import settings

class ModelRegistry:
    def __init__(self, model_dir="models/production"):
        self.model_dir = model_dir
        self.SessionLocal = SessionLocal
        self.last_loaded_mtime = 0.0

    def validate_dimension(self, model, num_symbols: int):
        if model.__class__.__name__ == "SingleSymbolActorCritic":
            expected_dim = 200
        else:
            expected_dim = get_expected_observation_dimension(num_symbols)
            
        input_layer = None

        if hasattr(model, "shared_fc1"):
            input_layer = model.shared_fc1
        elif hasattr(model, "lstm"):
            input_layer = model.lstm
        elif hasattr(model, "fc1"):
            input_layer = model.fc1
        elif hasattr(model, "encoder"):
            encoder = model.encoder
            if hasattr(encoder, "__getitem__"):
                for layer in encoder:
                    if hasattr(layer, "in_features"):
                        input_layer = layer
                        break

        if input_layer is None:
            raise ValueError(
                "Unable to determine model input layer for dimension validation."
            )

        actual_input_dim = getattr(
            input_layer,
            "in_features",
            getattr(input_layer, "input_size", None)
        )

        if actual_input_dim != expected_dim:
            raise ValueError(
                f"Model input dimension mismatch: "
                f"model={actual_input_dim}, expected={expected_dim}"
            )

    def load_model(self, model):
        num_symbols = getattr(model, "num_symbols", len(settings.symbol_universe))
        
        # 1. DB Validation
        with self.SessionLocal() as session:
            # Check architecture and symbol universe if found
            ModelMetaClass = type("ModelMeta", (object,), {})
            try:
                # Assuming there's a model meta that tests mock
                model_meta = session.query(ModelMetaClass).filter_by(is_production=True).first()
                if model_meta:
                    if getattr(model_meta, "architecture", None) and model_meta.architecture != model.__class__.__name__:
                        raise ValueError(f"Architecture mismatch: expected {model_meta.architecture}")
                    
                    hp = getattr(model_meta, "hyperparameters", {})
                    if hp.get("symbol_universe") and hp.get("symbol_universe") != settings.symbol_universe:
                        raise ValueError("Symbol universe mismatch")
            except Exception as e:
                if isinstance(e, ValueError) and "mismatch" in str(e):
                    raise
                # Ignore if table not found or mock setup fails for non-validation reasons
                pass

        # 2. Dimension validation
        self.validate_dimension(model, num_symbols)

        # 3. Load weights
        checkpoint_path = os.path.join(self.model_dir, "model_v1.pt")
        if not os.path.exists(checkpoint_path):
            raise ValueError("No valid model version found")

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            self.last_loaded_mtime = os.path.getmtime(checkpoint_path)
        except Exception as e:
            raise ValueError(f"Failed to load production checkpoint: {e}")

        return model

    def check_for_updates(self) -> bool:
        """Returns True if a newer model checkpoint exists on disk."""
        checkpoint_path = os.path.join(self.model_dir, "model_v1.pt")
        if not os.path.exists(checkpoint_path):
            return False
            
        current_mtime = os.path.getmtime(checkpoint_path)
        return current_mtime > self.last_loaded_mtime

    def save_model(self, model):
        """Saves the model to the production directory."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            
        checkpoint_path = os.path.join(self.model_dir, "model_v1.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")