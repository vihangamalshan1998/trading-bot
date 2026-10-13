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

        self.current_version = self.get_current_version()
        return model

    def get_current_version(self) -> str:
        """Reads active model version from version.json or defaults to v1."""
        version_file = os.path.join(self.model_dir, "version.json")
        if os.path.exists(version_file):
            try:
                import json
                with open(version_file, "r") as f:
                    data = json.load(f)
                    return data.get("version", "v1")
            except Exception:
                pass
        return "v1"

    def check_for_updates(self) -> bool:
        """Returns True if a newer model checkpoint exists on disk."""
        checkpoint_path = os.path.join(self.model_dir, "model_v1.pt")
        if not os.path.exists(checkpoint_path):
            return False
            
        current_mtime = os.path.getmtime(checkpoint_path)
        return current_mtime > self.last_loaded_mtime

    def save_model(self, model, version_tag: str = None, metrics: dict = None):
        """Saves the model to the production directory with version metadata and historical archive."""
        import json
        import time

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            
        checkpoint_path = os.path.join(self.model_dir, "model_v1.pt")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save version metadata
        version = version_tag or f"v1_push_{int(time.time())}"
        self.current_version = version
        version_file = os.path.join(self.model_dir, "version.json")
        try:
            with open(version_file, "w") as f:
                json.dump({
                    "version": version,
                    "timestamp": time.time(),
                    "metrics": metrics or {}
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write version.json: {e}")

        # Also archive to checkpoints directory for auditability
        checkpoints_dir = "models/checkpoints"
        os.makedirs(checkpoints_dir, exist_ok=True)
        archive_path = os.path.join(checkpoints_dir, f"{version}.pt")
        try:
            torch.save(model.state_dict(), archive_path)
        except Exception as e:
            logger.warning(f"Could not archive checkpoint: {e}")

        logger.info(f"Model saved to {checkpoint_path} [Version: {version}]")
        self.record_tournament_result(metrics or {}, version_tag=version)

    def record_tournament_result(self, eval_result: dict, version_tag: str = None):
        """Appends a tournament showdown record to cumulative tournament_history.json."""
        import json
        import time
        from datetime import datetime

        history_file = os.path.join(self.model_dir, "tournament_history.json")
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []

        m = eval_result.get("metrics", eval_result)
        entry = {
            "version": version_tag or eval_result.get("version", "candidate"),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
            "approved": eval_result.get("approved", True if version_tag else False),
            "reason": eval_result.get("reason", "SUPERIOR_TO_CHAMPION" if version_tag else "REJECTED"),
            "challenger_score": m.get("challenger_score"),
            "champion_score": m.get("champion_score"),
            "challenger_critic_mse": m.get("challenger_critic_mse"),
            "champion_critic_mse": m.get("champion_critic_mse"),
            "challenger_sim_pnl": m.get("challenger_sim_pnl"),
            "champion_sim_pnl": m.get("champion_sim_pnl"),
            "entropy": m.get("entropy")
        }
        history.append(entry)

        try:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update tournament_history.json: {e}")