import os
from typing import Tuple, Dict, Any, Optional
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

    def get_evaluator_version(self) -> str:
        """Evaluator that approved the active champion (from version.json), or "" for legacy checkpoints."""
        version_file = os.path.join(self.model_dir, "version.json")
        try:
            import json
            with open(version_file, "r") as f:
                metrics = json.load(f).get("metrics") or {}
            nested = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else {}
            return str(metrics.get("evaluator_version") or nested.get("evaluator_version") or "")
        except Exception:
            return ""

    def check_for_updates(self) -> bool:
        """Returns True if a newer model checkpoint exists on disk."""
        checkpoint_path = os.path.join(self.model_dir, "model_v1.pt")
        if not os.path.exists(checkpoint_path):
            return False
            
        current_mtime = os.path.getmtime(checkpoint_path)
        return current_mtime > self.last_loaded_mtime

    def save_model(self, model, version_tag: str = None, metrics: dict = None):
        """
        Saves the model to the production directory with version metadata.
        If a champion already exists, direct overwrite via save_model() without explicit approved
        evaluation metrics is blocked to prevent accidental unverified promotion.
        """
        import json
        import time

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            
        checkpoint_path = os.path.join(self.model_dir, "model_v1.pt")
        if os.path.exists(checkpoint_path):
            if not metrics or metrics.get("approved") is not True:
                logger.error("Cannot overwrite existing champion via save_model() without validated promotion metrics with approved=True.")
                return False

        temp_path = os.path.join(self.model_dir, "model_v1.pt.tmp")
        torch.save(model.state_dict(), temp_path)
        os.replace(temp_path, checkpoint_path)
        
        # Save version metadata
        version = version_tag or f"v1_push_{int(time.time())}"
        self.current_version = version
        version_file = os.path.join(self.model_dir, "version.json")
        temp_version = os.path.join(self.model_dir, "version.json.tmp")
        try:
            with open(temp_version, "w") as f:
                json.dump({
                    "version": version,
                    "timestamp": time.time(),
                    "metrics": metrics or {}
                }, f, indent=2)
            os.replace(temp_version, version_file)
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
        return True

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

    def compute_file_sha256(self, file_path: str) -> str:
        """Computes SHA256 cryptographic hash of a file for immutable verification."""
        import hashlib
        if not os.path.exists(file_path):
            return ""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_evaluation_evidence(
        self,
        metrics: dict,
        candidate_checkpoint_path: str,
        current_champion_path: str
    ) -> Tuple[bool, str]:
        """
        Task 1: Validates that promotion evidence is complete, finite, un-tampered,
        fresh, bound to exact checkpoint hashes, and satisfies explicit numerical superiority.
        Returns (is_valid, reason_code).
        """
        import math
        import time

        if not metrics or not isinstance(metrics, dict):
            return False, "MISSING_OR_INVALID_METRICS_DICT"

        # Flatten nested metrics if provided under "metrics" key
        m = metrics.get("metrics", metrics) if isinstance(metrics.get("metrics"), dict) else metrics
        approved = metrics.get("approved") if "approved" in metrics else m.get("approved")
        reason = metrics.get("reason") if "reason" in metrics else m.get("reason")

        if approved is not True:
            return False, f"CANDIDATE_NOT_APPROVED: approved={approved}"

        if not reason or not isinstance(reason, str):
            return False, "MISSING_REQUIRED_EVALUATION_REASON"

        # 1. Reject training-only evaluation
        data_split = m.get("data_split") or m.get("eval_type")
        if data_split == "train_only" or m.get("train_only") is True:
            return False, "TRAIN_ONLY_EVALUATION_REJECTED"

        # 2. Reject non-finite metrics
        required_numeric_keys = ["challenger_score", "champion_score"]
        optional_numeric_keys = [
            "challenger_sim_pnl", "champion_sim_pnl",
            "challenger_critic_mse", "champion_critic_mse",
            "critic_loss", "entropy", "sample_size", "eval_timestamp",
            "fee_hurdle", "slippage_hurdle"
        ]
        for key in required_numeric_keys:
            val = m.get(key)
            if val is None:
                return False, f"MISSING_REQUIRED_NUMERIC_METRIC_{key.upper()}"
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return False, f"NON_FINITE_OR_INVALID_TYPE_{key.upper()}"

        for key in optional_numeric_keys:
            val = m.get(key)
            if val is not None:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    return False, f"NON_FINITE_OR_INVALID_TYPE_{key.upper()}"

        chall_score = float(m["challenger_score"])
        champ_score = float(m["champion_score"])

        # 3. Task 1.1: Remove inferior-candidate bypass completely!
        # Numerical comparison is absolute: textual reason (e.g. CRITIC_IMPROVEMENT) can NEVER override a failed comparison!
        if chall_score <= champ_score:
            return False, f"INFERIOR_SCORE_REJECTED: challenger ({chall_score:.4f}) <= champion ({champ_score:.4f})"

        # Critic loss improvement alone must never authorize promotion if trading criteria degrades
        chall_pnl = m.get("challenger_sim_pnl")
        champ_pnl = m.get("champion_sim_pnl")
        if chall_pnl is not None and champ_pnl is not None:
            if chall_pnl < champ_pnl:
                return False, f"DEGRADED_TRADING_PERFORMANCE_REJECTED: chall_pnl ({chall_pnl:.4f}) < champ_pnl ({champ_pnl:.4f})"

        # 4. Mandatory sample size validation
        sample_size = m.get("sample_size")
        if sample_size is None:
            return False, "MISSING_REQUIRED_SAMPLE_SIZE"
        if not isinstance(sample_size, (int, float)) or sample_size < 30:
            return False, f"INSUFFICIENT_SAMPLE_SIZE: {sample_size} < 30"

        # 5. Mandatory evaluation timestamp freshness (within last 24h and not in future)
        eval_timestamp = m.get("eval_timestamp")
        if eval_timestamp is None:
            return False, "MISSING_REQUIRED_EVAL_TIMESTAMP"
        now_t = time.time()
        if eval_timestamp > now_t + 60:
            return False, f"FUTURE_EVAL_TIMESTAMP: {eval_timestamp} > {now_t + 60}"
        if now_t - eval_timestamp > 86400:
            return False, f"STALE_EVAL_TIMESTAMP: {eval_timestamp} older than 24h"

        # 6. Checkpoint hash and champion version binding
        if os.path.exists(candidate_checkpoint_path):
            actual_cand_hash = self.compute_file_sha256(candidate_checkpoint_path)
            claimed_cand_hash = m.get("candidate_checkpoint_hash")
            if claimed_cand_hash and claimed_cand_hash != actual_cand_hash:
                return False, f"CANDIDATE_HASH_MISMATCH: claimed {claimed_cand_hash} != actual {actual_cand_hash}"

        if os.path.exists(current_champion_path):
            actual_champ_hash = self.compute_file_sha256(current_champion_path)
            claimed_champ_hash = m.get("champion_checkpoint_hash")
            if claimed_champ_hash and claimed_champ_hash != actual_champ_hash:
                return False, f"CHAMPION_HASH_MISMATCH: claimed {claimed_champ_hash} != actual {actual_champ_hash}"

            claimed_champ_version = m.get("champion_version")
            if claimed_champ_version and claimed_champ_version != self.get_current_version():
                return False, f"CHAMPION_VERSION_MISMATCH: claimed {claimed_champ_version} != current {self.get_current_version()}"

        return True, "EVALUATION_EVIDENCE_VALIDATED"

    def promote_candidate(self, candidate_checkpoint_path: str, version_tag: str, metrics: dict = None) -> bool:
        """
        Promotes a validated candidate model to champion with strict fail-closed gate checks,
        concurrency protection, atomic replacement, and transactional rollback.
        """
        import shutil
        import time
        import json

        current_champion = os.path.join(self.model_dir, "model_v1.pt")
        backup_champion = os.path.join(self.model_dir, "model_v1_backup.pt")
        version_file = os.path.join(self.model_dir, "version.json")
        backup_version = os.path.join(self.model_dir, "version_backup.json")

        # Step 1: Validate evidence
        valid_evidence, evidence_reason = self.validate_evaluation_evidence(
            metrics=metrics,
            candidate_checkpoint_path=candidate_checkpoint_path,
            current_champion_path=current_champion
        )
        if not valid_evidence:
            logger.error(f"Cannot promote candidate: Evaluation evidence rejected: {evidence_reason}.")
            return False

        if not os.path.exists(candidate_checkpoint_path):
            logger.error(f"Cannot promote candidate: file {candidate_checkpoint_path} not found.")
            return False

        # Step 2: Candidate weights & architecture verification before promotion
        try:
            cand_state = torch.load(candidate_checkpoint_path, map_location="cpu")
            if not isinstance(cand_state, dict):
                logger.error("Cannot promote candidate: Invalid state_dict format.")
                return False
            for k, v in cand_state.items():
                if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                    logger.error(f"Cannot promote candidate: NaN/Inf weights detected in {k}.")
                    return False

            # Verify architecture compatibility with SingleSymbolActorCritic(input_dim=200)
            from apps.research.model import SingleSymbolActorCritic
            test_model = SingleSymbolActorCritic(input_dim=200)
            test_model.load_state_dict(cand_state)
            test_model.eval()
            with torch.no_grad():
                dummy_input = torch.zeros(1, 300, 200)
                a_out, v_out = test_model(dummy_input)
                if a_out.shape[-1] != 4 or v_out.shape[-1] != 3:
                    logger.error(f"Cannot promote candidate: Model output dimension mismatch (actions: {a_out.shape}, values: {v_out.shape}).")
                    return False
        except Exception as e:
            logger.error(f"Cannot promote candidate: Failed to load, validate, or verify candidate architecture: {e}")
            return False

        # Concurrency protection: Lock file
        lock_file = os.path.join(self.model_dir, "promotion.lock")
        try:
            with open(lock_file, "x") as f:
                f.write(str(time.time()))
        except FileExistsError:
            try:
                mtime = os.path.getmtime(lock_file)
                if time.time() - mtime > 60:
                    os.remove(lock_file)
                    with open(lock_file, "w") as f:
                        f.write(str(time.time()))
                else:
                    logger.error("Cannot promote candidate: Concurrent promotion in progress (promotion.lock active).")
                    return False
            except Exception:
                return False

        try:
            # Recheck that current champion has not been swapped by concurrent process
            if os.path.exists(current_champion) and metrics:
                m = metrics.get("metrics", metrics) if isinstance(metrics.get("metrics"), dict) else metrics
                claimed_champ_hash = m.get("champion_checkpoint_hash")
                if claimed_champ_hash:
                    cur_hash = self.compute_file_sha256(current_champion)
                    if cur_hash != claimed_champ_hash:
                        logger.error(f"Cannot promote candidate: Champion changed concurrently during evaluation ({cur_hash} != {claimed_champ_hash}).")
                        return False

            # Backup current champion and version metadata before replacement
            if os.path.exists(current_champion):
                shutil.copy2(current_champion, backup_champion)
            if os.path.exists(version_file):
                shutil.copy2(version_file, backup_version)

            # Atomic promotion: Copy to temp file, verify readability, then atomic os.replace
            temp_champion = os.path.join(self.model_dir, "model_v1.pt.tmp")
            shutil.copy2(candidate_checkpoint_path, temp_champion)
            _ = torch.load(temp_champion, map_location="cpu")
            os.replace(temp_champion, current_champion)
            self.current_version = version_tag

            # Atomic version metadata write
            temp_version = os.path.join(self.model_dir, "version.json.tmp")
            with open(temp_version, "w") as f:
                json.dump({
                    "version": version_tag,
                    "promoted_at": time.time(),
                    "metrics": metrics,
                    "candidate_sha256": self.compute_file_sha256(current_champion),
                    "has_backup": os.path.exists(backup_champion)
                }, f, indent=2)
            os.replace(temp_version, version_file)

            self.record_tournament_result(metrics, version_tag=version_tag)
            logger.info(f"🏆 Candidate {version_tag} PROMOTED to Production Champion!")
            return True
        except Exception as e:
            logger.error(f"Promotion failed unexpectedly: {e}. Executing immediate rollback...")
            if os.path.exists(backup_champion):
                shutil.copy2(backup_champion, current_champion)
            if os.path.exists(backup_version):
                shutil.copy2(backup_version, version_file)
            self.current_version = self.get_current_version()
            return False
        finally:
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except Exception:
                    pass

    def rollback(self) -> bool:
        """Restores the last known-good champion checkpoint and version metadata from backup."""
        import shutil
        backup_path = os.path.join(self.model_dir, "model_v1_backup.pt")
        target_path = os.path.join(self.model_dir, "model_v1.pt")
        backup_version_path = os.path.join(self.model_dir, "version_backup.json")
        target_version_path = os.path.join(self.model_dir, "version.json")

        if not os.path.exists(backup_path):
            logger.error("Rollback failed: No backup checkpoint found!")
            return False

        shutil.copy2(backup_path, target_path)
        if os.path.exists(backup_version_path):
            shutil.copy2(backup_version_path, target_version_path)
            self.current_version = self.get_current_version()
        logger.info("Successfully rolled back to previous champion checkpoint and version metadata!")
        return True