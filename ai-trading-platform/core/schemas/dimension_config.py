import torch
import math
from core.logging.logger import logger
from core.config.settings import settings

# Canonical State Dimensions
DIM_PORTFOLIO = 9
DIM_MARKET = 25
DIM_POSITION = 12
DIM_MACRO = 8 # (sentiment, vol, regime, + 5 padded)
DIM_EVENT = 5

def get_expected_observation_dimension(num_symbols: int) -> int:
    """Calculates the exact required dimension for the State Vector."""
    return DIM_PORTFOLIO + (num_symbols * (DIM_MARKET + DIM_POSITION)) + DIM_MACRO

def validate_observation(observation: torch.Tensor, expected_dim: int, symbols: list):
    """
    Phase 1: Strict state-dimension validation.
    Rejects wrong shape, NaN, Inf, and ensures symbol list matches.
    """
    if not isinstance(observation, torch.Tensor):
        raise TypeError("Observation must be a torch.Tensor")
        
    if len(observation.shape) != 2:
        raise ValueError(f"Observation must be 2D (batch_size, dim), got {len(observation.shape)}D")
        
    actual_dim = observation.size(1)
    if actual_dim != expected_dim:
        logger.error(f"STATE DIMENSION MISMATCH: Expected {expected_dim}, got {actual_dim}")
        raise ValueError(f"STATE DIMENSION MISMATCH: Expected {expected_dim}, got {actual_dim}")
        
    if torch.isnan(observation).any():
        logger.error("STATE VALIDATION FAILED: NaN detected in observation tensor.")
        raise ValueError("STATE VALIDATION FAILED: NaN detected.")
        
    if torch.isinf(observation).any():
        logger.error("STATE VALIDATION FAILED: Inf detected in observation tensor.")
        raise ValueError("STATE VALIDATION FAILED: Inf detected.")
        
    # Validate Canonical Symbol Ordering matches the provided lists
    expected_symbols = settings.symbol_universe
    if symbols != expected_symbols:
        logger.error(f"SYMBOL ORDER MISMATCH: Expected {expected_symbols}, got {symbols}")
        raise ValueError(f"SYMBOL ORDER MISMATCH: Expected {expected_symbols}, got {symbols}")
        
    return True
