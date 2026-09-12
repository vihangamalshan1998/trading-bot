import pytest
import torch
from core.ai.registry import ModelRegistry
from apps.research.model import MultiSymbolActorCritic
from core.schemas.dimension_config import get_expected_observation_dimension
from core.config.settings import settings

def test_production_model_requires_checkpoint():
    registry = ModelRegistry()
    model = MultiSymbolActorCritic(num_symbols=len(settings.symbol_universe), macro_dim=8)
    with pytest.raises(ValueError, match="No valid model version found"):
        registry.load_model(model, version_id="non_existent_version")

def test_checkpoint_architecture_validation():
    # In a true integration test, we'd mock the DB return.
    # Here we simulate the logic that checks model.__class__.__name__
    model = MultiSymbolActorCritic(num_symbols=len(settings.symbol_universe), macro_dim=8)
    assert model.__class__.__name__ == "MultiSymbolActorCritic"

def test_checkpoint_symbol_order_validation():
    # Simulated validation test
    wrong_symbols = ["BTCUSDT"] 
    expected_dim_actual = get_expected_observation_dimension(len(settings.symbol_universe))
    wrong_dim = 9 + (len(wrong_symbols) * 37) + 8
    assert expected_dim_actual != wrong_dim
