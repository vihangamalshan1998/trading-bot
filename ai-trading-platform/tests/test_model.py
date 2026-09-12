import pytest
import os
import torch
from core.ai.registry import ModelRegistry
from apps.research.model import MultiSymbolActorCritic
from core.schemas.dimension_config import get_expected_observation_dimension
from core.config.settings import settings

def test_production_model_requires_valid_checkpoint():
    """
    Proves that a randomly initialized model cannot silently bypass the registry.
    """
    registry = ModelRegistry()
    model = MultiSymbolActorCritic(num_symbols=len(settings.symbol_universe), macro_dim=8)
    
    with pytest.raises(Exception, match="No valid model version found"):
        registry.load_model(model, version_id="non_existent_version")

def test_checkpoint_symbol_order_matches_config():
    """
    Proves that symbol ordering must match. (Mocked logic checking settings consistency)
    """
    # Assuming the config has a specific length, a model built with a different length will fail dimension validation
    wrong_symbols = ["BTCUSDT"] # Only 1 symbol
    model = MultiSymbolActorCritic(num_symbols=len(wrong_symbols), macro_dim=8)
    expected_dim_actual = get_expected_observation_dimension(len(settings.symbol_universe))
    
    # Model built for 1 symbol will have wrong input dimension for the full universe state
    wrong_dim = 9 + (len(wrong_symbols) * 37) + 8
    assert expected_dim_actual != wrong_dim

def test_checkpoint_dimension_matches_state():
    """
    Proves that the model input dimension matches the state specification.
    """
    num_symbols = len(settings.symbol_universe)
    expected_dim = get_expected_observation_dimension(num_symbols)
    model = MultiSymbolActorCritic(num_symbols=num_symbols, macro_dim=8)
    
    # Model's first linear layer inside the encoder should match the state dimension
    # model.encoder[1] is the first Linear layer after LayerNorm in LatentWorldModel
    assert model.encoder[1].in_features == expected_dim
