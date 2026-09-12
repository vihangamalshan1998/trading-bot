import pytest
import torch
from unittest.mock import patch, MagicMock
from core.ai.registry import ModelRegistry
from apps.research.model import MultiSymbolActorCritic
from apps.trading_bot.main import ProductionTradingBot
from core.schemas.dimension_config import get_expected_observation_dimension
from core.config.settings import settings

def test_production_bot_fails_without_active_model():
    # B. Production bot fails if no active model exists
    with patch("core.ai.registry.ModelRegistry.load_model") as mock_load:
        mock_load.side_effect = ValueError("No valid model version found")
        with pytest.raises(RuntimeError, match="NO PRODUCTION INFERENCE"):
            # This should crash during __init__ preventing a random model from taking over
            ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT"])

def test_production_bot_loads_active_model():
    # A. ProductionTradingBot loads ModelRegistry
    with patch("core.ai.registry.ModelRegistry.load_model") as mock_load:
        mock_load.return_value = MagicMock()
        bot = ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT"])
        assert mock_load.called
        assert bot.has_valid_model is True

def test_production_model_requires_active_checkpoint():
    registry = ModelRegistry()
    model = MultiSymbolActorCritic(num_symbols=len(settings.symbol_universe), macro_dim=8)
    
    # Patch the DB session to return None for active model
    with patch.object(registry, 'SessionLocal') as mock_session_maker:
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query().filter_by().first.return_value = None
        
        with pytest.raises(ValueError, match="No valid model version found"):
            registry.load_model(model)

def test_random_model_cannot_be_used_for_production():
    with patch("core.ai.registry.ModelRegistry.load_model") as mock_load:
        mock_load.side_effect = ValueError("Simulated DB Load Failure")
        with pytest.raises(RuntimeError, match="NO TRADING"):
            # If load_model fails, the bot MUST crash, not fallback to a random model.
            ProductionTradingBot(symbols=["BTCUSDT"])

def test_checkpoint_architecture_validation():
    registry = ModelRegistry()
    model = MultiSymbolActorCritic(num_symbols=len(settings.symbol_universe), macro_dim=8)
    
    with patch.object(registry, 'SessionLocal') as mock_session_maker:
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        
        # Return a model meta with WRONG architecture
        mock_meta = MagicMock()
        mock_meta.architecture = "WrongArchitectureRNN"
        mock_meta.hyperparameters = {}
        mock_session.query().filter_by().first.return_value = mock_meta
        
        with pytest.raises(ValueError, match="Architecture mismatch"):
            registry.load_model(model)

def test_checkpoint_symbol_order_validation():
    registry = ModelRegistry()
    model = MultiSymbolActorCritic(num_symbols=len(settings.symbol_universe), macro_dim=8)
    
    with patch.object(registry, 'SessionLocal') as mock_session_maker:
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        
        # Return a model meta with WRONG symbol universe
        mock_meta = MagicMock()
        mock_meta.architecture = "MultiSymbolActorCritic"
        mock_meta.hyperparameters = {"symbol_universe": ["DOGEUSDT"]} # Wrong
        mock_session.query().filter_by().first.return_value = mock_meta
        
        with pytest.raises(ValueError, match="Symbol universe mismatch"):
            registry.load_model(model)
