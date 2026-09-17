import torch
import torch.nn as nn
from core.schemas.dimension_config import get_expected_observation_dimension

class WorldModel(nn.Module):
    """
    Phase 10: World Model (State-Transition Predictor).
    Given the current state (S_t) and action (A_t), predicts the next state (S_{t+1}) 
    and the expected reward (R_{t+1}).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(WorldModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Predict the delta (change) in the state rather than the absolute next state
        self.state_predictor = nn.Linear(hidden_dim, state_dim)
        
        # Predict the expected reward
        self.reward_predictor = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state: [batch_size, state_dim]
        action: [batch_size, action_dim]
        """
        x = torch.cat([state, action], dim=-1)
        encoded = self.encoder(x)
        
        state_delta = self.state_predictor(encoded)
        next_state_pred = state + state_delta # ResNet style
        
        reward_pred = self.reward_predictor(encoded)
        
        return next_state_pred, reward_pred
