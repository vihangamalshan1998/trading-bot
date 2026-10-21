import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingNet(nn.Module):
    """
    A foundational Multi-Layer Perceptron (MLP) for predicting Q-values or Policy probabilities.
    Maps a 9-dimensional futures market state to a 5-dimensional action space.
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, output_dim: int = 5):
        super(TradingNet, self).__init__()
        
        # Simple feed-forward architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Optional: BatchNorm to stabilize features with varying scales
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, input_dim)
        Returns:
            torch.Tensor: Tensor of shape (batch_size, output_dim)
        """
        # Note: If batch size is 1, BatchNorm might throw an error during training
        # We handle this by conditionally applying it or ensuring batch_size > 1
        
        out = self.fc1(x)
        if out.size(0) > 1:
            out = self.bn1(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        if out.size(0) > 1:
            out = self.bn2(out)
        out = F.relu(out)
        
        # Raw logits or Q-values
        out = self.fc3(out)
        
        return out

class MultiSymbolTradingNet(nn.Module):
    """
    Phase 3: Upgraded neural network that ingests normalized Portfolio and Position states,
    25-dim dense market features, and outputs continuous (action, confidence, target_size) vectors.
    """
    def __init__(self, num_symbols: int = 1, macro_dim: int = 8, hidden_dim: int = 128):
        super(MultiSymbolTradingNet, self).__init__()
        self.num_symbols = num_symbols
        self.actions_per_symbol = 3 # (action_type, confidence, target_size)
        
        # Phase 3 Total input: 9 (Portfolio) + N * (25 Market + 12 Position) + Macro
        input_dim = 9 + (num_symbols * 37) + macro_dim
        output_dim = num_symbols * self.actions_per_symbol
        
        # Phase 4: Explicit Input Normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, input_dim)
        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_symbols, actions_per_symbol)
        """
        # Ensure inputs are perfectly normalized before the MLP
        x = self.input_norm(x)
        
        out = self.fc1(x)
        if out.size(0) > 1:
            out = self.bn1(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        if out.size(0) > 1:
            out = self.bn2(out)
        out = F.relu(out)
        
        # Continuous outputs bound to [-1, 1]
        action_logits = torch.tanh(self.fc3(out))
        
        # Reshape so each symbol has its own action vector
        # shape: (batch_size, num_symbols, actions_per_symbol)
        action_logits = action_logits.view(x.size(0), self.num_symbols, self.actions_per_symbol)
        
        return action_logits

class MultiSymbolActorCritic(nn.Module):
    """
    Phase 8: Multi-symbol shared AI policy.
    Implements a shared encoder with separate Actor and Critic heads.
    - Actor outputs (action, confidence, size) per symbol.
    - Critic outputs a single scalar value representing the total portfolio expected return.
    """
    def __init__(self, num_symbols: int = 1, macro_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.num_symbols = num_symbols
        self.actions_per_symbol = 3
        
        input_dim = 9 + (num_symbols * 37) + macro_dim
        
        # Shared feature extractor
        self.shared_norm = nn.LayerNorm(input_dim)
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor Head
        self.actor_fc = nn.Linear(hidden_dim, num_symbols * self.actions_per_symbol)
        
        # Critic Head (Outputs a single global value for the entire portfolio)
        self.critic_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor):
        # Forward pass returning both action logits and state value
        x = self.shared_norm(x)
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        # Actor: (batch_size, num_symbols * 3) -> bound to [-1, 1]
        actor_out = torch.tanh(self.actor_fc(x))
        actor_out = actor_out.view(x.size(0), self.num_symbols, self.actions_per_symbol)
        
        # Critic: (batch_size, 1)
        critic_out = self.critic_fc(x)
        
        return actor_out, critic_out

class SingleSymbolActorCritic(nn.Module):
    """
    Phase 2 Upgrade: ActorCritic for a single symbol state with LSTM Memory and Limit Orders.
    Input dims: 200 (God Mode Architecture with 70 blank padding slots)
    """
    FEATURE_BUFFERS = ("feat_mean", "feat_std", "feat_mask", "feat_fitted")

    def __init__(self, input_dim: int = 200, hidden_dim: int = 256):
        super().__init__()

        # Per-slot input standardization, saved inside the checkpoint so preprocessing always travels
        # with the weights. Identity by default: checkpoints saved before this existed load unchanged.
        # LayerNorm across all 200 slots alone let large slots (order-book qty ~1, ratios ~2) drown the
        # small but important price signals (returns ~0.001); standardizing each slot first fixes that.
        # feat_mask zeroes RESERVED / DEPRECATED / dead slots so future canvas slots cannot inject noise.
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        self.register_buffer("feat_mask", torch.ones(input_dim))
        self.register_buffer("feat_fitted", torch.zeros(()))

        self.shared_norm = nn.LayerNorm(input_dim)
        
        # Replace Linear layers with an LSTM core
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Actor Head: action, confidence, size, price_offset (4 outputs)
        self.actor_fc = nn.Linear(hidden_dim, 4)
        
        # Critic Head (Now predicts 3 horizons: 5m, 1h, 4h)
        self.critic_fc = nn.Linear(hidden_dim, 3)
        
    def forward(self, x: torch.Tensor, hidden_state=None, return_hidden=False, temperature: float = 6.0):
        # Support both (batch_size, input_dim) and (batch_size, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch_size, 1, input_dim)

        if bool(self.feat_fitted):
            x = torch.clamp((x - self.feat_mean) / self.feat_std, -5.0, 5.0) * self.feat_mask

        x = self.shared_norm(x)
        
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        # Take the output from the last time step in the sequence
        last_out = lstm_out[:, -1, :]
        
        actor_raw = self.actor_fc(last_out)
        if temperature is not None and temperature > 1.0:
            # Direction (index 0): standard tanh for strong directional conviction [-1.0, 1.0]
            # Confidence (1), Size (2), Offset (3): temperature-scaled tanh prevents saturated clipping
            # to +/-0.9999, ensuring dynamic, continuous sensitivity across all market regimes
            act_dir = torch.tanh(actor_raw[:, 0:1])
            act_rest = torch.tanh(actor_raw[:, 1:] / temperature)
            actor_out = torch.cat([act_dir, act_rest], dim=-1)
        else:
            actor_out = torch.tanh(actor_raw)
            
        critic_out = self.critic_fc(last_out)
        
        if return_hidden:
            return actor_out, critic_out, new_hidden
        return actor_out, critic_out

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            self.feat_mean.copy_(mean)
            self.feat_std.copy_(torch.clamp(std, min=1e-6))
            self.feat_mask.copy_(mask)
            self.feat_fitted.fill_(1.0)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Checkpoints from before per-slot standardization have no feature buffers: load them as identity.
        state_dict = dict(state_dict)
        identity = {
            "feat_mean": torch.zeros_like(self.feat_mean),
            "feat_std": torch.ones_like(self.feat_std),
            "feat_mask": torch.ones_like(self.feat_mask),
            "feat_fitted": torch.zeros_like(self.feat_fitted),
        }
        for name in self.FEATURE_BUFFERS:
            state_dict.setdefault(name, identity[name])
        return super().load_state_dict(state_dict, *args, **kwargs)
