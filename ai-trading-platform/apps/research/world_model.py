import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    """
    Recurrent State Space Model for Latent Dynamics.
    """
    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, state_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # RNN Cell (Memory)
        self.cell = nn.GRUCell(embed_dim + action_dim, hidden_dim)
        
        # Transition Model (Predicts next prior state)
        self.fc_prior_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_prior_2 = nn.Linear(hidden_dim, state_dim * 2) # mean, std
        
        # Posterior Model (Corrects prior with actual observation)
        self.fc_post_1 = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.fc_post_2 = nn.Linear(hidden_dim, state_dim * 2) # mean, std

    def forward_prior(self, embed: torch.Tensor, prev_action: torch.Tensor, prev_hidden: torch.Tensor):
        x = torch.cat([embed, prev_action], dim=-1)
        hidden = self.cell(x, prev_hidden)
        
        x = F.elu(self.fc_prior_1(hidden))
        stats = self.fc_prior_2(x)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1
        
        return hidden, mean, std
        
    def forward_posterior(self, hidden: torch.Tensor, embed: torch.Tensor):
        x = torch.cat([hidden, embed], dim=-1)
        x = F.elu(self.fc_post_1(x))
        stats = self.fc_post_2(x)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1
        
        return mean, std

class LatentWorldModel(nn.Module):
    """
    Phase 3: Upgraded World Model supporting complex normalized Portfolio/Position states,
    continuous (action, confidence, target_size) outputs, and 25-dim dense market features.
    """
    def __init__(self, num_symbols: int = 1, macro_dim: int = 8, 
                 embed_dim: int = 256, hidden_dim: int = 256, state_dim: int = 32):
        super().__init__()
        self.num_symbols = num_symbols
        self.macro_dim = macro_dim
        
        # New Action Space: 3 floats per symbol (action_type, confidence, target_size)
        self.actions_per_symbol = 3
        self.total_action_dim = num_symbols * self.actions_per_symbol
        
        # Phase 3 Obs Space: 9 (Portfolio) + N * (25 Market + 12 Position) + Macro
        obs_dim = 9 + (num_symbols * 37) + macro_dim
        
        # 1. Encoder (Observation -> Embedding)
        self.encoder = nn.Sequential(
            nn.LayerNorm(obs_dim), # Phase 4: Explicit Input Normalization
            nn.Linear(obs_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU()
        )
        
        # 2. RSSM (Latent Dynamics)
        self.rssM = RSSM(action_dim=self.total_action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim, state_dim=state_dim)
        
        # 3. Decoder (Latent -> Reconstructed Observation)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, obs_dim)
        )
        
        # 4. Reward Predictor (Latent -> Expected Reward)
        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
        # 5. Actor (Policy) - Now outputs continuous actions passed through Tanh [-1, 1]
        self.actor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.total_action_dim),
            nn.Tanh() # Constrain output to [-1, 1] for the Box space
        )
        
        # 6. Critic (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
    def sample_state(self, mean: torch.Tensor, std: torch.Tensor):
        if self.training:
            eps = torch.randn_like(mean)
            return mean + std * eps
        return mean

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor, prev_hidden: torch.Tensor):
        """
        Processes one timestep.
        obs: (batch, obs_dim)
        prev_action: (batch, total_action_dim)
        prev_hidden: (batch, hidden_dim)
        """
        batch_size = obs.size(0)
        
        # Phase 6: Strict Dimensional Asserts
        expected_obs_dim = 9 + (self.num_symbols * 37) + self.macro_dim
        assert obs.size(1) == expected_obs_dim, f"State dim mismatch: Expected {expected_obs_dim}, got {obs.size(1)}"
        assert prev_action.size(1) == self.total_action_dim, f"Action dim mismatch: Expected {self.total_action_dim}, got {prev_action.size(1)}"
        assert prev_hidden.size(1) == self.rssM.hidden_dim, f"Hidden dim mismatch: Expected {self.rssM.hidden_dim}, got {prev_hidden.size(1)}"

        # 1. Encode observation
        embed = self.encoder(obs)
        
        # 2. Compute Prior Dynamics
        hidden, prior_mean, prior_std = self.rssM.forward_prior(embed, prev_action, prev_hidden)
        
        # 3. Compute Posterior (Correction using actual observation)
        post_mean, post_std = self.rssM.forward_posterior(hidden, embed)
        
        # 4. Sample Latent State
        state = self.sample_state(post_mean, post_std)
        
        # 5. Concatenate memory and state for downstream tasks
        feat = torch.cat([state, hidden], dim=-1)
        
        # 6. Predictions
        reconstructed_obs = self.decoder(feat)
        pred_reward = self.reward_model(feat)
        
        # Continuous action output clamped to [-1, 1]
        actions = self.actor(feat) 
        value = self.critic(feat)
        
        # Reshape actions to (batch, num_symbols, actions_per_symbol)
        batch_size = obs.size(0)
        actions = actions.view(batch_size, self.num_symbols, self.actions_per_symbol)
        
        return {
            "hidden": hidden,
            "state": state,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "post_mean": post_mean,
            "post_std": post_std,
            "reconstructed_obs": reconstructed_obs,
            "pred_reward": pred_reward,
            "actions": actions, # Renamed from action_logits
            "value": value
        }
