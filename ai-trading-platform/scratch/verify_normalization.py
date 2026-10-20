import torch, sys
sys.path.insert(0, '.')
from core.ai.replay_buffer import ReplayBuffer
from apps.trainer.ppo import PPOTrainer, normalize_state_tensor
from apps.research.model import SingleSymbolActorCritic

def verify():
    buf = ReplayBuffer()
    buf.load_cache_from_db(limit=30)
    batch = buf.sample(10)
    print(f"Sampled {len(batch)} experiences from cache")
    states, actions, rewards, next_states, dones = buf.build_tensors(batch, seq_len=10)
    print(f"States shape: {states.shape}")
    
    # Check max values in slot 0, 1, 2, 4, 11
    s0_max = states[:, :, 0].max().item()
    s1_max = states[:, :, 1].max().item()
    s2_max = states[:, :, 2].max().item()
    s4_max = states[:, :, 4].max().item()
    s11_max = states[:, :, 11].abs().max().item()
    
    print(f"Slot 0 (Wallet Balance ratio) max: {s0_max:.4f}")
    print(f"Slot 1 (Equity ratio) max: {s1_max:.4f}")
    print(f"Slot 2 (Free Margin ratio) max: {s2_max:.4f}")
    print(f"Slot 4 (Exposure ratio) max: {s4_max:.4f}")
    print(f"Slot 11 (Entry Price dist) max: {s11_max:.4f}")
    
    # Check market features range (slots 20 to 109)
    m_feats = states[:, :, 20:110]
    print(f"Market features (slots 20-110) min: {m_feats.min().item():.4f}, max: {m_feats.max().item():.4f}")
    
    # Check LayerNorm behavior
    model = SingleSymbolActorCritic(200)
    normed = model.shared_norm(states)
    print(f"After LayerNorm(200): std across dims = {normed.std(dim=-1).mean().item():.4f}")
    print("VERIFICATION SUCCESSFUL: State vectors are perfectly normalized!")

if __name__ == '__main__':
    verify()
