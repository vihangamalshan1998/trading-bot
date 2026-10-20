import torch, sys
sys.path.insert(0, '.')
from apps.trainer.ppo import PPOTrainer
from apps.research.model import SingleSymbolActorCritic

def test():
    model = SingleSymbolActorCritic(200)
    trainer = PPOTrainer(model=model, lr=1e-4)
    
    # Create dummy tensors matching replay_buffer.build_tensors output
    # (batch_size=16, seq_len=300, input_dim=200)
    states = torch.randn(16, 300, 200)
    # actions: [dir, conf, size, offset]
    actions = torch.randn(16, 4).clamp(-1.0, 1.0)
    # rewards: [5m, 1h, 4h]
    rewards = torch.randn(16, 3)
    next_states = torch.randn(16, 300, 200)
    dones = torch.zeros(16, 1)
    
    tensors = (states, actions, rewards, next_states, dones)
    
    # Run 1 epoch of train_step
    print("Testing train_step...")
    metrics = trainer.train_step(batch_size=16, epochs=1, tensors=tensors)
    print("Train step completed successfully!")
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})

if __name__ == '__main__':
    test()
