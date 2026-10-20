import sys, torch, json
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry

def test():
    model = SingleSymbolActorCritic(input_dim=200)
    registry = ModelRegistry()
    model = registry.load_model(model)
    model.eval()

    # Target:
    # confidence: 0.7306179404258728
    # target_size: 0.26900315284729004
    # critic_return: -0.035249...
    # critic_1h: -0.065199...
    # critic_4h: 0.030755...

    # Test 1: All zeros for 300 steps
    z300 = torch.zeros(1, 300, 200)
    with torch.no_grad():
        a1, c1 = model(z300)
    print("Test 1 (zeros 300):")
    print(f"  act: {a1[0].numpy()}")
    print(f"  critic: {c1[0].numpy()}")
    print(f"  conf: {1.0 / (1.0 + torch.exp(-a1[0][1])):.6f}")

    # Test 2: All zeros for 1 step
    z1 = torch.zeros(1, 1, 200)
    with torch.no_grad():
        a2, c2 = model(z1)
    print("Test 2 (zeros 1):")
    print(f"  act: {a2[0].numpy()}")
    print(f"  critic: {c2[0].numpy()}")
    print(f"  conf: {1.0 / (1.0 + torch.exp(-a2[0][1])):.6f}")

    # Test 3: What if actor_out is literally tanh(actor_fc(zeros))?
    with torch.no_grad():
        # What is actor_fc(zeros_256)?
        zeros_h = torch.zeros(1, 256)
        a_bias = torch.tanh(model.actor_fc(zeros_h))
        c_bias = model.critic_fc(zeros_h)
    print("Test 3 (actor_fc & critic_fc on zeros hidden):")
    print(f"  act: {a_bias[0].numpy()}")
    print(f"  critic: {c_bias[0].numpy()}")
    print(f"  conf: {1.0 / (1.0 + torch.exp(-a_bias[0][1])):.6f}")

    # Test 4: What if lstm is fed nan_to_num on empty/nan?
    # Test 5: Check different sequence lengths of zeros
    for seq_len in [1, 2, 5, 10, 30, 50, 100, 300]:
        z = torch.zeros(1, seq_len, 200)
        with torch.no_grad():
            a, c = model(z)
        print(f"Seq len {seq_len}: critic={c[0].numpy()}, a[1]={a[0][1].item():.4f}")

if __name__ == '__main__':
    test()
