import sys, torch
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic

def test_backup():
    m = SingleSymbolActorCritic(input_dim=200)
    m.load_state_dict(torch.load("/var/www/trading-bot/ai-trading-platform/models/production/model_v1_backup.pt", map_location="cpu"))
    m.eval()
    
    z = torch.zeros(1, 300, 200)
    with torch.no_grad():
        a, c = m(z)
    print("BACKUP MODEL ON ZEROS (300):")
    print("  act:", a.numpy())
    print("  critic:", c.numpy())
    print("  conf:", 1.0 / (1.0 + torch.exp(-a[0][1]).item()))
    print("  size:", 1.0 / (1.0 + torch.exp(-a[0][2]).item()))

    # With portfolio = 5000:
    obs = [5000.0, 5000.0, 5000.0] + [0.0] * 197
    t5000 = torch.tensor([[obs] * 300], dtype=torch.float32)
    with torch.no_grad():
        a5, c5 = m(t5000)
    print("BACKUP MODEL ON 5000 PORTFOLIO:")
    print("  act:", a5.numpy())
    print("  critic:", c5.numpy())
    print("  conf:", 1.0 / (1.0 + torch.exp(-a5[0][1]).item()))
    print("  size:", 1.0 / (1.0 + torch.exp(-a5[0][2]).item()))

if __name__ == '__main__':
    test_backup()
