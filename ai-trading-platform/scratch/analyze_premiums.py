import json
from core.database.session import SessionLocal
from core.database.models.ai import Experience

def analyze_exchange_premiums():
    session = SessionLocal()
    # Get the last 1000 experiences
    experiences = session.query(Experience).order_by(Experience.id.desc()).limit(1000).all()
    
    okx_premiums = []
    bybit_premiums = []
    okx_leads = []
    
    for exp in experiences:
        if exp.market_state and isinstance(exp.market_state, list) and len(exp.market_state) > 0:
            last_step = exp.market_state[-1] # The most recent timestep of the 120-window
            if len(last_step) >= 143:
                okx_premiums.append(float(last_step[140]))
                bybit_premiums.append(float(last_step[141]))
                okx_leads.append(float(last_step[142]))
    
    session.close()
    
    if not okx_premiums:
        print("No valid data found or slots are empty/0.0")
        return
        
    avg_okx_prem = sum(okx_premiums) / len(okx_premiums)
    avg_bybit_prem = sum(bybit_premiums) / len(bybit_premiums)
    avg_lead = sum(okx_leads) / len(okx_leads)
    
    non_zero_leads = [x for x in okx_leads if x != 0.0]
    
    print(f"Total Analyzed: {len(okx_premiums)}")
    print(f"Average OKX Premium (Slot 140): {avg_okx_prem:.6f}")
    print(f"Average Bybit Premium (Slot 141): {avg_bybit_prem:.6f}")
    print(f"Average OKX Momentum Lead (Slot 142): {avg_lead:.6f}")
    print(f"How often did OKX lead/lag heavily? (Non-zero lead count): {len(non_zero_leads)} out of {len(okx_premiums)}")
    
    if len(non_zero_leads) > 0:
        print(f"Max OKX Lead Spike: {max(non_zero_leads)}")
        print(f"Max OKX Lag Drop: {min(non_zero_leads)}")

if __name__ == "__main__":
    analyze_exchange_premiums()
