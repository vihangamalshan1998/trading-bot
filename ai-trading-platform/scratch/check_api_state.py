import urllib.request
import json

try:
    req = urllib.request.urlopen("http://localhost:8000/api/state")
    data = json.loads(req.read().decode())
    symbols = list(data.get("market_states", {}).keys())
    print("API Symbols count:", len(symbols))
    print("API Symbols:", symbols)
    print("Portfolio Equity:", data.get("equity"))
    print("Positions:", data.get("positions"))
except Exception as e:
    print("Error:", e)
