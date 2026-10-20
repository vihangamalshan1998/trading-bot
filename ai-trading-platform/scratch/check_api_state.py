import urllib.request, json

data = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/state").read())
states = data.get("market_states", {})
if "BTCUSDT" in states:
    v = dict(states["BTCUSDT"])
    if "features" in v:
        del v["features"]
    if "state_vector" in v:
        del v["state_vector"]
    print("BTCUSDT in API:")
    print(json.dumps(v, indent=2))
