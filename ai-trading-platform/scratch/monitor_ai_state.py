import redis, time, json

r = redis.Redis()
initial_val = r.get("ai:state:BTCUSDT")
print("Initial ai:state:BTCUSDT:", initial_val.decode() if initial_val else None)

p = r.pubsub()
p.psubscribe("ai:state:*")
print("Listening for updates on ai:state:* for 10 seconds...")
start = time.time()
updated = False
while time.time() - start < 10:
    msg = p.get_message(timeout=1.0)
    if msg and msg['type'] == 'pmessage':
        print(f"Received update on {msg['channel'].decode()}: {msg['data'].decode()[:100]}")
        updated = True

if not updated:
    print("NO PUBSUB MESSAGES RECEIVED!")

# Also check if the key changed in Redis directly
time.sleep(2)
new_val = r.get("ai:state:BTCUSDT")
print("\nAfter 12s, ai:state:BTCUSDT is:", new_val.decode() if new_val else None)
print("Did it change?", new_val != initial_val)
