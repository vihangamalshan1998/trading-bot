import sys, runpy, importlib.util as u
spec = u.spec_from_file_location("apps.trading_bot.main", "/tmp/proof/apps/trading_bot/main.py")
m = u.module_from_spec(spec)
sys.modules["apps.trading_bot.main"] = m
spec.loader.exec_module(m)
runpy.run_path("/tmp/proof/scratch/prove_tvl_collapse.py")
