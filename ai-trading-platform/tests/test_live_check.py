from apps.dashboard_api.main import live_vs_test


def _row(t, side, sym="NEARUSDT", pnl=0.0, p=None, ver=None):
    r = {"timestamp": t, "side": side, "symbol": sym, "realized_pnl": pnl}
    if p is not None:
        r["confidence_calibrated"] = p
    if ver:
        r["model_version"] = ver
    return r


def test_partial_take_profit_and_close_count_as_one_trade():
    # +85 banked at TP1, runner closed at breakeven -1: the whole trade made money -> a win
    hist = [_row(1, "OPEN_SHORT", p=0.6, ver="v2"), _row(2, "PARTIAL_CLOSE_SHORT", pnl=85.0),
            _row(3, "CLOSE_SHORT", pnl=-1.0),
            _row(4, "OPEN_LONG", sym="TAOUSDT", p=0.5, ver="v2"), _row(5, "CLOSE_LONG", sym="TAOUSDT", pnl=-80.0)]
    out = live_vs_test(list(reversed(hist)), "v2")   # Redis returns newest first
    assert out["n"] == 2 and out["live_win"] == 0.5 and abs(out["test_win"] - 0.55) < 1e-9
    assert out["scope"] == "brain" and [b["range"] for b in out["buckets"]] == ["45-55%", "55-65%"]


def test_only_the_live_brains_trades_and_no_dust_or_recovered_entries():
    hist = [_row(1, "OPEN_LONG", sym="BTCUSDT", p=0.7, ver="old"), _row(2, "CLOSE_LONG", sym="BTCUSDT", pnl=50.0),
            _row(3, "OPEN_LONG", sym="ETHUSDT", p=0.6, ver="new"), _row(4, "CLOSE_LONG", sym="ETHUSDT", pnl=10.0),
            _row(5, "OPEN_LONG", sym="XRPUSDT", p=0.6, ver="new"), _row(6, "DUST_CLEAN_OPEN_LONG", sym="XRPUSDT"),
            _row(7, "OPEN_SHORT", sym="SOLUSDT"), _row(8, "CLOSE_SHORT", sym="SOLUSDT", pnl=-20.0)]
    out = live_vs_test(hist, "new")
    assert out["n"] == 1 and out["live_win"] == 1.0


def test_untagged_history_falls_back_to_all_recent_trades():
    hist = [_row(1, "OPEN_LONG", p=0.4), _row(2, "CLOSE_LONG", pnl=-5.0)]
    out = live_vs_test(hist, "v2")
    assert out["scope"] == "all recent trades" and out["brain"] is None and out["n"] == 1
    assert live_vs_test([], "v2")["n"] == 0
