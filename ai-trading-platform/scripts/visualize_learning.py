"""
AI learning dashboard: is the bot finding an edge?

Run:  .\\venv\\Scripts\\python.exe scripts\\visualize_learning.py

Top row - TRAINING PROGRESS (models/production/tournament_history.json, v3 evaluator only):
  1. Holdout profit per decision: new brain vs current champion on unseen data, after fees.
     The edge appears when the new brain's line climbs above 0 (and above the champion).
  2. Critic skill (1h): how well the brain forecasts price moves vs "price won't move".
     > 0 = real forecasting skill.
  3. Tournament outcomes: how often new brains were rejected and why.

Bottom row - REAL TRADE OUTCOMES (trade_outcomes table: every record replayed with the live exits):
  4. The live bot's actual entries per coin: average result after fees (stops counted as losses).
  5. The live bot's entries: win / loss after fees.
  6. Market difficulty: at a random moment, does a long win, a short win, or do both lose after fees?
     An edge means picking the winning side clearly more often than this.

Legacy tournaments (before 21 Oct 2026) used junk rewards and are not plotted.
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emoji-safe on any Windows console
except Exception:
    pass

import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

from core.config.settings import settings
from core.risk.exit_policy import ROUND_TRIP_COST

HISTORY_FILE = "models/production/tournament_history.json"
FEE_PCT = ROUND_TRIP_COST * 100.0
GREEN, RED, GREY, CYAN, AMBER = "#4BFF4B", "#FF4B4B", "#888888", "#00E5FF", "#FFC107"


def load_tournaments():
    try:
        with open(HISTORY_FILE, "r") as f:
            hist = json.load(f)
    except Exception:
        return [], 0
    v3 = [e for e in hist if str(e.get("evaluator_version") or "").startswith("v3")]
    return v3, len(hist) - len(v3)


def load_outcomes():
    """Live entries joined with their replayed outcome, plus market-wide outcome counts."""
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        try:
            labelled = conn.execute(text(
                "SELECT COUNT(*) FROM trade_outcomes WHERE src = 'exp' AND long_pct IS NOT NULL")).scalar() or 0
        except Exception:
            return None
        graded = conn.execute(text(
            "SELECT COUNT(*) FROM experiences WHERE data_quality_flags = 'MKT_RET_V2'")).scalar() or 0
        market = conn.execute(text(
            "SELECT "
            f"SUM(long_pct - {FEE_PCT} > 0 AND short_pct - {FEE_PCT} <= 0), "
            f"SUM(short_pct - {FEE_PCT} > 0 AND long_pct - {FEE_PCT} <= 0), "
            f"SUM(long_pct - {FEE_PCT} <= 0 AND short_pct - {FEE_PCT} <= 0) "
            "FROM trade_outcomes WHERE src = 'exp' AND long_pct IS NOT NULL")).fetchone()
        entries = conn.execute(text(
            "SELECT e.symbol, e.action, o.long_pct, o.short_pct FROM trade_outcomes o "
            "JOIN experiences e ON e.id = o.exp_id "
            "WHERE o.src = 'exp' AND o.long_pct IS NOT NULL AND e.action IN ('OPEN_LONG', 'OPEN_SHORT')")).fetchall()
    return {
        "labelled": int(labelled), "graded": int(graded),
        "market": [int(x or 0) for x in market],
        "entries": [(s, a, float(lp), float(sp)) for s, a, lp, sp in entries],
    }


def empty(ax, title, msg):
    ax.set_title(title, fontsize=13)
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=GREY, fontsize=11, wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_progress(ax, v3):
    title = "1. Holdout profit per decision (after fees)"
    if not v3:
        empty(ax, title, "No tournaments with the new evaluator yet.\nRestart MASTER_TRAINER with the new code;\nthe first result comes after 100 steps.")
        return
    x = list(range(1, len(v3) + 1))
    chall = [e.get("challenger_sim_pnl") or 0.0 for e in v3]
    champ = [e.get("champion_sim_pnl") or 0.0 for e in v3]
    ax.plot(x, chall, color=CYAN, marker="o", markersize=3, label="New brain")
    ax.plot(x, champ, color=GREY, linestyle="--", label="Current champion")
    ax.axhline(0, color="white", linewidth=1)
    wins = [i for i, e in zip(x, v3) if e.get("approved")]
    if wins:
        ax.scatter(wins, [chall[i - 1] for i in wins], color=GREEN, s=80, zorder=5, label="TOURNAMENT WINNER")
    ax.fill_between(x, 0, [max(c, 0) for c in chall], color=GREEN, alpha=0.15)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Tournament # (every 100 training steps)")
    ax.set_ylabel("% per decision (0 = staying flat)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_skill(ax, v3):
    title = "2. Critic forecasting skill (1h)"
    rows = [(i + 1, e) for i, e in enumerate(v3) if isinstance(e.get("challenger_critic_skill"), list)]
    if not rows:
        empty(ax, title, "Recorded from the next tournament on.")
        return
    x = [i for i, _ in rows]
    skill = [e["challenger_critic_skill"][1] for _, e in rows]
    champ = [e["champion_critic_skill"][1] if isinstance(e.get("champion_critic_skill"), list) else None for _, e in rows]
    ax.plot(x, skill, color=CYAN, marker="o", markersize=3, label="New brain")
    if any(c is not None for c in champ):
        ax.plot(x, [c if c is not None else float("nan") for c in champ], color=GREY, linestyle="--", label="Current champion")
    ax.axhline(0, color="white", linewidth=1)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Tournament #")
    ax.set_ylabel("skill (> 0 = better than 'no move')")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_reasons(ax, v3, n_legacy):
    title = "3. Tournament outcomes"
    if not v3:
        empty(ax, title, f"{n_legacy} legacy tournaments hidden\n(scored with the old junk rewards).")
        return
    labels = {"SUPERIOR_TO_CHAMPION": "WINNER", "NO_EDGE_AFTER_COSTS": "No edge\nafter fees",
              "INFERIOR_TO_CHAMPION": "Worse than\nchampion", "CRITIC_DEGRADED": "Critic\nworse",
              "ACTION_MONOCULTURE": "One-sided\n(monoculture)"}
    counts = defaultdict(int)
    for e in v3:
        counts[labels.get(e.get("reason"), "Other")] += 1
    names = list(counts)
    vals = [counts[n] for n in names]
    colors = [GREEN if n == "WINNER" else (AMBER if n.startswith("No edge") else RED) for n in names]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", color="white", fontweight="bold")
    last = v3[-1]
    note = f"Last: {last.get('date', '')}  |  {n_legacy} legacy hidden"
    if last.get("blocked_symbols") is not None:
        note += f"\nLast winner policy: min conf {last.get('min_confidence')}, blocked {last.get('blocked_symbols')}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(note, fontsize=9, color=GREY)
    ax.grid(axis="y", alpha=0.3)


def entry_results(entries):
    """Net % after fees of each live entry under the real exits."""
    out = []
    for sym, action, lp, sp in entries:
        gross = lp if action == "OPEN_LONG" else sp
        out.append((sym, gross - FEE_PCT))
    return out


def plot_coins(ax, oc):
    title = "4. Live bot entries per coin (real exits, after fees)"
    if not oc or not oc["entries"]:
        empty(ax, title, "No labelled live entries yet.\nThe trainer labels history in the background\n(~2,000 records every 30 s).")
        return
    by = defaultdict(list)
    for sym, net in entry_results(oc["entries"]):
        by[sym].append(net)
    rows = sorted(((s, sum(v) / len(v), len(v)) for s, v in by.items()), key=lambda r: r[1])
    names = [s.replace("USDT", "") for s, _, _ in rows]
    vals = [m for _, m, _ in rows]
    bars = ax.barh(names, vals, color=[GREEN if v > 0 else RED for v in vals])
    for b, (_, m, n) in zip(bars, rows):
        ax.text(max(b.get_width(), 0.0), b.get_y() + b.get_height() / 2, f" {m:+.2f}% (n={n})", va="center",
                ha="left", color="white", fontsize=8)
    ax.axvline(0, color="white", linewidth=1)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("average % of position per trade")
    ax.grid(axis="x", alpha=0.3)


def plot_entry_pie(ax, oc):
    title = "5. Live bot entries: win / loss after fees"
    if not oc or not oc["entries"]:
        empty(ax, title, "No labelled live entries yet.")
        return
    nets = [n for _, n in entry_results(oc["entries"])]
    wins = sum(1 for n in nets if n > 0)
    losses = len(nets) - wins
    avg = sum(nets) / len(nets)
    ax.pie([wins, losses], labels=[f"Winners\n({wins})", f"Losers\n({losses})"], colors=[GREEN, RED],
           explode=(0.08, 0), autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11, "fontweight": "bold"})
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"Average per trade: {avg:+.3f}% after fees  |  {len(nets)} entries", fontsize=10,
                  color=GREEN if avg > 0 else RED)


def plot_market(ax, oc):
    title = "6. Market difficulty (any moment, after fees)"
    if not oc or sum(oc["market"]) == 0:
        empty(ax, title, "No labelled records yet.")
        return
    long_w, short_w, both_l = oc["market"]
    total = long_w + short_w + both_l
    names = ["Long wins", "Short wins", "Both lose\n(stay flat!)"]
    vals = [100 * v / total for v in (long_w, short_w, both_l)]
    bars = ax.bar(names, vals, color=[GREEN, RED, GREY])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", color="white", fontweight="bold")
    progress = 100 * oc["labelled"] / max(1, oc["graded"])
    if progress < 50:
        ax.text(0.5, 0.55, f"EARLY SAMPLE ({progress:.0f}% labelled) - not representative yet",
                ha="center", color=AMBER, fontsize=10, transform=ax.transAxes)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("% of moments")
    ax.set_xlabel(f"Labelled {oc['labelled']:,} of {oc['graded']:,} graded records ({progress:.0f}%)", fontsize=9, color=GREY)
    ax.grid(axis="y", alpha=0.3)


def print_summary(v3, n_legacy, oc):
    print(f"\nTournaments (new evaluator): {len(v3)}   (legacy hidden: {n_legacy})")
    if v3:
        last = v3[-1]
        print(f"  Last: {last.get('date')}  {last.get('reason')}  new {last.get('challenger_sim_pnl'):+.4f}% vs champion {last.get('champion_sim_pnl'):+.4f}% per decision")
        best = max(v3, key=lambda e: e.get("challenger_sim_pnl") or -1e9)
        print(f"  Best new brain so far: {best.get('challenger_sim_pnl'):+.4f}% per decision ({best.get('date')})")
        print(f"  Winners: {sum(1 for e in v3 if e.get('approved'))}")
    if oc:
        print(f"Real-outcome labels: {oc['labelled']:,} / {oc['graded']:,} graded records")
        if oc["entries"]:
            nets = [n for _, n in entry_results(oc["entries"])]
            print(f"  Live bot entries: {len(nets)}, win rate {sum(n > 0 for n in nets) / len(nets):.1%}, avg {sum(nets) / len(nets):+.3f}% after fees")


def plot_dashboard():
    print("📊 Analyzing AI learning (tournaments + real trade outcomes)...")
    v3, n_legacy = load_tournaments()
    try:
        oc = load_outcomes()
    except Exception as e:
        print(f"❌ Database error: {e}")
        oc = None
    print_summary(v3, n_legacy, oc)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(f"AI Trading Bot - Is it finding an edge?   ({datetime.now():%Y-%m-%d %H:%M})",
                 fontsize=20, fontweight="bold", color="cyan")
    plot_progress(axes[0][0], v3)
    plot_skill(axes[0][1], v3)
    plot_reasons(axes[0][2], v3, n_legacy)
    plot_coins(axes[1][0], oc)
    plot_entry_pie(axes[1][1], oc)
    plot_market(axes[1][2], oc)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    print("✅ Dashboard ready. Displaying window...")
    plt.show()


if __name__ == "__main__":
    plot_dashboard()
