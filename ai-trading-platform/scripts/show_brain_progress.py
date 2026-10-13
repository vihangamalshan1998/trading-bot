import json
import os
import sys

# Ensure UTF-8 output on Windows consoles if supported
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def main():
    history_file = "models/production/tournament_history.json"
    if not os.path.exists(history_file):
        print("No tournament history recorded yet.")
        return

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        print(f"Error reading tournament history: {e}")
        return

    if not history:
        print("Tournament history is empty.")
        return

    print("=" * 118)
    print("                              AI BRAIN EVOLUTION & TOURNAMENT LEADERBOARD")
    print("=" * 118)
    header = (
        f"{'#':<3} | {'DATE & TIME':<19} | {'VERSION':<18} | {'RESULT':<11} | "
        f"{'SCORE DELTA':<17} | {'CRITIC MSE DELTA':<23} | {'SIM PNL':<9} | {'ENTROPY':<8}"
    )
    print(header)
    print("-" * 118)

    base_mse = None
    for idx, h in enumerate(history, 1):
        c_score = h.get("challenger_score") or 0.0
        champ_score = h.get("champion_score") or 0.0
        c_mse = h.get("challenger_critic_mse") or 0.0
        champ_mse = h.get("champion_critic_mse") or 0.0
        c_pnl = h.get("challenger_sim_pnl") or 0.0
        champ_pnl = h.get("champion_sim_pnl") or 0.0
        entropy = h.get("entropy") or 0.0
        approved = h.get("approved", False)
        reason = h.get("reason", "")
        ver = h.get("version", "unknown")
        date_str = h.get("date", "unknown")

        if base_mse is None and c_mse > 0:
            base_mse = c_mse

        # Result badge (safe ASCII)
        if idx == 1 and "BASELINE" in reason:
            res_str = "[BASELINE]"
            score_delta_str = f"{c_score:.2f} (Base)"
            mse_delta_str = f"{c_mse:.2f} (Base Error)"
            pnl_str = f"+{c_pnl:.2f}"
        else:
            res_str = "[PROMOTED]" if approved else "[REJECTED]"
            
            # Score delta
            d_score = c_score - champ_score
            d_score_pct = (d_score / abs(champ_score) * 100) if champ_score else 0.0
            sign_s = "+" if d_score >= 0 else ""
            score_delta_str = f"{sign_s}{d_score:.2f} ({sign_s}{d_score_pct:.1f}%)"

            # MSE delta (Lower is better)
            d_mse = c_mse - champ_mse
            d_mse_pct = (d_mse / champ_mse * 100) if champ_mse else 0.0
            sign_m = "+" if d_mse >= 0 else ""
            indicator = "Better" if d_mse < 0 else "Worse"
            mse_delta_str = f"{sign_m}{d_mse:.2f} ({sign_m}{d_mse_pct:.1f}% {indicator})"

            # PnL
            sign_p = "+" if c_pnl >= 0 else ""
            pnl_str = f"{sign_p}{c_pnl:.2f}"

        entropy_str = f"{entropy:.2f}"
        row = (
            f"{idx:<3} | {date_str:<19} | {ver:<18} | {res_str:<11} | "
            f"{score_delta_str:<17} | {mse_delta_str:<23} | {pnl_str:<9} | {entropy_str:<8}"
        )
        print(row)

    print("=" * 118)

    # Latest Brain Detailed Head-to-Head Card
    latest = history[-1]

    l_ver = latest.get("version", "unknown")
    l_date = latest.get("date", "unknown")
    l_approved = latest.get("approved", False)
    l_reason = latest.get("reason", "N/A")
    l_score = latest.get("challenger_score", 0.0)
    l_cscore = latest.get("champion_score", 0.0)
    l_mse = latest.get("challenger_critic_mse", 0.0)
    l_cmse = latest.get("champion_critic_mse", 0.0)
    l_pnl = latest.get("challenger_sim_pnl", 0.0)
    l_cpnl = latest.get("champion_sim_pnl", 0.0)
    l_ent = latest.get("entropy", 0.0)

    print(f"\n[*] LATEST SHOWDOWN ANALYSIS: {l_ver} vs REIGNING CHAMPION")
    print("-" * 118)
    status_label = "APPROVED & DEPLOYED TO LIVE VPS BOT" if l_approved else "REJECTED (Retained Current Champion)"
    print(f" * Promotion Status:    {status_label}")
    print(f" * Matchup Reason:       {l_reason}")
    print(f" * Decision Score:       {l_score:.4f} vs Champion {l_cscore:.4f} (Net Gain: {l_score - l_cscore:+.4f})")
    
    mse_diff = l_mse - l_cmse
    mse_pct = (mse_diff / l_cmse * 100) if l_cmse else 0.0
    if mse_diff < 0:
        accuracy_verdict = f"{abs(mse_pct):.1f}% reduction in value prediction error (Sharper market anticipation)"
    else:
        accuracy_verdict = f"{mse_pct:.1f}% higher critic error"
    print(f" * Price Forecast MSE:  {l_mse:.4f} vs Champion {l_cmse:.4f} -> {accuracy_verdict}")

    pnl_diff = l_pnl - l_cpnl
    print(f" * Simulated PnL Edge:   {l_pnl:+.4f} vs Champion {l_cpnl:+.4f} (Advantage: {pnl_diff:+.4f})")

    ent_desc = "Optimal balance of exploration and conviction" if 0.5 <= l_ent <= 1.2 else "Watch closely"
    print(f" * Policy Entropy:       {l_ent:.4f} ({ent_desc})")

    if base_mse and base_mse > 0:
        cum_gain = ((l_mse - base_mse) / base_mse) * 100
        print(f" * Cumulative Progress:  Critic prediction error improved {abs(cum_gain):.1f}% since Day 1 Baseline ({base_mse:.2f} -> {l_mse:.2f})")
    
    print("=" * 118)

if __name__ == "__main__":
    main()
