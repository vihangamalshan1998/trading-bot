import json
import os
import sys

def main():
    history_file = "models/production/tournament_history.json"
    if not os.path.exists(history_file):
        print("No tournament history recorded yet.")
        return

    try:
        with open(history_file, "r") as f:
            history = json.load(f)
    except Exception as e:
        print(f"Error reading tournament history: {e}")
        return

    if not history:
        print("Tournament history is empty.")
        return

    print("=" * 110)
    print("   AI BRAIN EVOLUTION & TOURNAMENT PROGRESSION LEADERBOARD")
    print("=" * 110)
    print(f"{'#':<3} | {'DATE & TIME':<19} | {'VERSION':<18} | {'STATUS':<9} | {'SCORE (NEW vs OLD)':<20} | {'CRITIC MSE (NEW vs OLD)':<22} | {'PNL':<8}")
    print("-" * 110)

    for idx, h in enumerate(history, 1):
        status = "[WON]" if h.get("approved") else "[REJECTED]"
        c_score = h.get("challenger_score")
        champ_score = h.get("champion_score")
        score_str = f"{c_score:.2f} vs {champ_score:.2f}" if c_score and champ_score else "N/A"

        c_mse = h.get("challenger_critic_mse")
        champ_mse = h.get("champion_critic_mse")
        mse_str = f"{c_mse:.2f} vs {champ_mse:.2f}" if c_mse and champ_mse else "N/A"

        pnl = h.get("challenger_sim_pnl")
        pnl_str = f"+{pnl:.2f}" if pnl else "N/A"

        ver = h.get("version", "unknown")
        date_str = h.get("date", "unknown")

        print(f"{idx:<3} | {date_str:<19} | {ver:<18} | {status:<9} | {score_str:<20} | {mse_str:<22} | {pnl_str:<8}")

    print("=" * 110)
    latest = history[-1]
    print(f"Current Active Version: {latest.get('version')} (Updated: {latest.get('date')})")
    print("=" * 110)

if __name__ == "__main__":
    main()
