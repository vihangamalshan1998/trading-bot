import os
import pymysql
import matplotlib.pyplot as plt
from collections import defaultdict

print("📊 Analyzing AI Learning Database...")

# Database connection configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'ai_trading',
    'cursorclass': pymysql.cursors.DictCursor
}

def fetch_data():
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        # 1. Action Counts
        cursor.execute("SELECT action, COUNT(*) as c FROM experiences GROUP BY action")
        actions = cursor.fetchall()
        
        # 2. Win/Loss (4h Horizon)
        cursor.execute("SELECT COUNT(*) as c FROM experiences WHERE reward_4h > 0")
        wins = cursor.fetchone()['c']
        
        cursor.execute("SELECT COUNT(*) as c FROM experiences WHERE reward_4h < 0")
        losses = cursor.fetchone()['c']
        
        # 3. Average Rewards
        rewards_data = {}
        timeframes = ['reward', 'reward_5m', 'reward_1h', 'reward_4h']
        for tf in timeframes:
            cursor.execute(f"SELECT AVG({tf}) as avg_r FROM experiences WHERE {tf} IS NOT NULL")
            val = cursor.fetchone()['avg_r']
            rewards_data[tf] = float(val) if val is not None else 0.0

        conn.close()
        return actions, wins, losses, rewards_data

    except Exception as e:
        print(f"❌ Database Error: {e}")
        return None, None, None, None

def plot_dashboard():
    actions, wins, losses, rewards_data = fetch_data()
    if not actions:
        return
    
    # Setup the plot layout (1 row, 3 columns)
    plt.style.use('dark_background')  # Looks much cooler for AI trading!
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('🧠 AI Trading Bot - Learning Metrics Dashboard', fontsize=20, fontweight='bold', color='cyan')
    
    # -----------------
    # CHART 1: Actions
    # -----------------
    action_labels = [row['action'] for row in actions]
    action_counts = [row['c'] for row in actions]
    colors = ['#FF4B4B' if 'SHORT' in a else '#4BFF4B' if 'LONG' in a else '#888888' for a in action_labels]
    
    axes[0].bar(action_labels, action_counts, color=colors)
    axes[0].set_title('Buttons Pressed (Exploration Phase)', fontsize=14)
    axes[0].set_ylabel('Total Count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # -----------------
    # CHART 2: Win/Loss Ratio
    # -----------------
    labels = [f'Winners\n({wins})', f'Losers\n({losses})']
    sizes = [wins, losses]
    explode = (0.1, 0)  
    
    axes[1].pie(sizes, explode=explode, labels=labels, colors=['#4BFF4B', '#FF4B4B'], 
            autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Win/Loss Ratio (4-Hour Horizon)', fontsize=14)
    
    # -----------------
    # CHART 3: Average Rewards
    # -----------------
    tf_labels = ['Immediate\n(Fee Penalty)', '5-Min', '1-Hour', '4-Hour']
    tf_vals = [rewards_data['reward'], rewards_data['reward_5m'], rewards_data['reward_1h'], rewards_data['reward_4h']]
    
    # Color red for negative, green for positive
    r_colors = ['#FF4B4B' if v < 0 else '#4BFF4B' for v in tf_vals]
    
    bars = axes[2].bar(tf_labels, tf_vals, color=r_colors)
    axes[2].set_title('Average Reward & Punishment', fontsize=14)
    axes[2].set_ylabel('Reward Value')
    axes[2].axhline(0, color='white', linewidth=1)
    
    # Add exact values on top of bars
    for bar in bars:
        yval = bar.get_height()
        offset = 5 if yval >= 0 else -15
        axes[2].text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.1f}', 
                 ha='center', va='bottom', color='white', fontweight='bold')
    
    # Final adjustments
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("✅ Successfully generated Dashboard! Displaying window...")
    plt.show()

if __name__ == "__main__":
    plot_dashboard()
