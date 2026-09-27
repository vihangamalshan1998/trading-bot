## 1. Lack of Temporal Memory (The "Goldfish" Problem)
**Impact on Win Rate: SEVERE**
*   **The Master Prompt says:** *"The model must understand sequences rather than only one market snapshot. Investigate: GRU, LSTM, Transformer."*
*   **Our Current System:** Our PyTorch model (`ppo.py`) uses a standard Multi-Layer Perceptron (MLP). It looks at the 200-slot vector for a single second, makes a decision, and immediately forgets everything. It has no short-term memory.
*   **Why it hurts Win Rate:** In crypto, price *momentum* and *sequence* matter. A crash that happens slowly over 2 hours means something very different than a flash-crash that happens in 3 seconds. Because our AI has no memory (LSTM/GRU), it cannot "feel" momentum. It trades purely on static snapshots.


## 2. Single-Step Reward vs. Multi-Horizon Prediction
**Impact on Win Rate: MEDIUM**
*   **The Master Prompt says:** *"The AI should eventually predict multiple future horizons (5m, 15m, 1h, 4h)."*
*   **Our Current System:** Our PPO calculates reward based strictly on the immediate next execution step. It is essentially a high-frequency scalper.
*   **Why it hurts Win Rate:** It will close a winning trade too early for a tiny profit because it isn't "looking ahead" to the 1-hour or 4-hour trend to ride massive macro waves.

---

# Projected Impact on Final Goal Achievement

If we successfully implement both the **LSTM** and the **Multi-Horizon Prediction**, the impact on your final goal (consistent, automated profitability) will be massive:

1.  **Profit Factor Explosion:** Your Win Rate might only go up slightly (e.g., from 48% to 55%), but your **Profit Factor** will skyrocket. Because of the Multi-Horizon prediction, the AI will learn to cut losses quickly but hold winning trades for hours, resulting in massive $500 wins and tiny $10 losses.
2.  **Choppy Market Immunity:** Because the LSTM gives the AI memory, it will recognize when the market is just bouncing sideways ("choppy"). Instead of getting faked out and losing fees on fake breakouts, it will recognize the pattern and correctly choose to **HOLD** until the sequence changes.

This upgrade transforms the bot from a "reactive scalper" into a "strategic swing trader."

---

# The V3 "Brain" Implementation Plan

Because both of these upgrades require changing the physical structure of the AI's neural network, we will do them at the exact same time. This means we only have to wipe the current training memory once. 

Here is the exact step-by-step engineering plan to build it:

### Phase 1: Upgrading the Neural Network (`core/ai/model.py`)
*   **Task:** Rip out the basic `nn.Linear` layers and replace them with an `nn.LSTM` (Long Short-Term Memory) module.
*   **Task:** Split the `Critic` network into three distinct output heads (`value_5m`, `value_1h`, `value_4h`).
*   **Result:** The AI's physical brain can now store hidden states over time and output predictions for the future.

### Phase 2: Upgrading the Database Fetcher (`core/ai/replay_buffer.py`)
*   **Task:** Rewrite the `sample()` function. Currently, it pulls 64 random, disconnected rows from MySQL. It must be rewritten to pull 64 continuous **Time Sequences** (e.g., a block of 16 consecutive seconds). 
*   **Task:** Modify the `calculate_returns()` function. It must now look into the future of the dataset and calculate exactly how much profit the trade made at the 5-minute mark, 1-hour mark, and 4-hour mark.

### Phase 3: Upgrading the Teacher (`apps/trainer/ppo.py`)
*   **Task:** Update the PPO math to pass the LSTM's `hidden_states` (`hx`, `cx`) forward through time during the training loop.
*   **Task:** Rewrite the `critic_loss` function. Instead of just grading the AI on its immediate next step, the teacher will now grade it out of 3: `total_loss = loss_5m + loss_1h + loss_4h`.
*   **Task:** Delete `model_v1.pt`, restart the VPS process, and begin the V3 training epoch.
