import numpy as np


class SumTree:
    """
    Phase 4: Prioritized Experience Replay (PER) SumTree.

    A binary tree where each leaf stores a priority (|reward|^alpha),
    and each internal node stores the sum of its children's priorities.
    This allows O(log n) proportional sampling instead of O(n) linear search.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, leaf_idx: int, change: float):
        parent = (leaf_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _leaf_for_value(self, value: float) -> int:
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        leaf_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, change)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, change)

    def sample_batch(self, batch_size: int) -> list:
        """
        Stratified sampling: divides total priority into equal segments,
        samples one experience from each. Better coverage than pure random.
        Returns list of (leaf_idx, priority, data).
        """
        results = []
        if self.total_priority <= 0 or self.n_entries == 0:
            return results
        segment = self.total_priority / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            leaf_idx = self._leaf_for_value(value)
            data_idx = leaf_idx - (self.capacity - 1)
            exp = self.data[data_idx]
            if exp is not None:
                results.append((leaf_idx, float(self.tree[leaf_idx]), exp))
        return results

    def rebuild_from_list(self, experiences: list, alpha: float = 0.6) -> None:
        """
        Populate from a list of Experience objects.

        Priority = (best_available_reward + epsilon)^alpha

        Uses reward_4h if Teacher's Assistant has graded it (more accurate),
        otherwise falls back to immediate reward.
        Minimum floor of 0.1 ensures HOLD experiences (reward=0) are still
        sampled — preventing the model from forgetting HOLD is a valid action.
        """
        self.write_idx = 0
        self.n_entries = 0
        self.tree[:] = 0.0
        self.data[:] = None
        for exp in experiences:
            # Prefer the multi-horizon 4h reward if graded — more meaningful signal.
            # Fall back to immediate reward for ungraded experiences.
            best_reward = abs(exp.reward_4h) if exp.reward_4h is not None else abs(exp.reward or 0.0)
            # Floor of 0.1 prevents HOLD (reward=0) from having near-zero priority.
            # Without this, HOLDs are sampled 45,000x less than large-loss trades,
            # causing the model to forget HOLD is a valid action (mode collapse).
            priority = max((best_reward + 1e-5) ** alpha, 0.1)
            self.add(priority, exp)

