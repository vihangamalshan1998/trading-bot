import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

class HistoricalMarketDataset(Dataset):
    """
    A PyTorch Dataset that loads historical market features for offline RL simulation.
    In a full RL pipeline, this would either stream from MySQL or load pre-fetched numpy arrays.
    """
    def __init__(self, raw_data: List[Dict[str, Any]]):
        self.data = raw_data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the feature vector for a specific step.
        """
        row = self.data[idx]
        
        # Example representation of the 6 core features the network expects
        # In reality, balance and position size are appended dynamically by the Environment,
        # but for supervised learning of future prices, this dataset could be used directly.
        features = [
            row.get("spread_bps", 0.0),
            row.get("mid_price", 0.0),
            row.get("micro_price", 0.0),
            row.get("imbalance", 0.0),
            row.get("vwap_recent", 0.0)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
