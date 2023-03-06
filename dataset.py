from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class NCFDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        # rating for negative and positive is already set in the preprocess.py
        user_id, item_id, rating = self.data.iloc[index]
        return torch.LongTensor([user_id, item_id, rating])
