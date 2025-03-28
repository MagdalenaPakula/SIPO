import os
import sys

from torch.utils.data import Dataset

from src.tools import load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class CustomDataset(Dataset):
    def __init__(self, data=None, targets=None, data_path=None, tokenizer=None):
        if data_path:
            self.data, self.targets = load(data_path)
        else:
            self.data = data
            self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.targets[index]
        if self.tokenizer:
            sample = self.tokenizer.tokenize(sample)
        return sample, label
