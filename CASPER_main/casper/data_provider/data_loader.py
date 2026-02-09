import numpy as np
from torch.utils.data import DataLoader
from .dataset import TrainDataset, ValidationDataset

class TrainSampler:
    def __init__(self):
        pass

    def __iter__(self):
        while True:
            yield np.random.randint(0, 998244353)

class TrainLoader(DataLoader):
    def __init__(self, args):
        self.dataset = TrainDataset(args)
        self.sampler = TrainSampler()
        super().__init__(
            self.dataset,
            batch_size=None,
            sampler=self.sampler,
            num_workers=6,
            pin_memory=True,
        )

class ValidationSampler:
    def __init__(self, end_iter):
        self.end_iter = end_iter

    def __iter__(self):
        i = 0
        while True:
            if i == self.end_iter:
                return
            else:
                yield i
                i += 1

class ValidationLoader(DataLoader):
    def __init__(self, args):
        self.dataset = ValidationDataset(args)
        self.sampler = ValidationSampler(end_iter=len(self.dataset))
        super().__init__(
            self.dataset,
            batch_size=None,
            sampler=self.sampler,
            num_workers=1,
            pin_memory=True,
        )