import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import TrainDataset, ValidationDataset
from casper.utils import seed_worker

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
        generator = torch.Generator()
        generator.manual_seed(int(getattr(args, "seed", 42)))
        super().__init__(
            self.dataset,
            batch_size=None,
            sampler=self.sampler,
            num_workers=6,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
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
        generator = torch.Generator()
        generator.manual_seed(int(getattr(args, "seed", 42)))
        super().__init__(
            self.dataset,
            batch_size=None,
            sampler=self.sampler,
            num_workers=1,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
