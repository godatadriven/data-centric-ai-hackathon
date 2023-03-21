from __future__ import annotations

from typing import List

import numpy as np
import torchvision
from loguru import logger
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self):
        super().__init__()

        self.label_mapping = {7: 1}

        dataset = torchvision.datasets.MNIST("../../data/", train=True, download=True)
        self._x = dataset.data
        self._y_original = dataset.targets.numpy()
        self._y_current = np.array([
            y if y not in self.label_mapping else self.label_mapping[y]
            for y in self._y_original
        ])
        self._annotations_bought = []
        self._include_mask = np.array([True] * len(self._x))

    def __getitem__(self, item):
        return self._x[item], self._y_current[item]

    def __len__(self):
        return len(self._x)

    @property
    def annotations_bought(self):
        return self._annotations_bought

    @annotations_bought.getter
    def annotations_bought(self):
        return self._annotations_bought

    @annotations_bought.setter
    def annotations_bought(self, value):
        raise ValueError("Cheater!")

    @property
    def labels(self):
        return self._y_current

    def buy_annotation(self, sample_id: List[int]) -> TrainDataset:

        if sample_id in self._annotations_bought:
            logger.warning(f"You already bought an annotation for sample with ID {sample_id}!")

        self._annotations_bought.append(sample_id)
        self._y_current[sample_id] = self._y_original[sample_id]
        return self

    def include_datapoints(self, sample_ids: List[int]) -> TrainDataset:
        self._include_mask[np.array(sample_ids)] = True
        return self

    def exclude_datapoints(self, sample_ids: List[int]) -> TrainDataset:
        self._include_mask[np.array(sample_ids)] = False
        return self

    def get_train_subset(self) -> Dataset:
        return TrainDataSubSet(self._x[self._include_mask], self._y_current[self._include_mask])


class TrainDataSubSet(Dataset):

    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return len(self._x)


class ValidationDataset(Dataset):

    def __init__(self):
        super().__init__()

        dataset = torchvision.datasets.MNIST("../../data/", train=False, download=True)
        self._x = dataset.data
        self._y = dataset.targets

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return len(self._x)
