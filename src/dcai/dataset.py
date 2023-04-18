from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torchvision
from loguru import logger
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_ROOT = REPO_ROOT / "data"


class TrainDataset(Dataset):
    """Your working train data set.

    This class loads MNIST data and maps the true label 7 to given label 1.

    You can buy an annotation using (this will cost you two annotations):
    ```
    train_dataset.buy_annotation([1, 2])
    ```
    Obviously, it does not make sense to buy annotations for data points that have a given label other than 1.
    You are responsible for not making this mistake.

    You can exclude a data point using:
    ```
    train_dataset.exclude_datapoints([4012, 2031])
    ```
    These data points will not be used when fitting the model in the ScoreTracker function. This is intended for:
     * excluding data points for which the labels could be wrong
     * class balancing

    ⚠️ IMPORTANT: the data points that you exclude are still part of the dataset. This means you can expect
    `train_dataset.x`, `train_dataset.y`, `len(train_dataset)`, etc. to be identical to the values before the operation.
    Only if you use the `train_dataset.subset_from_include_mask()`, you will get a subset.

    A data point can be (re-)included by:
    ```
    train_dataset.include_datapoints([4012])
    ```

    You can flush all masked values using a boolean mask. A common usage is:
    ```
    bool_mask = train_dataset.y != 1  # at least same length as the number of samples in train
    train_dataset.set_include_mask(bool_mask)
    train_dataset.include_datapoints(train_dataset.annotations_bought)
    ```
    """

    def __init__(self):
        super().__init__()

        self.label_mapping = {7: 1}

        dataset = torchvision.datasets.MNIST(str(DATA_ROOT), train=True, download=True)
        self._x = dataset.data
        self._y_original = dataset.targets.numpy()
        self._y_current = np.array(
            [
                y if y not in self.label_mapping else self.label_mapping[y]
                for y in self._y_original
            ]
        )
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
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y_current

    def buy_annotation(self, sample_id: List[int]) -> TrainDataset:
        if sample_id in self._annotations_bought:
            logger.warning(
                f"You already bought an annotation for sample with ID {sample_id}!"
            )

        self._annotations_bought.append(sample_id)
        self._y_current[sample_id] = self._y_original[sample_id]
        return self

    def set_include_mask(self, include_mask: List[bool]) -> TrainDataset:
        self._include_mask = np.array(include_mask)
        return self

    def include_datapoints(self, sample_ids: List[int]) -> TrainDataset:
        self._include_mask[np.array(sample_ids)] = True
        return self

    def exclude_datapoints(self, sample_ids: List[int]) -> TrainDataset:
        self._include_mask[np.array(sample_ids)] = False
        return self

    def subset_from_include_mask(self) -> Dataset:
        return TrainDataSubSet(
            self._x[self._include_mask], self._y_current[self._include_mask]
        )


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

        dataset = torchvision.datasets.MNIST(str(DATA_ROOT), train=False, download=True)
        self._x = dataset.data
        self._y = dataset.targets

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return len(self._x)
