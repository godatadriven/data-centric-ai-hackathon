from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import DataLoader

from dcai.dataset import TrainDataset, ValidationDataset
from dcai.model import LitSimpleMnist


@dataclass
class Attempt:
    n_annotations_bought: int
    precision_class_1: float
    recall_class_1: float
    precision_class_7: float
    recall_class_7: float


class ScoreTracker:

    def __init__(self, team_name: str):
        self.team_name = team_name
        self.all_attempts: List[Attempt] = []

    def train_and_score_model(self, train_dataset: TrainDataset, plot_confusion_matrix: bool = True) -> LitSimpleMnist:

        progress_bar = RichProgressBar()

        vds = ValidationDataset()
        model = LitSimpleMnist()
        trainer = pl.Trainer(limit_train_batches=None, max_epochs=1, callbacks=[progress_bar])
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(train_dataset.subset_from_include_mask(), batch_size=32),
            val_dataloaders=DataLoader(vds, batch_size=1000),
        )

        if plot_confusion_matrix:
            confusion_matrix = model.conf_matrix.compute()
            plt.imshow(confusion_matrix.T)
            plt.title("Confusion Matrix")
            plt.xlabel("predicted_label")
            plt.ylabel("true label")

        precision = model.precision.compute().numpy()
        recall = model.recall.compute().numpy()

        attempt = Attempt(
            n_annotations_bought=len(train_dataset.annotations_bought),
            precision_class_1=precision[1],
            recall_class_1=recall[1],
            precision_class_7=precision[7],
            recall_class_7=recall[7],
        )

        self.all_attempts.append(attempt)
        return model

    def plot_scores(self):

        x = [a.n_annotations_bought for a in self.all_attempts]
        p1 = [a.precision_class_1 for a in self.all_attempts]
        r1 = [a.recall_class_1 for a in self.all_attempts]
        p7 = [a.precision_class_7 for a in self.all_attempts]
        r7 = [a.recall_class_7 for a in self.all_attempts]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
        ax[0].set_title("Precision")
        ax[0].scatter(x, p1, marker="x", label="1")
        ax[0].scatter(x, p7, marker="o", label="7")
        ax[0].legend()
        ax[0].set_xlabel("Number of annotations bought")
        ax[0].set_ylabel("Precision")
        ax[0].set_ylim([0, 1])

        ax[1].set_title("Recall")
        ax[1].scatter(x, r1, marker="x", label="1")
        ax[1].scatter(x, r7, marker="o", label="7")
        ax[1].legend()
        ax[1].set_xlabel("Number of annotations bought")
        ax[1].set_ylabel("Recall")
        ax[1].set_ylim([0, 1])

        plt.show()
