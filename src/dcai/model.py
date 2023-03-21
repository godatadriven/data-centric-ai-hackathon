from typing import Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim

cpu = torch.device('cpu')


class SimpleMnistModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output

    @property
    def num_classes(self):
        return 10


class SimpleMnistFitter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleMnistModel()

        self.conf_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=self.model.num_classes,
            normalize="none"
        )

        self.precision = torchmetrics.Precision(
            "multiclass",
            num_classes=self.model.num_classes,
            average=None
        )

        self.recall = torchmetrics.Recall(
            "multiclass",
            num_classes=self.model.num_classes,
            average=None
        )

    def forward(self, x, **kwargs: Any) -> Any:
        x = torch.asarray(x.reshape(len(x), 1, *x.shape[1:]))
        x = x.to(torch.float32)
        x = x / 127. - 1

        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.nn.CrossEntropyLoss()(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def on_validation_start(self) -> None:
        self.conf_matrix.reset()
        self.precision.reset()
        self.recall.reset()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:

        x, y = batch
        pred = self(x)

        self.conf_matrix(pred, y)
        val_loss = torch.nn.CrossEntropyLoss()(pred, y)

        self.precision(pred, y)
        self.recall(pred, y)

        self.log("val_loss", val_loss)
        return None
