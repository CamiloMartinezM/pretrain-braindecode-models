"""PyTorch Lightning Modules."""

from typing import TYPE_CHECKING

import lightning as pl
import torch
from torch import nn, optim
from torcheeg import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.metric import Metric

from pretrain_braindecode_models.config import logger

if TYPE_CHECKING:
    from collections.abc import Mapping


class ClassificationLightningModule(pl.LightningModule):
    """A PyTorch Lightning module for EEG classification tasks.

    This module wraps a Braindecode model and handles the training, validation,
    and testing loops, metric calculation, and optimizer configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_class: type[optim.Optimizer],
        optimizer_kwargs: dict,
        n_classes: int,
        class_weights: torch.Tensor | None = None,
        metrics: dict[str, type[Metric] | tuple[type[Metric], dict[str, str]]] = {
            "balanced_accuracy": (Accuracy, {"average": "weighted"}),
            "f1_score": (F1Score, {"average": "macro"}),
            "precision": (Precision, {"average": "macro"}),
            "recall": (Recall, {"average": "macro"}),
            "accuracy": (Accuracy, {"average": "macro"}),
        },
        aug_transform: nn.Module | transforms.Compose | None = None,
    ) -> None:
        """Initialize the Lightning module for classification.

        Args:
            model (nn.Module): The classification model to be trained.
            optimizer_class (type[optim.Optimizer]): The optimizer class to use.
            optimizer_kwargs (dict): Keyword arguments for the optimizer.
            n_classes (int): Number of output classes.
            class_weights (torch.Tensor | None): Class weights for handling class imbalance.
                By default, no class weights are used.
            metrics (dict): A dictionary of metric names to metric classes or tuples of metric
                classes and their kwargs. Defaults to common classification metrics, e.g.,
                ```
                {
                    "balanced_accuracy": (Accuracy, {"average": "weighted"}),
                    "f1_score": (F1Score, {"average": "macro"}),
                    "precision": (Precision, {"average": "macro"}),
                    "recall": (Recall, {"average": "macro"}),
                    "accuracy": (Accuracy, {"average": "macro"}),
                }
                ```
            aug_transform (nn.Module | transforms.Compose | None): Optional data augmentation
                transform to apply during training. If None, no augmentation is applied.
                Defaults to None.
        """
        super().__init__()
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.n_classes = n_classes
        self._metrics = metrics
        self.aug_transform = aug_transform

        # Use CrossEntropyLoss for multi-class classification
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # --- Setup Metrics using TorchMetrics ---
        # They automatically handle device placement and aggregation.
        metric_args = {"task": "multiclass", "num_classes": self.n_classes}

        self.metrics_dict_: Mapping[str, Metric | MetricCollection] = {}
        for metric_display_name, metric_info in self._metrics.items():
            if isinstance(metric_info, tuple):
                metric, kwargs = metric_info
            else:
                metric = metric_info
                kwargs = {}

            self.metrics_dict_[metric_display_name] = metric(**metric_args, **kwargs)

        self.metrics_dict = MetricCollection(self.metrics_dict_)
        self.train_metrics = self.metrics_dict.clone(prefix="train_")
        self.val_metrics = self.metrics_dict.clone(prefix="val_")

        logger.debug(
            f"Setup lightning module with metrics: {self.metrics_dict}, "
            f"criterion: {self.criterion} and augmentation transforms: {self.aug_transform}"
        )

    def metrics_list(self) -> list[str]:
        """Return a list of metric names used in the module."""
        return list(self._metrics.keys())

    def model_checkpoint_string(self) -> str:
        """Return a string representation of the model checkpoint.

        For example, `"{Val Balanced Acc:.2f}-{Val F1 Score:.2f}"`.

        This can be used for the `ModelCheckpoint` like:
        ```
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="{epoch}-" + lightning_module.model_checkpoint_string(),
                ...
            )
        ```
        """
        metric_strings = []
        for metric in self.val_metrics:
            metric_str = "{" + f"{metric}" + ":.4f}"
            metric_strings.append(metric_str)
        return "-".join(metric_strings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Step for a single batch in training."""
        x, y = batch

        # Apply data augmentation if pipeline is defined
        if self.aug_transform is not None:
            x = self.aug_transform(eeg=x)["eeg"]

        logits = self(x)
        loss = self.criterion(logits, y)

        # Log loss and metrics for this step
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_metrics.update(logits, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Step for a single batch in validation."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log loss and metrics for this step
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_metrics.update(logits, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure the optimizer for the model."""
        # The optimizer is configured here, as is standard in Lightning
        return self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
