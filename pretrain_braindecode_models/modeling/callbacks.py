"""Utility classes for training callbacks."""

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

from pretrain_braindecode_models.config import logger
from pretrain_braindecode_models.modeling.plotting import (
    plot_history,
    save_classification_analysis,
)
from pretrain_braindecode_models.utils.misc import (
    clean_lightning_metric,
    process_lightning_logs_to_history,
)


class ClassificationAnalysisCallback(Callback):
    """A callback to generate and save classification analysis artifacts.

    Artifacts include: confusion matrix & classification report at key training moments:

    - Before training starts (Epoch 0)
    - At the end of the best epoch found so far
    - At the end of the final training epoch
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        *,
        classes: Sequence[int | str] = [0, 1],
        run_dir: Path,
        monitor_metric: str = "val_balanced_accuracy",
        mode: str = "max",
    ) -> None:
        """Initialize the classification analysis callback.

        Args:
            train_dataloader (DataLoader): The training dataloader to use for evaluation.
            val_dataloader (DataLoader): The validation dataloader to use for evaluation.
            classes (Sequence[int | str]): The list of class labels in the classification task.
                Default is [0, 1].
            run_dir (Path): The directory to save the output plots and reports.
            monitor_metric (str): The metric to monitor for determining the "best" epoch.
            mode (str): The mode for the monitored metric ('min' or 'max').
        """
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.classes = [str(c) for c in classes]
        self.run_dir = run_dir
        self.monitor_metric = monitor_metric
        self.mode = mode

        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = -1

        self.output_dir = self.run_dir / "classification_analysis"
        self.output_dir.mkdir(exist_ok=True)

    def _get_predictions_and_targets(
        self,
        model: pl.LightningModule,
        dataloader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get predictions and true targets from the given model and dataloader."""
        model.eval()

        # Run predictions on the validation set
        all_preds = []
        all_targets = []
        for batch in dataloader:
            x, y = batch
            x = x.to(model.device)
            with torch.no_grad():
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        return all_targets, all_preds

    def _generate_artifacts(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        model: pl.LightningModule,
        stage: str,
    ) -> None:
        """Generate and save the analysis plots and reports."""
        logger.info(f"--- Generating classification analysis for '{stage}' stage ---")

        # Ensure model is in eval mode for consistent predictions
        model.eval()

        all_val_targets, all_val_preds = self._get_predictions_and_targets(
            model, self.val_dataloader
        )
        all_train_targets, all_train_preds = self._get_predictions_and_targets(
            model, self.train_dataloader
        )

        # Use the save_classification_analysis function to save artifacts: confusion matrix
        # & classification report
        save_classification_analysis(
            y_true=all_train_targets,
            y_pred=all_train_preds,
            classes=self.classes,
            cm_filepath=self.output_dir / f"{stage}_train_confusion_matrix.jpg",
            report_filepath=self.output_dir / f"{stage}_train_classification_report.txt",
            cm_title="Training Confusion Matrix",
            figsize=(6, 4),
            show=False,
            dpi=300,
        )

        save_classification_analysis(
            y_true=all_val_targets,
            y_pred=all_val_preds,
            classes=self.classes,
            cm_filepath=self.output_dir / f"{stage}_val_confusion_matrix.jpg",
            report_filepath=self.output_dir / f"{stage}_val_classification_report.txt",
            cm_title="Validation Confusion Matrix",
            figsize=(6, 4),
            show=False,
            dpi=300,
        )

        # Put the model back in training mode
        model.train()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Call before the first training epoch."""
        self._generate_artifacts(trainer, pl_module, "epoch_000_initial")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check for new best epoch."""
        current_score = trainer.callback_metrics.get(self.monitor_metric)
        if current_score is None:
            return

        is_new_best = False
        if (self.mode == "max" and current_score > self.best_score) or (
            self.mode == "min" and current_score < self.best_score
        ):
            is_new_best = True

        if is_new_best and current_score.item() > 0.0:
            self.best_score = current_score.item()
            self.best_epoch = trainer.current_epoch

            logger.info(
                f"New best epoch {self.best_epoch + 1} with {self.monitor_metric}: "
                f"{self.best_score:.4f}"
            )

            # Delete previous best artifacts
            for f in self.output_dir.glob("epoch_*_best_*"):
                f.unlink(missing_ok=True)

            self._generate_artifacts(trainer, pl_module, f"epoch_{self.best_epoch + 1:03d}_best")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Call after the last training epoch."""
        # Generate report for the final model state
        self._generate_artifacts(
            trainer, pl_module, f"epoch_{trainer.current_epoch + 1:03d}_final"
        )


class LoggingCallback(Callback):
    """A PyTorch Lightning callback that logs progress to a loguru logger.

    This replaces the default tqdm progress bar.
    """

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # noqa: ARG002
        outputs,  # noqa: ANN001
        batch,  # noqa: ANN001, ARG002
        batch_idx: int,
    ) -> None:
        """Log training progress at the end of each batch."""
        if outputs is None or (isinstance(outputs, dict) and "loss" not in outputs):
            return

        if not isinstance(outputs, dict):
            return

        loss = outputs["loss"].item()
        # Log every N batches to avoid flooding the logs
        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"Epoch {trainer.current_epoch} | "
                f"Train Batch {batch_idx + 1}/{trainer.num_training_batches} | "
                f"Loss: {loss:.4f}"
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,  # noqa: ANN001
        batch,  # noqa: ANN001
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log validation progress at the end of each batch."""
        # This hook is less commonly used for logging, as we usually care about the epoch-end
        # summary. You can add logic here if you need per-batch validation logs.

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        """Log a summary of metrics at the end of the validation epoch."""
        metrics = trainer.callback_metrics
        log_str = f"Epoch {trainer.current_epoch} Summary:"
        for key, value in metrics.items():
            if "step" not in key:
                log_str += f" | {key}: {value.item():.4f}"
        logger.success(log_str)


class PlottingCallback(Callback):
    """A PyTorch Lightning callback to plot training history."""

    def __init__(
        self,
        run_dir: Path,
        metrics: list[str],
        best_model_metric: str = "val_balanced_accuracy",
        plot_kwargs: dict = {},
    ) -> None:
        """Initialize the plotting callback.

        Args:
            run_dir (Path): The directory where the run outputs will be saved.
            metrics (list[str]): A list of metrics to plot. These must exist in the logged metrics.
            best_model_metric (str): The metric to use for determining the best model.
            plot_kwargs: Additional keyword arguments for the plotting function, i.e.,
            `pretrain_braindecode_models/modeling/plotting.py:plot_history`.
        """
        super().__init__()
        self.run_dir = run_dir
        self.metrics = metrics
        self.best_model_metric = clean_lightning_metric(best_model_metric)
        self.history: dict[str, list[float]] = defaultdict(list)
        self.plot_kwargs = plot_kwargs

        logger.debug(
            f"Initialized PlottingCallback with metrics: {metrics} "
            f"and best_model_metric: {best_model_metric} with additional "
            f"plot_kwargs: {plot_kwargs}"
        )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # noqa: ARG002
    ) -> None:
        """Plot training history at the end of each validation epoch."""
        # Ensure the logger is a CSVLogger and has a log_dir
        if (
            not hasattr(trainer, "logger")
            or trainer.logger is None
            or not hasattr(trainer.logger, "log_dir")
            or not trainer.logger.log_dir
        ):
            logger.warning(
                "PlottingCallback requires a logger with a 'log_dir' attribute, like CSVLogger."
            )
            return

        log_dir = Path(trainer.logger.log_dir)

        try:
            # Use the robust processing function to read the CSV and create the history dict
            history_for_plotting = process_lightning_logs_to_history(log_dir)
            history_dict = history_for_plotting.to_dict()

            if (
                not history_for_plotting
                or "val_loss" not in history_dict
                or "train_loss" not in history_dict
            ):
                logger.debug("Not enough data in metrics.csv to plot history yet.")
                return

            # Find the best epoch based on the specified validation metric
            self.best_epoch_idx = None
            if history_dict.get(self.best_model_metric):
                # Determine if lower is better
                lower_is_better = "loss" in self.best_model_metric.lower()

                if lower_is_better:
                    self.best_epoch_idx = int(np.argmin(history_dict[self.best_model_metric]))
                else:
                    self.best_epoch_idx = int(np.argmax(history_dict[self.best_model_metric]))
            else:
                logger.warning(
                    f"Best model metric '{self.best_model_metric}' not found in history."
                )

            plot_history(
                history=history_for_plotting,
                metrics=self.metrics,
                filepath=str(self.run_dir / "classification_history.png"),
                early_stop_epoch=self.best_epoch_idx,
                **{"show": False, "verbose": False, "is_combined": False, **self.plot_kwargs},
            )

        except FileNotFoundError:
            logger.debug("metrics.csv not found yet, skipping plot.")
        except Exception as e:
            logger.error(f"PlottingCallback failed: {e}")


class EarlyStopping:
    """Early stopping utility class.

    This class monitors a specified metric during training and stops the training process if
    the metric does not improve for a specified number of epochs (patience).

    It can also save the best model based on the validation loss ("val_loss") or a logged
    validation metric ("val_metric").
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0,
        early_stop: str = "val_loss",
        best_model: str = "val_loss",
        mode: str = "min",
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize the early stopping mechanism.

        Args:
            patience (int): Number of epochs with no improvement after which training will be
                stopped.
            delta (float): Minimum change to qualify as an improvement.
            early_stop (str): Metric to monitor for early stopping. Can only be
                "val_loss" or "val_metric".
            best_model (str): Metric to monitor for best model saving. Can only be
                "val_loss" or "val_metric".
            mode (str): One of {min, max}. In `min` mode, training will stop when the quantity
                monitored has stopped decreasing. In `max` mode it will stop when the quantity
                monitored has stopped increasing.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.best_model_state: dict | None = None
        self.verbose = verbose

        self.best_val_early_stop: float | None = None
        self.best_val_metric = 0 if mode == "max" else float("inf")
        self.early_stop_metric = early_stop
        self.early_stop_epoch: int | None = None
        self.best_model_metric = best_model
        self.best_model_epoch: int | None = None

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        epoch: int,
        val_metric: float | None = None,
    ) -> bool:
        """Check if the early stopping condition is met.

        Args:
            val_loss (float): Validation loss for the current epoch.
            model (nn.Module): The model to save the state of.
            epoch (int): Current epoch number.
            val_metric (float | None): Validation metric for the current epoch.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """  # --- Part 1: Early Stopping Logic  ---
        if val_metric is None and self.early_stop_metric != "val_loss":
            raise ValueError("val_metric must be provided if early_stop_metric is not val_loss")

        early_stop_score = val_metric if self.early_stop_metric != "val_loss" else val_loss

        if early_stop_score is None:
            raise ValueError("Early stopping score cannot be None")

        # Determine if lower is better for the early stopping metric
        es_lower_is_better = "loss" in self.early_stop_metric.lower()

        if self.best_val_early_stop is None:
            self.best_val_early_stop = early_stop_score
        elif (
            es_lower_is_better and early_stop_score >= self.best_val_early_stop - self.delta
        ) or (
            not es_lower_is_better and early_stop_score <= self.best_val_early_stop + self.delta
        ):
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        else:
            self.best_val_early_stop = early_stop_score
            self.counter = 0

        # --- Part 2: Best Model Checkpointing Logic ---
        checkpoint_score = val_metric if self.best_model_metric != "val_loss" else val_loss

        if checkpoint_score is None:
            raise ValueError("Checkpoint score cannot be None")

        is_better = False
        if self.mode == "min":
            # Use strict less than '<'
            if checkpoint_score < self.best_val_metric:
                is_better = True
        else:  # mode == "max"
            # Use strict greater than '>'
            if checkpoint_score > self.best_val_metric:
                is_better = True

        if is_better:
            self.best_val_metric = checkpoint_score
            self.best_model_state = deepcopy(model.state_dict())
            self.best_model_epoch = epoch
            if self.verbose:
                logger.success(
                    f"New best model found at epoch {epoch + 1} with score: {checkpoint_score:.4f}"
                )

        # --- Part 3: Trigger Stop ---
        if self.counter >= self.patience:
            self.early_stop_epoch = epoch
            self.early_stop = True
            logger.info(f"Early stopping triggered at epoch {epoch + 1}.")

        return self.early_stop
