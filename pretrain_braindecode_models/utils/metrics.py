"""Utility functions for creating metrics and callbacks for model evaluation."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring, PrintLog

from pretrain_braindecode_models.config import logger
from pretrain_braindecode_models.utils.custom_types import PredictorProtocol


def metric_wrapper(
    metric: Callable,
    model: PredictorProtocol,
    X: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> float:
    """Wrap a function to compute a metric for a model."""
    if hasattr(model, "predict"):
        y_pred = model.predict(X)
    else:
        raise ValueError(
            "The model must have a 'predict' method. "
            "If using a PyTorch model, ensure it is wrapped in a skorch Classifier."
        )
    return metric(y.flatten(), y_pred.flatten(), **kwargs)


def balanced_accuracy_metric(
    model: PredictorProtocol,
    X: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> float:
    """Compute balanced accuracy for a model."""
    return metric_wrapper(balanced_accuracy_score, model, X, y, **kwargs)


def precision_metric(
    model: PredictorProtocol,
    X: torch.Tensor,
    y: torch.Tensor,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] | None = "macro",
    zero_division: int | Literal["warn"] = 0,
) -> float:
    """Compute precision for a model."""
    return metric_wrapper(
        precision_score, model, X, y, average=average, zero_division=zero_division
    )


def recall_metric(
    model: PredictorProtocol,
    X: torch.Tensor,
    y: torch.Tensor,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] | None = "macro",
    zero_division: int | Literal["warn"] = 0,
) -> float:
    """Compute recall for a model."""
    return metric_wrapper(recall_score, model, X, y, average=average, zero_division=zero_division)


def f1_score_metric(
    model: PredictorProtocol,
    X: torch.Tensor,
    y: torch.Tensor,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] | None = "macro",
) -> float:
    """Compute F1 score for a model."""
    return metric_wrapper(f1_score, model, X, y, average=average)


def classification_metrics_callbacks(
    *,
    patience: int | None = 25,
    earlystopping_metric: str | None = "valid_bal_acc",
    checkpoint_metric: str | None = "valid_bal_acc",
    checkpoint_dir: Path | str | None = "skorch_checkpoint",
    checkpoint_cb: bool = True,
    earlystopping_cb: bool = True,
) -> list[tuple[str, EpochScoring | Checkpoint | EarlyStopping | PrintLog]]:
    """Create callbacks for classification metrics.

    Supported callbacks:
    - `"train_bal_acc"`: Balanced accuracy on training set.
    - `"valid_bal_acc"`: Balanced accuracy on validation set.
    - `"train_f1"`: F1 score on training set.
    - `"valid_f1"`: F1 score on validation set.
    - `"train_precision"`: Precision on training set.
    - `"valid_precision"`: Precision on validation set.
    - `"train_recall"`: Recall on training set.
    - `"valid_recall"`: Recall on validation set.
    - **PrintLog**: Custom logger for skorch that uses the config's logger.
    - **EarlyStopping**: Stop training when a metric has stopped improving.
    - **Checkpoint**: Save a model checkpoint when a metric improves. It will create a directory
    (checkpoint_dir), in which the `history.json` (which contains the history for the specific
    best epoch) and `full_history.json` (which contains the history of the entire training). It
    will also create the files: `best_criterion_{best_epoch}.pt`, `best_optimizer_{best_epoch}.pt`,
    `best_model_params_{best_epoch}.pt`.

    Args:
        patience (int | None): Number of epochs to wait for improvement if early stopping is
            enabled. If None, defaults to 25.
        earlystopping_metric (str | None): The metric to monitor for early stopping if enabled.
            If None, defaults to "valid_bal_acc".
        checkpoint_metric (str | None): The metric to monitor for model checkpointing if enabled.
            If None, defaults to "valid_bal_acc".
        checkpoint_dir (Path | str | None): Directory to save the model checkpoints if enabled.
        checkpoint_cb (bool): Whether to include the checkpoint callback.
        earlystopping_cb (bool): Whether to include the early stopping callback.

    Returns:
        (list[tuple[str, EpochScoring | Checkpoint | EarlyStopping | PrintLog]]): List of tuples
            containing metric names and their corresponding EpochScoring callbacks:
    """
    train_bal_acc = EpochScoring(
        scoring=balanced_accuracy_metric,
        on_train=True,
        name="train_bal_acc",
        lower_is_better=False,
    )
    valid_bal_acc = EpochScoring(
        scoring=balanced_accuracy_metric,
        on_train=False,
        name="valid_bal_acc",
        lower_is_better=False,
    )
    train_f1 = EpochScoring(
        scoring=f1_score_metric,
        on_train=True,
        name="train_f1",
        lower_is_better=False,
    )
    valid_f1 = EpochScoring(
        scoring=f1_score_metric,
        on_train=False,
        name="valid_f1",
        lower_is_better=False,
    )
    train_precision = EpochScoring(
        scoring=precision_metric,
        on_train=True,
        name="train_precision",
        lower_is_better=False,
    )
    valid_precision = EpochScoring(
        scoring=precision_metric,
        on_train=False,
        name="valid_precision",
        lower_is_better=False,
    )
    train_recall = EpochScoring(
        scoring=recall_metric,
        on_train=True,
        name="train_recall",
        lower_is_better=False,
    )
    valid_recall = EpochScoring(
        scoring=recall_metric,
        on_train=False,
        name="valid_recall",
        lower_is_better=False,
    )

    logger_printer = PrintLog(sink=logger.info)

    callbacks: list[tuple[str, EpochScoring | Checkpoint | EarlyStopping | PrintLog]] = [
        ("train_bal_acc", train_bal_acc),
        ("valid_bal_acc", valid_bal_acc),
        ("train_f1", train_f1),
        ("valid_f1", valid_f1),
        ("train_precision", train_precision),
        ("valid_precision", valid_precision),
        ("train_recall", train_recall),
        ("valid_recall", valid_recall),
        # skorch sees this name and will use our callback INSTEAD of its default stdout printer
        ("logger_print_log", logger_printer),
    ]

    # --- Add the Checkpoint callback ---
    if checkpoint_cb:
        # This callback will monitor the given monitor metric (by default, 'valid_bal_acc' score).
        # Whenever a new best score is found, it saves the model's parameters
        # to a temporary file named 'best_model.pt'.
        checkpoint_metric = checkpoint_metric or "valid_bal_acc"
        checkpoint_dir = checkpoint_dir or "skorch_checkpoint"

        cp = Checkpoint(
            monitor=f"{checkpoint_metric}_best",  # Monitor the 'best' flag of our scoring callback
            # File to save the best model weights
            f_params="best_model_params.pt",
            # File to save the best optimizer state
            f_optimizer="best_optimizer.pt",
            # File to save the best criterion state
            f_criterion="best_criterion.pt",
            f_history="history.json",  # File to save the training history
            load_best=True,  # Automatically load the best weights at the end of training
            dirname=str(checkpoint_dir),  # Directory to store the checkpoint files
            sink=logger.info,
        )

        # History Checkpoint: Saves the full history at the end of every epoch
        history_cp = Checkpoint(
            monitor=None,  # type: ignore[reportArgumentType]
            f_history="full_history.json",  # Only save the history file
            dirname=str(checkpoint_dir),
            # By not specifying a monitor, this triggers on every epoch.
            # We set f_params=None to avoid saving the model weights again here.
            f_params=None,  # type: ignore[reportArgumentType]
            f_optimizer=None,  # type: ignore[reportArgumentType]
            f_criterion=None,  # type: ignore[reportArgumentType]
            load_best=False,  # VERY IMPORTANT: Do not load from this checkpoint
        )

        callbacks.append(("checkpoint", cp))
        callbacks.append(("history_checkpoint", history_cp))

    # --- EarlyStopping callback ---
    if earlystopping_cb:
        if patience is None:
            logger.warning("Early stopping is enabled but no patience is set, defaulting to 25.")
            patience = 25

        if earlystopping_metric is None:
            logger.warning(
                "Early stopping is enabled but no monitor metric is set, defaulting to "
                "'valid_bal_acc'."
            )
            earlystopping_metric = "valid_bal_acc"

        # This will monitor the given monitor metric (by default, validation balanced accuracy).
        # If it doesn't improve for `patience` epochs, training will stop.
        early_stopping = EarlyStopping(
            monitor=earlystopping_metric,  # The metric to watch
            patience=patience,  # Number of epochs to wait for improvement
            # For accuracy/F1, higher is better; for loss lower is better
            lower_is_better="loss" in earlystopping_metric,
        )
        callbacks.append(("early_stopping", early_stopping))

    return callbacks
