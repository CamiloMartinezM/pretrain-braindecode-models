"""Custom type definitions."""

import enum
from collections.abc import Hashable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypedDict, TypeVar, runtime_checkable

import numpy as np
import pandas as pd
import torch
from mne.decoding import Scaler as MNEScaler
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import Dataset

from pretrain_braindecode_models.config import DEVICE, logger

K = TypeVar("K", bound=Hashable)

DimOrder = Literal[
    "NCT",
    "NTC",  # 3D: (Batch, Channels, Time) or (Batch, Time, Channels)
    "CT",
    "TC",  # 2D: (Channels, Time) or (Time, Channels)
    "NC",
    "CN",  # 2D: (Batch, Channels) - useful for flattened features
]

ArrayOrTensor = np.ndarray | torch.Tensor
ScikitScaler = StandardScaler | MinMaxScaler | RobustScaler | MaxAbsScaler
MNEOrScikitScaler = MNEScaler | ScikitScaler


class RunStatus(enum.Enum):
    """Enumeration for run status, i.e., `Literal["running", "completed", "failed", "empty"]`."""

    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"
    EMPTY = "empty"


class RunInfo(TypedDict):
    """Holds information about a single best run."""

    path: Path
    best_value: float


@dataclass
class TrainingHistoryClassification:
    """A data container for the complete training history of a model run in a classification task.

    **Required fields:**
        - `train_loss`: List of training loss values.
        - `val_loss`: List of validation loss values.

    **Optional fields:**
        - `train_loss_step`: List of training loss values per optimization step (if available).
        - `val_loss_step`: List of validation loss values per optimization step (if available).
        - `train_balanced_accuracy`, `val_balanced_accuracy`: List of balanced accuracy values.
        - `train_accuracy`, `val_accuracy`: List of accuracy values.
        - `train_precision`, `val_precision`: List of precision values.
        - `train_recall`, `val_recall`: List of recall values.
        - `train_f1_score`, `val_f1_score`: List of F1 score values.
        - `early_stop_epoch`: The epoch number (0-indexed) at which early stopping occurred.
        - `lr`: List of learning rate values for each epoch.
    """

    # --- Required fields ---
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)

    # --- Optional fields ---
    train_loss_step: list[float] = field(default_factory=list)
    val_loss_step: list[float] = field(default_factory=list)
    train_balanced_accuracy: list[float] = field(default_factory=list)
    val_balanced_accuracy: list[float] = field(default_factory=list)
    train_f1_score: list[float] = field(default_factory=list)
    val_f1_score: list[float] = field(default_factory=list)
    train_precision: list[float] = field(default_factory=list)
    val_precision: list[float] = field(default_factory=list)
    train_recall: list[float] = field(default_factory=list)
    val_recall: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)

    lr: list[float] = field(default_factory=list)
    early_stop_epoch: int | None = None

    def to_dict(self, *, ignore_empty: bool = True) -> dict:
        """Convert the dataclass to a dictionary, optionally filtering empty lists.

        Args:
            ignore_empty (bool): If True, keys with empty lists as values will be excluded from
                the resulting dictionary. Defaults to True.

        Returns:
            dict: The dictionary representation of the training history.
        """
        # `asdict` is a convenient function from the dataclasses module
        # that recursively converts a dataclass instance to a dict.
        data = asdict(self)

        # Make sure that "train_loss" and "val_loss" are the first keys
        ordered_keys = ["train_loss", "val_loss"] + [
            k for k in data if k not in ("train_loss", "val_loss")
        ]
        data = {k: data[k] for k in ordered_keys}

        if not ignore_empty:
            return data

        # Filter out keys where the value is an empty list
        return {
            key: value
            for key, value in data.items()
            if not (isinstance(value, list) and not value)
        }


@runtime_checkable
class ScalerProtocol(Protocol):
    """Protocol for scaler-like objects that can fit and transform data."""

    def fit(self, epochs_data, y=None) -> "ScalerProtocol": ...  # noqa: ANN001, D102
    def transform(self, epochs_data) -> np.ndarray: ...  # noqa: ANN001, D102
    def fit_transform(self, epochs_data, y=None) -> np.ndarray: ...  # noqa: ANN001, D102
    def inverse_transform(self, data) -> np.ndarray: ...  # noqa: ANN001, D102


@runtime_checkable
class PredictorProtocol(Protocol):
    """Protocol for a model that can make predictions."""

    def predict(self, X: torch.Tensor | Dataset) -> torch.Tensor:
        """Predict labels for the input data."""
        ...


class SupportsArithmetic(Protocol):
    """Protocol for objects that support arithmetic operations."""

    def __sub__(self, other) -> "SupportsArithmetic": ...  # noqa: ANN001, D105
    def __add__(self, other) -> "SupportsArithmetic": ...  # noqa: ANN001, D105
    def __mul__(self, other) -> "SupportsArithmetic": ...  # noqa: ANN001, D105


class SupportsComparison(Protocol):
    """Protocol for objects that support comparison operations."""

    def __lt__(self, other) -> bool: ...  # noqa: ANN001, D105
    def __le__(self, other) -> bool: ...  # noqa: ANN001, D105
    def __gt__(self, other) -> bool: ...  # noqa: ANN001, D105
    def __ge__(self, other) -> bool: ...  # noqa: ANN001, D105
