"""Training functions."""

import inspect
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import pandas as pd
import torch
from braindecode.classifier import EEGClassifier
from cyclopts import App
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from skorch.helper import predefined_split
from skorch.history import History
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchview import draw_graph

from pretrain_braindecode_models.config import DEVICE, MODELS_DIR
from pretrain_braindecode_models.modeling import models
from pretrain_braindecode_models.modeling.augmentations import build_augmentation_pipeline
from pretrain_braindecode_models.modeling.callbacks import (
    ClassificationAnalysisCallback,
    LoggingCallback,
    PlottingCallback,
)
from pretrain_braindecode_models.modeling.lightning_modules import ClassificationLightningModule
from pretrain_braindecode_models.modeling.plotting import (
    TrainingHistoryClassification,
    plot_confusion_matrix,
    plot_history,
)
from pretrain_braindecode_models.utils.custom_types import (
    ArrayOrTensor,
    DimOrder,
    ScalerProtocol,
)
from pretrain_braindecode_models.utils.loading import save_as_pickle, save_json
from pretrain_braindecode_models.utils.metrics import classification_metrics_callbacks
from pretrain_braindecode_models.utils.misc import (
    as_tensors_on_device,
    convert_skorch_history_to_plotting_format,
    get_init_args,
    get_n_classes,
    is_strictly_increasing,
    process_lightning_logs_to_history,
    reorder_dict_keys,
)
from pretrain_braindecode_models.utils.preprocessing import preprocess_and_scale_data

app = App()


def get_scheduler(
    optimizer: optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int | None = None,
    factor: float = 0.1,
    patience: int | None = 5,
    max_lr: float = 0.01,
    min_lr: float = 0.00001,
    *,
    reduce_lr_on_plateau: bool = False,
    one_cycle_lr: bool = True,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Get a learning rate scheduler based on the provided parameters."""
    # If patience is not defined, set it to num_epochs // 4
    if patience is None:
        logger.warning(
            f"Patience is None. Setting it to num_epochs // 4 = {num_epochs // 4}",
        )
        patience = num_epochs // 4

    # Initialize learning rate scheduler
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    if reduce_lr_on_plateau and one_cycle_lr:
        logger.warning(
            "Both `use_reduce_lr_on_plateau` and `use_one_cycle_lr` are set to True. "
            "Using `use_reduce_lr_on_plateau`.",
        )

    if reduce_lr_on_plateau:
        logger.info(
            f"Using ReduceLROnPlateau with patience {patience} and factor {factor} for lr "
            f"reduction.",
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    elif one_cycle_lr:
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for OneCycleLR.")

        logger.info(
            f"Using OneCycleLR with max_lr {max_lr}, num_epochs {num_epochs} and steps_per_epoch "
            f"{steps_per_epoch} for learning rate cycling.",
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
        )
    else:
        logger.info("No learning rate scheduler specified.")
        scheduler = None

    return scheduler


def get_optimizer_class(optimizer_name: str) -> type[optim.Optimizer]:
    """Get the optimizer class from `torch.optim` by its string name.

    Args:
        optimizer_name (str): The case-insensitive name of the optimizer to use
            (e.g., "Adam", "AdamW", "SGD", "RMSprop").

    Returns:
        (type[optim.Optimizer]): The optimizer class from `torch.optim`.

    Raises:
        ValueError: If the specified optimizer_name is not found in `torch.optim`.
    """
    # --- Dynamically find the optimizer class from `torch.optim` ---
    # We use case-insensitivity for convenience (e.g., "adam" works as well as "Adam")
    try:
        # `getattr` attempts to retrieve an attribute (the optimizer class) by its string name
        optimizer_class = getattr(optim, optimizer_name, None)
        if optimizer_class is None:
            # Try case-insensitive match
            for name, obj in inspect.getmembers(optim, inspect.isclass):
                if issubclass(obj, optim.Optimizer) and name.lower() == optimizer_name.lower():
                    optimizer_class = obj
                    break
        if optimizer_class is None:
            raise ValueError  # Trigger the except block
    except (AttributeError, ValueError):
        # List all available optimizers for a helpful error message
        available_optimizers = [
            name
            for name, obj in inspect.getmembers(optim, inspect.isclass)
            if issubclass(obj, optim.Optimizer) and name != "Optimizer"
        ]
        raise ValueError(
            f"Unsupported optimizer: '{optimizer_name}'. "
            f"Please choose from the available optimizers in `torch.optim`: {available_optimizers}"
        ) from None
    else:
        return optimizer_class


def get_optimizer(
    model_or_params: nn.Module | Iterator[nn.Parameter],
    optimizer_name: str,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> optim.Optimizer:
    """Instantiate a PyTorch optimizer by its string name.

    This function acts as a factory, dynamically finding and creating an
    optimizer from the `torch.optim` module.

    Args:
        model_or_params (nn.Module | Iterator[nn.Parameter]): The PyTorch model (nn.Module) whose
            parameters will be optimized, or an iterator of parameters.
        optimizer_name (str): The case-insensitive name of the optimizer to use
            (e.g., "Adam", "AdamW", "SGD", "RMSprop").
        optimizer_kwargs (dict[str, Any]): A dictionary of keyword arguments to pass to the
            optimizer's constructor (e.g., {'lr': 0.001, 'weight_decay': 1e-5}).
            If None, empty dict will be used.

    Returns:
        (optim.Optimizer): An initialized PyTorch Optimizer instance.

    Raises:
        ValueError: If the specified optimizer_name is not found in `torch.optim`.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # Get the model parameters if a full model is passed
    if isinstance(model_or_params, nn.Module):
        params = model_or_params.parameters()
    else:
        params = model_or_params  # Assume it's already an iterator of parameters

    optimizer_class = get_optimizer_class(optimizer_name)
    logger.info(
        f"Instantiating optimizer '{optimizer_class.__name__}' with parameters: {optimizer_kwargs}"
    )

    # Instantiate the optimizer with the model's parameters and provided kwargs
    return optimizer_class(params, **optimizer_kwargs)


def load_model(
    architecture: Callable[[], nn.Module],
    state_dict_path: Path | str | None = None,
    device: torch.device | str = DEVICE,
    eval_mode: bool = True,  # noqa: FBT001, FBT002
    **kwargs: dict,
) -> nn.Module:
    """Load a model architecture and optionally, a state dictionary.

    Args:
        architecture (Callable): Model architecture function, which returns a model instance
        state_dict_path (Path | str | None): Path to the model state dictionary.
            If None, the model is initialized without loading weights.
        device (torch.device | str): Device to load the model on (default: DEVICE)
        eval_mode (bool): Whether to set the model to evaluation mode (default: True)
        *args: Additional arguments for the `architecture` callable
        **kwargs: Additional arguments for the `architecture` callable

    Returns:
        torch.nn.Module: Loaded model
    """
    model = architecture(**kwargs)

    if state_dict_path is not None:
        logger.info(f"Loading model state dict from {state_dict_path}")
        model.load_state_dict(torch.load(state_dict_path, map_location=device))

    model.to(device)
    if eval_mode:
        model.eval()  # Set to evaluation mode
    else:
        model.train()

    return model


def setup_train_model(
    X_train: ArrayOrTensor,
    y_train: ArrayOrTensor,
    X_test: ArrayOrTensor,
    y_test: ArrayOrTensor,
    *,
    config: dict,
    device: torch.device = DEVICE,
) -> tuple[
    tuple[np.ndarray, np.ndarray, ScalerProtocol | None],
    tuple[nn.Module, type[optim.Optimizer], type[nn.Module]],
    dict,
]:
    """Set up the training process for the given model and return the metadata the run would have.

    Args:
        X_train (torch.Tensor | np.ndarray): Training input data.
        y_train (torch.Tensor | np.ndarray): Training target data.
        X_test (torch.Tensor | np.ndarray): Testing input data.
        y_test (torch.Tensor | np.ndarray): Testing target data.
        config (dict): Configuration dictionary containing training parameters, model parameters,
            dataset parameters, and loss parameters. Example:
            ```
            config = {
                "task": "regression",
                "experiment_name": "my_experiment",
                ...,
            }
            ```
        device (torch.device, optional): Device to run the model on, by default `DEVICE`.

    Returns:
        tuple: A tuple containing:
        - tuple[np.ndarray, np.ndarray, ScalerProtocol | None]: Processed training and
            testing data (X_train, X_test, scaler)
        - tuple[nn.Module, optim.Optimizer | type[optim.Optimizer], nn.Module | type[nn.Module]]:
            Model, optimizer, and criterion to use for training. The optimizer and criterion will
            be classes, not instances.
        - dict: Metadata dictionary with experiment details

    Raises:
        ValueError: If the input data cannot be processed correctly or if the scaler is not
            provided and the data is not in the expected format.
    """
    # Get big chunks of config
    training_params = config.get("training_params", {})
    model_params = config.get("model", {})
    dataset_params = config.get("dataset_params", {})

    # Get specific parameters
    model_class = model_params.get("model_class")
    num_epochs = training_params.get("num_epochs", 100)
    batch_size = training_params.get("batch_size", 32)
    precision = dataset_params.get("precision", "float64")
    dim_order_mapping: dict[str, DimOrder] = {
        "channels_first": "NCT",
        "time_first": "NTC",
    }
    dim_order: DimOrder = dim_order_mapping[
        config["dataset_params"].get("dim_order", "channels_first")
    ]
    optimizer = training_params.get("optimizer", "Adam")
    optimizer_kwargs = training_params.get("optimizer_kwargs", {"lr": 1e-4})

    # Save it in the model config too
    config["training_params"]["num_epochs"] = num_epochs
    config["training_params"]["batch_size"] = batch_size
    config["training_params"]["optimizer"] = optimizer
    config["training_params"]["optimizer_kwargs"] = optimizer_kwargs

    config["dataset_params"]["precision"] = precision

    # Ensure the shape of the input data is correct
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            f"Input data must be 3D tensors (n_epochs, n_channels, n_times). "
            f"Got X_train shape: {X_train.shape}, X_test shape: {X_test.shape}",
        )

    # Instantiate Model
    logger.info("--- Instantiating Model ---")

    # Dynamically determine shape kwargs based on data and config
    if dim_order == "NCT":
        shape_kwargs = {
            "input_features": X_train.shape[1],
            "input_seq_len": X_train.shape[2],
            "n_outputs": get_n_classes(y_train, y_test),
        }
    else:  # NTC
        shape_kwargs = {
            "input_features": X_train.shape[2],
            "input_seq_len": X_train.shape[1],
            "n_outputs": get_n_classes(y_train, y_test),
        }

    config["model"]["shape_kwargs"] = shape_kwargs  # Save the shape kwargs used

    is_tcn = config["model"]["model_kwargs"].get("braindecode_model_name", "").lower() == "tcn"

    # If the task is classification, remove "tcn_pooling_strategy" from the parameters if the
    # model is not TCN, as it is only valid for TCN models
    if not is_tcn and "tcn_pooling_strategy" in config["model"]["model_kwargs"]:
        config["model"]["model_kwargs"]["tcn_pooling_strategy"] = []

    model_params = config.get("model", {})
    model = models.get_model(model_params, data_shape_kwargs=shape_kwargs)

    # Instantiate the scaler
    scaler = models.get_scaler(
        config["training_params"].pop("scaler"),
        config["training_params"].pop("scaler_kwargs", {}),
    )

    logger.info(
        f"Setting up model training on {device} with "
        f"epochs={num_epochs} and batch_size={batch_size}"
    )

    logger.info("Input data for model training:")
    if dim_order == "time_first":
        logger.info(f"  - X_train shape (EEG): {tuple(X_train.shape)} (N, T, C)")
        logger.info(f"  - y_train shape (labels): {tuple(y_train.shape)} (N,)")
        logger.info(f"  - X_test shape (EEG): {tuple(X_test.shape)} (N, T, C)")
        logger.info(f"  - y_test shape (labels): {tuple(y_test.shape)} (N,)")
    else:  # channels_first
        logger.info(f"  - X_train shape (EEG): {tuple(X_train.shape)} (N, C, T)")
        logger.info(f"  - y_train shape (labels): {tuple(y_train.shape)} (N,)")
        logger.info(f"  - X_test shape (EEG): {tuple(X_test.shape)} (N, C, T)")
        logger.info(f"  - y_test shape (labels): {tuple(y_test.shape)} (N,)")

    # Use the scaler for EEG data ONLY if it's provided.
    # e.g., scaler = Scaler(scalings="mean", with_mean=True, with_std=True)
    # It returns NumPy arrays in the correct shape for the model ('NTC').
    X_train_processed, X_test_processed, scaler = preprocess_and_scale_data(
        X_train,
        X_test,
        scaler,
        # Get precision from the dataset metadata, with a safe default
        precision=precision,
        required_input_order="NCT",
        required_output_order=dim_order,
    )

    # Find the optimizer
    my_optimizer = get_optimizer_class(optimizer)

    logger.info(f"Using optimizer: {my_optimizer}")

    # Find the criterion / loss function
    my_criterion = nn.CrossEntropyLoss
    logger.info(f"Using loss function: {my_criterion}")

    # Ensure model is on the correct device
    model.to(device)

    # Create the model
    model_kwargs = get_init_args(model)
    logger.info(f"Model kwargs for class={model_class}: {model_kwargs}")

    # Save metadata of the model
    config["model"]["model_kwargs"] = model_kwargs  # Save the actual kwargs used
    config["model"]["model_class"] = model.__class__.__name__
    config["model"]["model_name"] = model.name if hasattr(model, "name") else None

    metadata = {
        **config,
        "num_params": {
            "total": models.count_parameters(model),
            "trainable": models.count_parameters(model, only_trainable=True),
        },
        "shapes": {
            "input": X_train_processed.shape,
            "output": y_train.shape,
        },
        "criterion": my_criterion,
        "optimizer": my_optimizer,
        "scaler": {
            "scaler_str": scaler,
            "scaler_class": scaler.__class__ if scaler else None,
            "scaler_kwargs": get_init_args(scaler) if scaler else None,
        },
    }

    metadata = reorder_dict_keys(
        metadata,
        [
            "experiment_name",
            "from_config",
            "models_subdir",
            "dataset_params",
            "model",
            "num_params",
            "shapes",
            "criterion",
            "optimizer",
            "scaler",
            "training_params",
        ],
    )

    return (
        (X_train_processed, X_test_processed, scaler),
        (model, my_optimizer, my_criterion),
        metadata,
    )


def setup_classifier_training(
    model: nn.Module,
    X_train: np.ndarray,
    y_expr_train: np.ndarray,
    X_test: np.ndarray,
    y_expr_test: np.ndarray,
    *,
    metadata: dict[str, Any],
    run_dir: Path = MODELS_DIR,
    device: torch.device = DEVICE,
) -> tuple[
    tuple[TensorDataset, TensorDataset],
    tuple[LabelEncoder | None, np.ndarray, np.ndarray, dict[str, Any]],
]:
    """Set up classifier training.

    Args:
        model (nn.Module): The EEGClassifier model to train.
        X_train (np.ndarray): The training input data.
        y_expr_train (np.ndarray): The training expression labels.
        X_test (np.ndarray): The testing input data.
        y_expr_test (np.ndarray): The testing expression labels.
        metadata (dict[str, Any]): Metadata dictionary of training hyperparameters and where to
            store training information.
        run_dir (Path): The directory to save model checkpoints and logs.
        device (torch.device): The device to train the model on.

    Returns:
        tuple: A tuple with:
        - The training and testing datasets
        - The label encoder if needed, unique classes, class weights, and metadata.
    """
    all_y = np.concatenate((np.unique(y_expr_train), np.unique(y_expr_test)))
    unique_classes = np.unique(all_y)

    # If unique_classes is numeric, then it doesn't need a label encoder
    if np.issubdtype(unique_classes.dtype, np.number):
        unique_classes = np.sort(unique_classes)
        no_label_encoder_needed = True
    else:
        no_label_encoder_needed = False

    # Make sure it's sorted
    if not is_strictly_increasing(unique_classes):
        raise ValueError("Classes must be sorted in increasing order.")

    # input_size = (8, 64, int(EEG_WINDOW_SECONDS * sfreq))
    input_size = (
        8,  # Batch size (e.g., 8)
        X_train.shape[1],  # Number of channels (e.g., 64)
        X_train.shape[2],  # Number of time points (e.g., EEG_WINDOW_SECONDS * sfreq)
    )

    logger.info(f"Input size for model graph: {input_size}")

    # Visualize the model graph
    model_graph = draw_graph(
        model,
        graph_name=(
            model.name
            if hasattr(model, "name") and isinstance(model.name, str)
            else model.__class__.__name__
        ),
        input_size=input_size,
        device=device,
        expand_nested=True,
        hide_inner_tensors=True,
        hide_module_functions=False,
        roll=False,
        save_graph=True,
        filename="model_graph",
        directory=str(run_dir / "figures"),
    )
    model_graph.resize_graph(scale=10.0)
    model_graph.visual_graph.render(
        "model_graph", directory=str(run_dir / "figures"), format="svg"
    )
    logger.info(f"Model graph saved to: {run_dir / 'figures' / 'model_graph.png'}")

    # Encode the labels if needed
    le: LabelEncoder | None = None
    if no_label_encoder_needed:
        logger.debug(
            "No LabelEncoder needed, using numeric labels directly "
            f"(unique_classes={unique_classes})"
        )
        unique_classes_enc = unique_classes
        y_train_enc = y_expr_train
        y_test_enc = y_expr_test
    else:
        # Use LabelEncoder to convert string labels to integers
        le = LabelEncoder()
        le.fit(unique_classes)
        y_train_enc = np.asarray(le.transform(y_expr_train))
        y_test_enc = np.asarray(le.transform(y_expr_test))
        unique_classes_enc = np.asarray(le.transform(unique_classes))
        logger.debug(
            f"LabelEncoder classes_: {le.classes_} (from unique_classes: {unique_classes})"
        )

    X_train_t, y_train_t, X_test_t, y_test_t = as_tensors_on_device(
        X_train,
        y_train_enc,
        X_test,
        y_test_enc,
        device="cpu",
        dtype=torch.float32,
    )
    y_train_t, y_test_t = y_train_t.long(), y_test_t.long()

    del X_train, y_train_enc, X_test, y_test_enc  # Free memory

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    class_weights = compute_class_weight(
        "balanced",
        classes=unique_classes_enc,
        y=y_train_t.numpy(),
    )

    logger.debug(f"Computed class weights: {class_weights} for classes: {unique_classes_enc}")

    patience = metadata["training_params"].get("patience")  # Early stopping patience
    # If patience < 1, then it's a fraction of the num_epochs
    if patience is not None and patience <= 1:
        patience *= int(max(1, metadata["training_params"].get("num_epochs", 1)))

    # Update patience in metadata
    metadata["training_params"]["patience"] = patience

    return (train_dataset, test_dataset), (le, unique_classes_enc, class_weights, metadata)


def train_classifier_lightning(
    model: nn.Module,
    X_train: np.ndarray,
    y_expr_train: np.ndarray,
    X_test: np.ndarray,
    y_expr_test: np.ndarray,
    *,
    scaler: ScalerProtocol | None = None,
    class_labels: dict[int, str] | None = None,
    run_dir: Path,
    metadata: dict[str, Any],
    device: torch.device = DEVICE,
) -> None:
    """Set up and train a classifier using PyTorch Lightning."""
    # Save scaler as pkl
    if scaler is not None:
        save_as_pickle(run_dir / "scaler.pkl", scaler)
        logger.info(f"Scaler saved to: {run_dir / 'scaler.pkl'}")

    training_params = metadata.get("training_params", {})

    (train_dataset, test_dataset), (le, classes, class_weights, metadata) = (
        setup_classifier_training(
            model,
            X_train,
            y_expr_train,
            X_test,
            y_expr_test,
            metadata=metadata,
            run_dir=run_dir,
            device=device,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params["batch_size"],
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=training_params["batch_size"],
        pin_memory=True,
        shuffle=False,
    )

    n_classes = len(le.classes_) if le is not None else classes.shape[0]

    # --- Setup Augmentation pipeline ---
    augmentation_params = training_params.get("augmentation_params", None)

    # Build the augmentation pipeline if augmentation parameters are provided
    if augmentation_params and augmentation_params.get("apply", False):
        augmentation_pipeline = build_augmentation_pipeline(augmentation_params)
        logger.info(f"Set-up the following augmentation pipeline: {augmentation_pipeline}")
    else:
        augmentation_pipeline = None

    # --- Setup Lightning Module ---
    optimizer_class = get_optimizer_class(training_params["optimizer"])
    lightning_module = ClassificationLightningModule(
        model=model,
        optimizer_class=optimizer_class,
        optimizer_kwargs=training_params["optimizer_kwargs"],
        n_classes=n_classes,
        class_weights=(
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        ),
        aug_transform=augmentation_pipeline,
    )

    # --- Setup Callbacks ---
    # For example, ["Balanced Acc", "F1", "Precision", "Recall"]
    metrics_to_plot = lightning_module.metrics_list()
    best_model_metric = training_params.get("checkpoint_metric", "val_balanced_accuracy")

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch}-" + lightning_module.model_checkpoint_string(),
        monitor=best_model_metric,
        mode="max" if "loss" not in best_model_metric else "min",
        save_top_k=1,
    )
    plotting_cb = PlottingCallback(
        run_dir=run_dir,
        metrics=metrics_to_plot,
        best_model_metric=best_model_metric,
        plot_kwargs=training_params.get("plot_kwargs", {}),
    )

    if le is not None:
        classes_str = le.classes_.tolist()
    elif class_labels:
        classes_str = [class_labels.get(c, str(c)) for c in classes]
    else:
        classes_str = [str(c) for c in classes]

    analysis_cb = ClassificationAnalysisCallback(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        classes=classes_str,
        run_dir=run_dir,
        monitor_metric=best_model_metric,
        mode="max" if "loss" not in best_model_metric else "min",
    )

    callbacks = [checkpoint_cb, plotting_cb, analysis_cb]
    if training_params.get("use_earlystopping", False):
        patience = metadata["training_params"]["patience"]
        callbacks.append(
            EarlyStopping(
                monitor=training_params.get("earlystopping_metric", "val_balanced_accuracy"),
                patience=patience,
                mode=(
                    "min"
                    if "loss"
                    in training_params.get("earlystopping_metric", "val_balanced_accuracy")
                    else "max"
                ),
            )
        )

    # Add the custom logging callback
    callbacks.append(LoggingCallback())

    # --- Setup Trainer ---
    trainer = pl.Trainer(
        accelerator="gpu" if DEVICE.type == "cuda" else "cpu",
        devices=torch.cuda.device_count(),
        # strategy="ddp_find_unused_parameters_true"
        # if torch.cuda.device_count() > 1 else "auto",
        strategy="ddp",
        # Automatically detect the number of nodes from SLURM environment
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),  # noqa: PLW1508
        max_epochs=training_params["num_epochs"],
        callbacks=callbacks,
        logger=CSVLogger(save_dir=run_dir),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=False,
        default_root_dir=run_dir,
    )

    logger.info("--- Starting Lightning Classifier Training ---")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # --- Post-Training ---
    # The best model checkpoint is automatically tracked by Lightning
    best_model_path = checkpoint_cb.best_model_path
    logger.info(f"Best model saved at: {best_model_path}")

    # --- Process logs and save final metadata ---
    final_history = TrainingHistoryClassification()
    if trainer.logger and trainer.logger.log_dir:
        log_dir = Path(trainer.logger.log_dir)
        logger.info(f"Processing logger history in {log_dir}")
        final_history = process_lightning_logs_to_history(log_dir)

        # Find the best epoch from the processed history
        best_epoch = plotting_cb.best_epoch_idx
        metadata["history"] = final_history.to_dict()

    metadata["history"]["early_stop_epoch"] = best_epoch  # type: ignore[reportArgumentType]

    # Save final metadata (optional, as logger saves metrics.csv)
    # You can load trainer.logger.log_dir / "metrics.csv" to get the full history
    save_json(metadata, run_dir / "metadata.json")
    logger.info(f"Final metadata saved to: {run_dir / 'metadata.json'}")


def train_braindecode_eegclassifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_expr_train: np.ndarray,
    X_test: np.ndarray,
    y_expr_test: np.ndarray,
    *,
    metadata: dict[str, Any],
    training_params: dict[str, Any],
    criterion: type[nn.CrossEntropyLoss] = nn.CrossEntropyLoss,
    optimizer: type[optim.Optimizer] = optim.Adam,
    run_dir: Path = MODELS_DIR,
    device: torch.device = DEVICE,
) -> None:
    """Set up and trains an `EEGClassifier` from `braindecode`.

    It uses `braindecode.classifier.EEGClassifier`.

    Args:
        model (nn.Module): The EEGClassifier model to train.
        X_train (np.ndarray): The training input data.
        y_expr_train (np.ndarray): The training expression labels.
        X_test (np.ndarray): The testing input data.
        y_expr_test (np.ndarray): The testing expression labels.
        metadata (dict[str, Any]): Metadata dictionary to store training information.
        training_params (dict[str, Any]): Dictionary of training hyperparameters.
        criterion (type[nn.CrossEntropyLoss]): The loss function to use (as a class).
        optimizer (type[optim.Optimizer]): The optimizer to use (as a class).
        run_dir (Path): The directory to save model checkpoints and logs.
        device (torch.device): The device to train the model on.
    """
    all_y = np.concatenate((np.unique(y_expr_train), np.unique(y_expr_test)))
    unique_classes = np.unique(all_y)

    # Make sure it's sorted
    if not is_strictly_increasing(unique_classes):
        raise ValueError("Classes must be sorted in increasing order.")

    # input_size = (8, 64, int(EEG_WINDOW_SECONDS * sfreq))
    input_size = (
        8,  # Batch size (e.g., 8)
        X_train.shape[1],  # Number of channels (e.g., 64)
        X_train.shape[2],  # Number of time points (e.g., EEG_WINDOW_SECONDS * sfreq)
    )

    logger.info(f"Input size for model graph: {input_size}")

    # Visualize the model graph
    model_graph = draw_graph(
        model,
        graph_name=(
            model.name
            if hasattr(model, "name") and isinstance(model.name, str)
            else model.__class__.__name__
        ),
        input_size=input_size,
        device=device,
        expand_nested=True,
        hide_inner_tensors=True,
        hide_module_functions=False,
        roll=False,
        save_graph=True,
        filename="model_graph",
        directory=str(run_dir / "figures"),
    )
    model_graph.resize_graph(scale=10.0)
    model_graph.visual_graph.render(
        "model_graph", directory=str(run_dir / "figures"), format="svg"
    )
    logger.info(f"Model graph saved to: {run_dir / 'figures' / 'model_graph.png'}")

    # Encode the labels
    le = LabelEncoder()
    le.fit(unique_classes)
    y_train_enc = np.asarray(le.transform(y_expr_train))
    y_test_enc = np.asarray(le.transform(y_expr_test))
    expression_names = le.classes_

    X_train_t, y_train_t, X_test_t, y_test_t = as_tensors_on_device(
        X_train,
        y_train_enc,
        X_test,
        y_test_enc,
        device="cpu",
        dtype=torch.float32,
    )
    y_train_t, y_test_t = y_train_t.long(), y_test_t.long()

    del X_train, y_train_enc, X_test, y_test_enc  # Free memory

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(np.concatenate((y_train_t.numpy(), y_test_t.numpy()))),
        y=y_train_t.numpy(),
    )

    patience = training_params.get("patience")  # Early stopping patience
    # If patience < 1, then it's a fraction of the num_epochs
    if patience is not None and patience <= 1:
        patience *= int(max(1, training_params.get("num_epochs", 1)))

    # Update patience in metadata
    metadata["patience"] = patience

    callbacks = classification_metrics_callbacks(
        patience=patience,
        earlystopping_metric=training_params.get("earlystopping_metric"),
        checkpoint_metric=training_params.get("checkpoint_metric"),
        checkpoint_dir=run_dir / "checkpoint",
        checkpoint_cb=True,
        earlystopping_cb=training_params.get("use_earlystopping", False),
    )

    # Construct optimizer__lr, optimizer__weight_decay, etc. based on training_params
    optimizer_kwargs = {}
    for key, value in training_params.get("optimizer_kwargs", {}).items():
        new_key = f"optimizer__{key}"
        optimizer_kwargs[new_key] = value

    clf = EEGClassifier(
        model,
        criterion=criterion,
        criterion__weight=torch.Tensor(class_weights).to(device),
        iterator_train__shuffle=False,
        iterator_valid__shuffle=False,
        train_split=predefined_split(test_dataset),
        batch_size=training_params["batch_size"],
        callbacks=callbacks,
        device=device,
        classes=unique_classes,
        optimizer=optimizer,
        **optimizer_kwargs,
    )

    logger.info("--- Starting Classifier Training ---")
    clf.fit(train_dataset, y=None, epochs=training_params["num_epochs"])

    # -- Force loading of checkpoint / best_model_params.pt
    if (run_dir / "checkpoint" / "best_model_params.pt").exists():
        best_state_dict = torch.load(run_dir / "checkpoint" / "best_model_params.pt")
        clf.module_.load_state_dict(best_state_dict)
        model.load_state_dict(best_state_dict)

    # -- Set the model to evaluation mode --
    clf.module_.eval()

    # -- Re-load history from the full_history.json (if it exists) --
    full_history_path = run_dir / "checkpoint" / "full_history.json"
    if full_history_path.exists():
        logger.info(f"Loading full training history from: {full_history_path}")
        clf.history = History.from_file(full_history_path)

    # --- Post-Training Analysis ---
    df = pd.DataFrame(clf.history.to_list())

    history_for_plotting, metrics_to_plot, best_epoch = convert_skorch_history_to_plotting_format(
        history_df=df,
        best_metric=training_params.get("checkpoint_metric", "valid_bal_acc"),
        metric_prefix_map={
            "bal_acc": "Balanced Acuracy",
            "f1": "F1 Score",
            "precision": "Precision",
            "recall": "Recall",
        },
    )

    metadata["history"] = history_for_plotting
    metadata["history"]["early_stop_epoch"] = best_epoch  # type: ignore[reportArgumentType]
    save_json(metadata, run_dir / "metadata.json")

    plot_history(
        history=history_for_plotting,  # type: ignore[reportArgumentType, arg-type]
        metrics=metrics_to_plot,
        filepath=str(run_dir / "classification_history.png"),
        overall_best_epoch=best_epoch,
        # figtitle=f"{metadata.get('experiment_name', '')} Performance",
        show=False,
        **training_params.get("plot_kwargs", {}),
    )

    plot_confusion_matrix(
        clf,
        train_dataset,
        classes=expression_names.tolist(),
        filepath=str(run_dir / "train_confusion_matrix.png"),
        figsize=(30, 30),
        dpi=300,
        show=False,
    )
    plot_confusion_matrix(
        clf,
        test_dataset,
        classes=expression_names.tolist(),
        filepath=str(run_dir / "test_confusion_matrix.png"),
        figsize=(30, 30),
        dpi=300,
        show=False,
    )

    save_train_test_data = training_params.get("save_train_test_data", True)
    if save_train_test_data:
        logger.info("Saving train/test data used for training the classifier.")
        save_as_pickle(
            run_dir / "train_test_data.pkl",
            {
                "X_train": X_train_t.numpy(),
                "y_expr_train": y_train_t.numpy(),
                "X_test": X_test_t.numpy(),
                "y_expr_test": y_test_t.numpy(),
                "label_encoder": le,
            },
        )

    logger.success(f"Classifier and results saved to {run_dir}")


if __name__ == "__main__":
    app()
