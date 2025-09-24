"""Utility functions for preprocessing EEG data from multiple recordings."""

from pathlib import Path
from typing import Any, Literal, overload

import mne
import numpy as np
import psutil
import torch
from prettytable import PrettyTable
from sklearn.utils.validation import check_is_fitted

from pretrain_braindecode_models.config import DEVICE, logger
from pretrain_braindecode_models.modeling.models import get_scaler
from pretrain_braindecode_models.utils.colors import color_text, create_custom_colormap
from pretrain_braindecode_models.utils.custom_types import (
    ArrayOrTensor,
    MNEOrScikitScaler,
    ScalerProtocol,
    ScikitScaler,
)
from pretrain_braindecode_models.utils.folders import format_size_bytes
from pretrain_braindecode_models.utils.misc import (
    DimOrder,
    as_numpy_arrays,
    as_tensors_on_device,
    check_for_nan_or_inf,
    reorder_dims,
)


class MNEICA:
    """A wrapper for `mne.preprocessing.ICA`.

    It fits on training data and transforms both train and test sets.
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the MNEICA preprocessor.

        Args:
            **kwargs: Keyword arguments to be passed directly to the mne.preprocessing.ICA
                constructor, e.g., n_components=0.95, random_state=42
        """
        self.ica = mne.preprocessing.ICA(**kwargs)
        self.sfreq: float | None = None

    def fit_transform(self, X_continuous: np.ndarray, sfreq: float) -> np.ndarray:
        """Fit the ICA model on the continuous data and apply the transformation.

        Args:
            X_continuous (np.ndarray): Continuous EEG data with shape (channels, time).
            sfreq (float): The sampling frequency of the EEG data.

        Returns:
            np.ndarray: The cleaned data after ICA component removal.
        """
        n_channels, _ = X_continuous.shape
        logger.debug(f"Fitting and applying ICA on data of shape {X_continuous.shape}...")

        # MNE requires an Info object.
        # Create a dummy list of channel names for the info object
        ch_names = [f"EEG {i:03}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Create a RawArray object to use with ICA
        raw_data = mne.io.RawArray(X_continuous, info, verbose=False)

        # MNE requires high-pass filtering for ICA stability
        raw_data.filter(l_freq=1.0, h_freq=None, verbose=False)

        # Fit the ICA model
        self.ica.fit(raw_data, verbose=False)

        # pca_explained_variance_ndarray, shape (n_channels,)
        # If fit, the variance explained by each PCA component.
        logger.debug("ICA fitting complete")

        # Apply the ICA to remove artifacts (modifies raw_data in-place)
        self.ica.apply(raw_data, verbose=False)

        # Return the cleaned data
        cleaned = raw_data.get_data(return_times=False)
        if not isinstance(cleaned, np.ndarray):
            raise TypeError("Expected cleaned data to be a numpy array.")

        return cleaned


def save_preprocessor(
    processor: mne.preprocessing.ICA,
    filename: str,
    directory: Path,
) -> None:
    """Save a fitted preprocessor to disk.

    Args:
        processor (mne.preprocessing.ICA): The fitted ICA processor to save.
        filename (str): The name of the file to save the processor to. For instance,
            `f"{subject}_{name.lower()}-ica.fif"`.
        directory (Path): The directory to save the processor in.
    """
    # MNE ICA objects can be saved with their own method
    if isinstance(processor, mne.preprocessing.ICA):
        save_path = directory / filename
        processor.save(save_path, overwrite=True)
    else:
        raise TypeError(f"Unsupported preprocessor type to save: {type(processor)}")


def preprocess_and_scale_data(
    X_train: ArrayOrTensor,
    X_test: ArrayOrTensor,
    scaler: ScalerProtocol | None = None,
    *,
    precision: Literal["float64", "float32", "float16"] = "float32",
    required_input_order: DimOrder = "NCT",
    required_output_order: DimOrder = "NCT",
) -> tuple[np.ndarray, np.ndarray, ScalerProtocol | None]:
    """Preprocesses and scales training and testing data, handling type and dimension order.

    This function ensures data is in NumPy format, reorders dimensions if necessary
    to match the scaler's expectation, applies the scaler, and returns the
    processed NumPy arrays ready for final conversion to Tensors.

    Args:
        X_train (ArrayOrTensor): The raw training data (NumPy array or PyTorch Tensor).
        X_test (ArrayOrTensor): The raw testing data (NumPy array or PyTorch Tensor).
        scaler (ScalerProtocol | None): An optional scaler object (e.g., MNE Scaler) that
            implements `fit_transform` and `transform`. If None, data is returned unscaled.
        precision (Literal["float64", "float32", "float16"]): The precision to use for the
            final output arrays. Defaults to "float32".
        required_input_order (str): The dimension order required by the scaler, e.g.,
            "NCT" for (epochs, channels, time) for MNE's Scaler. Defaults to "NCT".
        required_output_order (str): The dimension order expected by the model, e.g.,
            "NTC" for (epochs, time, channels). Defaults to "NCT".

    Returns:
        (tuple[np.ndarray, np.ndarray, ScalerProtocol | None]): A tuple of
            `(X_train_processed, X_test_processed, scaler)` where:
            - X_train_processed (np.ndarray): The processed training data.
            - X_test_processed (np.ndarray): The processed testing data.
            - scaler (ScalerProtocol | None): The scaler used for preprocessing, or None if no
            scaler was used.
    """
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            f"Input data must be 3D. Got X_train shape: {X_train.shape}, "
            f"X_test shape: {X_test.shape}"
        )

    # Check if there are inf or nan values in X_train or X_test
    check_for_nan_or_inf(X_train, X_test)

    # 1. Ensure data is in NumPy format for processing
    X_train_np, X_test_np = as_numpy_arrays(X_train, X_test)

    # 2. Check and potentially reorder dimensions for the scaler
    # We assume the input data is (epochs, time, channels) -> "NTC" by default
    current_order: DimOrder = "NTC"

    # Heuristic check if data might already be in the target format
    # MNE Scaler expects (n_epochs, n_channels, n_times), where n_channels < n_times
    if X_train_np.shape[1] < X_train_np.shape[2]:
        # This looks like (N, C, T) already
        current_order = "NCT"
        logger.debug(
            f"Inferred current data order as {current_order} based on shape {X_train_np.shape}"
        )
    else:
        logger.debug(
            f"Inferred current data order as {current_order} based on shape {X_train_np.shape}"
        )

    # Reorder if the current format doesn't match what the scaler requires
    X_train_reordered = reorder_dims(X_train_np, current_order, required_input_order)
    X_test_reordered = reorder_dims(X_test_np, current_order, required_input_order)

    # 3. Apply the scaler if provided
    if scaler:
        logger.info(
            f"Fitting and transforming data with scaler. "
            f"Input shape to scaler: {X_train_reordered.shape}"
        )
        X_train_processed = scaler.fit_transform(X_train_reordered)
        X_test_processed_none = scaler.transform(X_test_reordered)

        if X_test_processed_none is None:
            raise ValueError("X_test is None after scaling. Check the scaler and input data.")
        X_test_processed = X_test_processed_none
    else:
        logger.warning("No scaler provided, data will be used unscaled.")
        X_train_processed = X_train_reordered
        X_test_processed = X_test_reordered

    # 4. Reorder the processed data back to the model's expected format (e.g., "NTC")
    if required_output_order != required_input_order:
        X_train_final = reorder_dims(
            X_train_processed, required_input_order, required_output_order
        )
        X_test_final = reorder_dims(X_test_processed, required_input_order, required_output_order)
    else:
        X_train_final = X_train_processed
        X_test_final = X_test_processed

    # Cast to the final target precision AFTER scaling
    final_dtype = np.dtype(precision)
    if X_train_final.dtype != final_dtype:
        logger.info(f"Casting processed data to final precision: {precision}")
        X_train_final = X_train_final.astype(final_dtype)
        X_test_final = X_test_final.astype(final_dtype)

    logger.success(f"Data preprocessed successfully. Final shape for model: {X_train_final.shape}")

    return X_train_final, X_test_final, scaler


def scaler_fit(
    data: ArrayOrTensor,
    *,
    scaler_name: str,
    scaler_kwargs: dict[str, Any] = {},
    param_shapes: dict[str, int] | None = None,
) -> tuple[np.ndarray, MNEOrScikitScaler | dict[str, MNEOrScikitScaler]]:
    """Fit a scaler to the output data, optionally per parameter group.

    Args:
        data (ArrayOrTensor): The data to scale. Shape: `(N, C, T)`.
        scaler_name (str): The name of the scaler to use. Supported options are:
            - "standard": `sklearn.preprocessing.StandardScaler`
            - "minmax": `sklearn.preprocessing.MinMaxScaler`
            - "robust": `sklearn.preprocessing.RobustScaler`
            - "maxabs": `sklearn.preprocessing.MaxAbsScaler`
            - "mne": `mne.preprocessing.scaler.Scaler`
        scaler_kwargs (dict[str, Any]): Additional keyword arguments to pass to the scaler
            constructor.
        param_shapes (dict[str, int] | None): A dictionary mapping each parameter group
            to its corresponding shape, e.g., `{"exp": 100, "eyes": 12, ...}`. The sum of all
            sizes must match the number of channels in the data. If provided, a separate scaler
            will be fitted for each parameter group. If None, a single scaler will be used
            for all channels.

    Returns:
        (tuple[np.ndarray, MNEOrScikitScaler | dict[str, MNEOrScikitScaler]]): A tuple of:
            - The scaled data as a NumPy array.
            - The fitted scaler(s). If `param_shapes` is provided, this will be a
              dictionary mapping each parameter group to its fitted scaler, e.g.,
              `{"exp": StandardScaler(), "eyes": StandardScaler(), ...}`. If no `param_shapes`
              is provided, this will be a single scaler instance.
    """
    data_np = as_numpy_arrays(data)  # data is a single tensor

    if param_shapes is not None and isinstance(param_shapes, dict):
        y_scaler = {}
        data_scaled_components = []

        start_idx = 0
        for param_name, size in param_shapes.items():
            # Get the corresponding slice of data
            end_idx = start_idx + size
            component_slice = slice(start_idx, end_idx)
            component_data_nct = data_np[:, component_slice, :]

            scaler = get_scaler(scaler_name, scaler_kwargs)  # Get the scaler
            logger.debug(
                f"Applying {scaler_name} scaler to component '{param_name}' with "
                f"slice {component_slice} and shape {component_data_nct.shape}",
            )

            # TODO: Make this compatible with ScalerProtocol
            comp_scaled, scaler = scaler_fit_transform(component_data_nct, scaler=scaler)  # type: ignore[reportArgumentType]

            # Store the scaler and the scaled component
            y_scaler[param_name] = scaler
            data_scaled_components.append(comp_scaled)
            start_idx = end_idx

        # Concatenate along feature axis
        data_final = np.concatenate(data_scaled_components, axis=1)
        logger.debug(
            f"Normalization of y complete: "
            f"{[w.shape for w in data_scaled_components]} -> {data_final.shape}"
        )
    else:
        scaler = get_scaler(scaler_name, scaler_kwargs)  # Get the scaler
        scaler.fit(data_np)  # Scaler works with (samples, features, times)
        data_final = scaler.transform(data_np)
        y_scaler = scaler

    if data_final is None:
        raise ValueError("Scaled data is None. Check the scaler and input data.")

    return data_final, y_scaler


def scaler_inverse_transform(
    data: torch.Tensor,
    scaler: ScalerProtocol | dict[str, ScalerProtocol],
    *,
    param_shapes: dict[str, int] | None = None,
    device: torch.device = DEVICE,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Apply the inverse transformation of a fitted scaler to outputs and targets.

    Args:
        data (torch.Tensor): The data to inverse transform. Shape: `(N, C, T)`.
        scaler (ScalerProtocol | dict[str, ScalerProtocol]): The fitted scaler(s) used for
            preprocessing. If a dictionary is provided, it should map each parameter group to its
            corresponding scaler, e.g., `{"exp": StandardScaler(), "eyes": StandardScaler(), ...}`.
            If a single scaler is provided, it will be used for all target data.
            The scaler must implement `fit_transform`, `transform`, and `inverse_transform`
            methods, e.g., `Scaler(scalings="mean", with_mean=True)` or `StandardScaler()`.
        param_shapes (dict[str, int] | None): A dictionary mapping each parameter group
            to its shape. If a dictionary is provided, it should map a parameter group to its
            corresponding size, e.g., `{"exp": 100, "eyes": 12, ...}`. The sum of all sizes must
            match the number of channels in the data. This is required if `scaler` is a dictionary.
            The keys must match those in `scaler`.
        device (torch.device): The PyTorch device to place the output tensors on.
        target_dtype (torch.dtype): The desired dtype for the output tensors.

    Returns:
        torch.Tensor: The inverse transformed data as a PyTorch tensor on the specified device
    """
    data_np = as_numpy_arrays(data)  # data is a single tensor

    if isinstance(scaler, dict):
        if not param_shapes:
            raise ValueError("param_shapes must be provided for per-component scaling.")

        if set(scaler.keys()) != set(param_shapes.keys()):
            raise ValueError("Keys of scaler and param_shapes must match.")

        # Data is NCT: (N, C, T)
        denorm_components = []
        start_idx = 0
        for param_name, param_size in param_shapes.items():
            end_idx = start_idx + param_size
            component_scaler = scaler[param_name]
            component_data = data_np[:, start_idx:end_idx, :]

            if isinstance(component_scaler, ScikitScaler):
                comp_denorm = scaler2d_inverse_transform_3d(
                    data_nct=component_data,
                    scaler=component_scaler,
                )
            else:  # MNE or other scaler implementing `inverse_transform`
                comp_denorm = component_scaler.inverse_transform(component_data)

            denorm_components.append(comp_denorm)
            start_idx = end_idx

        data_denorm = np.concatenate(denorm_components, axis=1)

    else:  # Single global scaler
        if isinstance(scaler, ScikitScaler):
            data_denorm = scaler2d_inverse_transform_3d(
                data_nct=data_np,
                scaler=scaler,
            )
        else:  # MNE or other scaler implementing `inverse_transform`
            data_denorm = scaler.inverse_transform(data_np)

    return as_tensors_on_device(data_denorm, device=device, dtype=target_dtype)


def scaler2d_inverse_transform_3d(
    data_nct: np.ndarray,
    scaler: ScalerProtocol,
) -> np.ndarray:
    """Apply the inverse transformation of a 2D scaler to 3D data by transposing and reshaping.

    The StandardScaler/MinMaxScaler/MaxAbsScaler/RobustScaler from scikit-learn expects a 2D array
    of shape `(n_samples, n_features)`.In this context, a "sample" for the scaler is a single
    snapshot in time across all channels. Thus, this function transposes the data to bring the
    feature dimension (`n_features`) to the last position before it can safely be reshaped to
    flatten the other dimensions `(n_samples, n_times)` for the scaler to receive.

    In code,
    ```
    >>> data_nct: (n_epochs, n_channels, n_times)
    >>> data_ntc = data_nct.transpose(0, 2, 1)  # (n_epochs, n_times, n_channels)
    >>> data_reshaped = data_ntc.reshape(n_epochs * n_times, n_channels)  # (n_epochs * n_times, n_channels)
    >>> inv_scaled_reshaped = scaler.inverse_transform(data_reshaped)  # (n_epochs * n_times, n_channels)
    >>> inv_scaled_ntc = inv_scaled_reshaped.reshape(n_epochs, n_times, n_channels)  # (n_epochs, n_times, n_channels)
    >>> inv_scaled_nct = inv_scaled_ntc.transpose(0, 2, 1)  # (n_epochs, n_channels, n_times) # Final output
    ```

    Args:
        data_nct (np.ndarray): Input data of shape `(n_samples, n_features, n_times)`, i.e.,
            in `"NCT"` format.
        scaler (ScalerProtocol): A scaler object that implements `inverse_transform` and is fitted.
            Can be sklearn's StandardScaler/MinMaxScaler/MaxAbsScaler/RobustScaler.

    Returns:
        np.ndarray: The inverse scaled data of the same shape as input
            `(n_samples, n_features, n_times)`.
    """  # noqa: E501
    if isinstance(scaler, ScikitScaler):
        # StandardScaler needs (n_samples, n_features), so we merge n_samples and n_times
        n_samples, n_features, n_times = data_nct.shape
        data_reshaped = data_nct.transpose(0, 2, 1).reshape(n_samples * n_times, n_features)
        return (
            scaler.inverse_transform(data_reshaped)
            .reshape(n_samples, n_times, n_features)
            .transpose(0, 2, 1)
        )

    raise NotImplementedError(
        "scaler2d_inverse_transform_3d only supports sklearn's StandardScaler/MinMaxScaler "
        "currently."
    )


@overload
def scaler2d_transform_3d(
    data_nct: np.ndarray,
    scaler: ScalerProtocol,
    *,
    fit: Literal[False] = False,
) -> np.ndarray: ...


@overload
def scaler2d_transform_3d(
    data_nct: np.ndarray,
    scaler: ScalerProtocol,
    *,
    fit: Literal[True],
) -> tuple[np.ndarray, ScalerProtocol]: ...


def scaler2d_transform_3d(
    data_nct: np.ndarray,
    scaler: ScalerProtocol,
    *,
    fit: bool = False,
) -> np.ndarray | tuple[np.ndarray, ScalerProtocol]:
    """Apply a 2D scaler to 3D data by transposing and reshaping.

    The StandardScaler/MinMaxScaler/MaxAbsScaler/RobustScaler from scikit-learn expects a 2D array
    of shape `(n_samples, n_features)`. In this context, a "sample" for the scaler is a single
    snapshot in time across all channels. Thus, this function transposes the data to bring the
    feature dimension (`n_features`) to the last position before it can safely be reshaped to
    flatten the other dimensions `(n_samples, n_times)` for StandardScaler/MinMaxScaler to receive.

    In code,
    ```
    >>> data_nct: (n_samples, n_features, n_times)
    >>> data_ntc = data_nct.transpose(0, 2, 1)  # (n_samples, n_times, n_features)
    >>> data_reshaped = data_ntc.reshape(n_samples * n_times, n_features)  # (n_samples * n_times, n_features)
    >>> scaled_reshaped = scaler.transform(data_reshaped)  # (n_samples * n_times, n_features)
    >>> scaled_ntc = scaled_reshaped.reshape(n_samples, n_times, n_features)  # (n_epochs, n_times, n_features)
    >>> scaled_nct = scaled_ntc.transpose(0, 2, 1)  # (n_samples, n_features, n_times) # Final output
    ```

    Args:
        data_nct (np.ndarray): Input data of shape `(n_samples, n_features, n_times)`, i.e.,
            in `"NCT"` format.
        scaler (ScalerProtocol): A scaler object that implements `transform` and is fitted
            if `fit=False`. If it's not fitted and `fit=True`, it will be fitted to the data.
            Can be sklearn's StandardScaler/MinMaxScaler/MaxAbsScaler/RobustScaler.
        fit (bool): Whether to fit the scaler to the data before transforming.
            If True, the scaler will be fitted to `data_nct` before transforming.
            Default is False.

    Returns:
        (np.ndarray | tuple[np.ndarray, ScalerProtocol]): The scaled data of the same shape as
            input `(n_samples, n_features, n_times)` or a tuple containing that and the fitted
            scaler (if `fit=True`, otherwise the original scaler).

    Raises:
        NotFittedError: If the given `scaler` is not fitted and `fit=False`.
    """  # noqa: E501
    if isinstance(scaler, ScikitScaler):
        # StandardScaler needs (n_samples, n_features), so we merge n_samples and n_times
        n_samples, n_features, n_times = data_nct.shape
        data_reshaped = data_nct.transpose(0, 2, 1).reshape(n_samples * n_times, n_features)

        if fit:
            scaler.fit(data_reshaped)

        check_is_fitted(scaler)  # Ensure the scaler is fitted
        scaled_data = scaler.transform(data_reshaped)

        if scaled_data is None:
            raise ValueError("Data is None after scaling. Check the scaler and input data.")

        # Reshape back to (n_samples, n_features, n_times)
        scaled_reshaped = scaled_data.reshape(n_samples, n_times, n_features).transpose(0, 2, 1)

        # Return the fitted scaler if requested
        if fit:
            return scaled_reshaped, scaler
        return scaled_reshaped

    raise NotImplementedError(
        "scaler2d_transform_3d only supports sklearn's StandardScaler/MinMaxScaler currently."
    )


def scaler_fit_transform(
    data_nct: np.ndarray,
    scaler: ScalerProtocol,
) -> tuple[np.ndarray, ScalerProtocol]:
    """Fit and transform data with a scaler.

    Args:
        data_nct (np.ndarray): Input data of shape `(n_samples, n_features, n_times)`, i.e.,
            in `"NCT"` format.
        scaler (ScalerProtocol): A scaler object that implements `fit_transform`.
            Can be MNE Scaler or sklearn's StandardScaler/MinMaxScaler/MaxAbsScaler/RobustScaler.

    Returns:
        (tuple[np.ndarray, ScalerProtocol]): A tuple of:
            - The scaled data of the same shape as input `(n_samples, n_features, n_times)`.
            - The fitted scaler.
    """
    if isinstance(scaler, ScikitScaler):
        # StandardScaler/MinMaxScaler needs (n_samples, n_features),
        # so we merge n_samples and n_times
        data_scaled_nct, scaler = scaler2d_transform_3d(
            data_nct=data_nct,
            scaler=scaler,
            fit=True,
        )
    else:
        # MNE or other scaler implementing `fit_transform`
        # MNE Scaler works directly with (n_samples, n_features, n_times)
        data_scaled_nct = scaler.fit_transform(data_nct)
    return data_scaled_nct, scaler
