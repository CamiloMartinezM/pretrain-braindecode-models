"""Builds data augmentation pipelines using the torcheeg library."""

import inspect
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.fft import fft, ifft
from torch.nn import functional as F  # noqa: N812
from torch.nn.functional import pad
from torcheeg import transforms
from torcheeg.transforms.torch import RandomEEGTransform

# --- GPU-Compatible Re-implementations of torcheeg.transforms ---


class RandomWindowSlice(RandomEEGTransform):
    """GPU-compatible version of [RandomWindowSlice](https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.transforms.RandomWindowSlice.html#torcheeg.transforms.RandomWindowSlice).

    Modified from: torcheeg/transforms/torch/random.py

    Randomly applies a slice transformation with a given probability, where the original time
    series is sliced by a window, and the sliced data is scaled to the original size (if
    `keep_input_shape` is True). It is worth noting that the random position where each channel
    slice starts is the same.

    .. code-block:: python

        transform = RandomWindowSlice()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomWindowSlice(window_size=100)
        transform(eeg=torch.randn(1, 32, 128))['eeg'].shape
        >>> (1, 32, 128)

        transform = RandomWindowSlice(p=1.0, series_dim=0)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        window_size (int): The window size of the slice, the original signal will be sliced to the
            window_size size, and then adaptively scaled to the input shape.
        window_size_ratio (float): The ratio of the window size to the time series length. The
            window size is calculated by multiplying the `window_size_ratio` with the time series
            length. This is used if `window_size` is not provided. Should be between 0.0 and 1.0.
            (default: :obj:`0.5`)
        series_dim (int): Dimension of the time series in the input tensor. (default: :obj:`-1`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between
            0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that
            masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline (bool): Whether to act on the baseline signal at the same time, if
            the baseline is passed in when calling. (default: :obj:`False`)
        keep_input_shape (bool): Whether to keep the input shape. If True, the sliced window will
            be scaled back to the original time series length with
            `torch.nn.functional.interpolate` with `mode='linear'`. If False, the output will have
            the same shape as the sliced window. (default: :obj:`True`)

    .. automethod:: __call__
    """

    def __init__(
        self,
        window_size_ratio: float = 0.5,
        window_size: int | None = None,
        series_dim: int = -1,
        p: float = 0.5,
        apply_to_baseline: bool = False,  # noqa: FBT001, FBT002
        keep_input_shape: bool = True,  # noqa: FBT001, FBT002
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Initialize RandomWindowSlice."""
        super().__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.series_dim = series_dim
        self.window_size = window_size
        self.window_size_ratio = window_size_ratio
        self.keep_input_shape = keep_input_shape

    def __call__(
        self,
        *args,  # noqa: ANN002
        eeg: torch.Tensor,
        baseline: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Apply a random window slicing with a specified probability.

        Args:
            *args: Additional positional arguments.
            eeg (torch.Tensor): The input EEG signal.
            baseline (torch.Tensor, optional): The corresponding baseline signal, if
                `apply_to_baseline` is set to True and baseline is passed, the baseline signal
                will be transformed with the same way as the experimental signal.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output EEG signal after applying a random window slicing.
        """
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        """Randomly slices a window and scales it to the original size."""
        if not (-len(eeg.shape) <= self.series_dim < len(eeg.shape)):
            raise ValueError(
                f"series_dim should be in range of [{-len(eeg.shape)}, {len(eeg.shape)})."
            )

        if len(eeg.shape) != 2 and len(eeg.shape) != 3:
            raise ValueError(
                "Window warping is only supported on 2D arrays or 3D arrays. For both cases, "
                f"series_dim={self.series_dim} should be the time dimension."
            )

        if self.series_dim < 0:
            self.series_dim = len(eeg.shape) + self.series_dim

        num_timesteps = eeg.shape[self.series_dim]
        if self.series_dim != (len(eeg.shape) - 1):
            transpose_dims = list(range(len(eeg.shape)))
            transpose_dims.pop(self.series_dim)
            transpose_dims = [*transpose_dims, self.series_dim]
            # Based on transpose_dims, create undo_transpose_dims
            undo_transpose_dims = [0] * len(eeg.shape)
            for i, dim in enumerate(transpose_dims):
                undo_transpose_dims[dim] = i
            eeg = eeg.permute(*transpose_dims)
        else:
            undo_transpose_dims = []

        # Use absolute window size if provided, else relative
        window_size = self.window_size or int(num_timesteps * self.window_size_ratio)
        if window_size >= num_timesteps:
            return eeg

        start = int(torch.randint(0, num_timesteps - window_size, (1,)).item())
        sliced_eeg = eeg[..., start : start + window_size]

        # Interpolate back to original size. Needs to be 3D+ for interpolate.
        is_2d = eeg.dim() == 2
        if is_2d:
            sliced_eeg = sliced_eeg.unsqueeze(0)  # Add a batch dim

        if self.keep_input_shape:
            rescaled_eeg = F.interpolate(
                sliced_eeg,
                size=num_timesteps,
                mode="linear",
                align_corners=False,
            )
        else:
            rescaled_eeg = sliced_eeg

        # If it was 2d, squeeze back
        if is_2d:
            rescaled_eeg = rescaled_eeg.squeeze(0)

        if self.series_dim != (len(eeg.shape) - 1):
            rescaled_eeg = rescaled_eeg.permute(*undo_transpose_dims)

        return rescaled_eeg

    @property
    def repr_body(self) -> dict:
        """Return a dictionary of the class attributes for representation."""
        return dict(
            super().repr_body,
            window_size=self.window_size,
            series_dim=self.series_dim,
        )


class RandomWindowWarp(RandomEEGTransform):
    """GPU-compatible version of [RandomWindowWarp](https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.transforms.RandomWindowWarp.html#torcheeg.transforms.RandomWindowWarp).

    Modified from: torcheeg/transforms/torch/random.py

    Apply the window warping with a given probability, where a part of time series data is warped
    by speeding it up or down.

    .. code-block:: python

        transform = RandomWindowWarp()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomWindowWarp(window_size=24, warp_size=48)
        transform(eeg=torch.randn(1, 32, 128))['eeg'].shape
        >>> (1, 32, 128)

        transform = RandomWindowWarp(p=1.0, series_dim=0)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        window_size_ratio (float): The ratio of the window size to the time series length. The
            window size is calculated by multiplying the `window_size_ratio` with the time series
            length. This is used if `window_size` is not provided. Should be between 0.0 and 1.0.
            (default: :obj:`0.5`)
        warp_size_ratio (float): The ratio of the warp size to the window size. The warp size is
            calculated by multiplying the `warp_size_ratio` with the window size. Naturally,
            if `warp_size_ratio` is larger than 1.0, it means slowing down, and if
            `warp_size_ratio` is smaller than 1.0, it means speeding up. This is used if
            `warp_size` is not provided. Should be larger than 0.0. (default: :obj:`2.0`)
        window_size (int): Randomly pick a window of size window_size on the time series to
            transform. (default: :obj:`-1`)
        warp_size (int): The size of the window after the warp. If warp_size is larger than
            window_size, it means slowing down, and if warp_size is smaller than window_size,
            it means speeding up. (default: :obj:`24`)
        series_dim (int): Dimension of the time series in the input tensor. (default: :obj:`-1`)
        p (float): Probability of applying random mask on EEG signal samples.
            Should be between 0.0 and 1.0, where 0.0 means no mask is applied to every sample and
            1.0 means that masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the
            baseline is passed in when calling. (default: :obj:`False`)
        keep_input_shape (bool): Whether to keep the input shape. If True, the warped signal will
            be scaled back to the original time series length with
            `torch.nn.functional.interpolate` with `mode='linear'`. If False, the output will have
            the shape of the warped signal. (default: :obj:`True`)

    .. automethod:: __call__
    """

    def __init__(
        self,
        window_size_ratio: float = 0.5,
        warp_size_ratio: float = 2.0,
        window_size: int | None = None,
        warp_size: int | None = None,
        series_dim: int = -1,
        p: float = 0.5,
        apply_to_baseline: bool = False,  # noqa: FBT001, FBT002
        keep_input_shape: bool = True,  # noqa: FBT001, FBT002
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Initialize RandomWindowWarp."""
        super().__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.window_size = window_size
        self.warp_size = warp_size
        self.window_size_ratio = window_size_ratio
        self.warp_size_ratio = warp_size_ratio
        self.series_dim = series_dim
        self.keep_input_shape = keep_input_shape

    def __call__(
        self,
        *args,  # noqa: ANN002
        eeg: torch.Tensor,
        baseline: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Apply a random window warping with a specified probability.

        Args:
            *args: Additional positional arguments.
            eeg (torch.Tensor): The input EEG signal.
            baseline (torch.Tensor, optional): The corresponding baseline signal, if
                apply_to_baseline is set to True and baseline is passed, the baseline signal
                will be transformed with the same way as the experimental signal.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output EEG signal after applying a random window warping.
        """
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        """Randomly warps a window of the time series by speeding it up or slowing it down."""
        if not (-len(eeg.shape) <= self.series_dim < len(eeg.shape)):
            raise ValueError(
                f"series_dim should be in range of [{-len(eeg.shape)}, {len(eeg.shape)})."
            )

        if len(eeg.shape) != 2 and len(eeg.shape) != 3:
            raise ValueError(
                "Window warping is only supported on 2D arrays or 3D arrays. For both cases, "
                f"series_dim={self.series_dim} should be the time dimension."
            )

        if self.series_dim < 0:
            self.series_dim = len(eeg.shape) + self.series_dim

        num_timesteps = eeg.shape[self.series_dim]

        if self.series_dim != (len(eeg.shape) - 1):
            transpose_dims = list(range(len(eeg.shape)))
            transpose_dims.pop(self.series_dim)
            transpose_dims = [*transpose_dims, self.series_dim]
            eeg = eeg.permute(*transpose_dims)
            # Based on transpose_dims, create undo_transpose_dims
            undo_transpose_dims = [0] * len(eeg.shape)
            for i, dim in enumerate(transpose_dims):
                undo_transpose_dims[dim] = i
        else:
            undo_transpose_dims = []

        # Use absolute window/warp sizes if provided, else relative
        window_size = self.window_size or int(num_timesteps * self.window_size_ratio)
        if window_size >= num_timesteps and self.keep_input_shape:
            return eeg

        # Use absolute warp size if provided, else relative
        warp_size = self.warp_size or int(window_size * self.warp_size_ratio)
        if warp_size == window_size:
            return eeg

        if num_timesteps - window_size > 0:
            start = int(torch.randint(0, num_timesteps - window_size, (1,)).item())
        else:
            start = 0

        start_seg = eeg[..., :start]
        window_seg = eeg[..., start : start + window_size]
        end_seg = eeg[..., start + window_size :]

        # Interpolate the window segment to the new warp size
        is_2d = window_seg.dim() == 2
        if is_2d:
            window_seg = window_seg.unsqueeze(0)

        warped_window = F.interpolate(
            window_seg,
            size=warp_size,
            mode="linear",
            align_corners=False,
        )

        # If it was 2d, squeeze back
        if is_2d:
            warped_window = warped_window.squeeze(0)

        # Concatenate the segments back together
        warped_eeg = torch.cat((start_seg, warped_window, end_seg), dim=-1)

        # Interpolate the entire warped signal back to the original length
        is_2d = warped_eeg.dim() == 2
        if is_2d:
            warped_eeg = warped_eeg.unsqueeze(0)

        if self.keep_input_shape:
            rescaled_eeg = F.interpolate(
                warped_eeg,
                size=num_timesteps,
                mode="linear",
                align_corners=False,
            )
        else:
            rescaled_eeg = warped_eeg

        if self.series_dim != (len(eeg.shape) - 1):
            rescaled_eeg = rescaled_eeg.permute(*undo_transpose_dims)

        # If it was 2d, squeeze back
        if is_2d:
            rescaled_eeg = rescaled_eeg.squeeze(0)

        return rescaled_eeg

    @property
    def repr_body(self) -> dict:
        """Return a dictionary of the class attributes for representation."""
        return dict(
            super().repr_body,
            window_size=self.window_size,
            warp_size=self.warp_size,
            series_dim=self.series_dim,
        )


class RandomFrequencyShift(RandomEEGTransform):
    """GPU-compatible version of [RandomFrequencyShift](https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.transforms.RandomFrequencyShift.html#torcheeg.transforms.RandomFrequencyShift).

    Modified from: torcheeg/transforms/torch/random.py

    Apply a frequency shift with a specified probability, after which the EEG signals of all
    channels are equally shifted in the frequency domain.

    .. code-block:: python

        transform = RandomFrequencyShift()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomFrequencyShift(sampling_rate=128, shift_min=4.0)
        transform(eeg=torch.randn(1, 32, 128))['eeg'].shape
        >>> (1, 32, 128)

        transform = RandomFrequencyShift(p=1.0, series_dim=0)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        sampling_rate (int): The original sampling rate in Hz (default: :obj:`128`)
        shift_min (float or int): The minimum shift in the random transformation.
            (default: :obj:`-2.0`)
        shift_max (float or int): The maximum shift in random transformation. (default: :obj:`2.0`)
        series_dim (int): Dimension of the time series in the input tensor. (default: :obj:`-1`)
        p (float): Probability of applying random mask on EEG signal samples. Should be between
            0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that
            masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the
            baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    """

    def __init__(
        self,
        p: float = 0.5,
        sampling_rate: int = 128,
        shift_min: float = -2.0,
        shift_max: float = 2.0,
        series_dim: int = 0,
        apply_to_baseline: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize RandomFrequencyShift."""
        super().__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.sampling_rate = sampling_rate
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.series_dim = series_dim

    def __call__(
        self,
        *args,  # noqa: ANN002
        eeg: torch.Tensor,
        baseline: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Apply a frequency shift with a specified probability.

        Args:
            *args: Additional positional arguments.
            eeg (torch.Tensor): The input EEG signal.
            baseline (torch.Tensor, optional): The corresponding baseline signal, if
                apply_to_baseline is set to True and baseline is passed, the baseline signal will
                be transformed with the same way as the experimental signal.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output EEG signal after applying a random sampling_rate shift.
        """
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        """Apply a frequency shift to the EEG signal."""
        # Get device and dtype from the input tensor
        device = eeg.device
        dtype = eeg.dtype

        if self.series_dim != (len(eeg.shape) - 1):
            permute_dims = list(range(len(eeg.shape)))
            permute_dims.pop(self.series_dim)
            permute_dims = [*permute_dims, self.series_dim]
            eeg = eeg.permute(permute_dims)
            # Based on transpose_dims, create undo_transpose_dims
            undo_permute_dims = [0] * len(eeg.shape)
            for i, dim in enumerate(permute_dims):
                undo_permute_dims[dim] = i
        else:
            undo_permute_dims = []

        N_orig = eeg.shape[-1]
        N_padded = 2 ** int(np.ceil(np.log2(np.abs(N_orig))))
        t = torch.arange(N_padded, device=device, dtype=dtype) / self.sampling_rate
        padded = pad(eeg, (0, N_padded - N_orig))

        if torch.is_complex(eeg):
            raise ValueError("eeg must be real.")

        N = padded.shape[-1]
        f = fft(padded, N, dim=-1)
        h = torch.zeros_like(f)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1 : N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1 : (N + 1) // 2] = 2

        analytical = ifft(f * h, dim=-1)

        shift = (
            torch.rand(1, device=device, dtype=dtype) * (self.shift_max - self.shift_min)
            + self.shift_min
        )
        shifted = analytical * torch.exp(torch.tensor(0 + 2j, device=device) * np.pi * shift * t)

        shifted = shifted[..., :N_orig].real.float()

        if self.series_dim != (len(eeg.shape) - 1):
            shifted = shifted.permute(*undo_permute_dims)

        return shifted

    @property
    def repr_body(self) -> dict:
        """Return a dictionary of the class attributes for representation."""
        return dict(
            super().repr_body,
            sampling_rate=self.sampling_rate,
            shift_min=self.shift_min,
            shift_max=self.shift_max,
            series_dim=self.series_dim,
        )


class RandomPCANoise(RandomEEGTransform):
    """GPU-compatible version of [RandomPCANoise](https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.transforms.RandomPCANoise.html#torcheeg.transforms.RandomPCANoise).

    Modified from: torcheeg/transforms/torch/random.py

    Add noise with a given probability, where the noise is added to the principal components of
    each channel of the EEG signal. In particular, the noise added by each channel is different.

    .. code-block:: python

        transform = RandomPCANoise()
        transform(eeg=torch.randn(32, 128))['eeg'].shape
        >>> (32, 128)

        transform = RandomPCANoise(mean=0.5, std=2.0, n_components=4)
        transform(eeg=torch.randn(1, 32, 128))['eeg'].shape
        >>> (1, 32, 128)

        transform = RandomPCANoise(p=1.0, series_dim=0)
        transform(eeg=torch.randn(128, 9, 9))['eeg'].shape
        >>> (128, 9, 9)

    Args:
        mean (float): The mean of the normal distribution of noise. (default: :obj:`0.0`)
        std (float): The standard deviation of the normal distribution of noise.
            (default: :obj:`0.0`)
        series_dim (int): Dimension of the time series in the input tensor. (default: :obj:`-1`)
        n_components (int): Number of components to add noise. if n_components is not set, the
            first two components are used to add noise.
        p (float): Probability of applying random mask on EEG signal samples. Should be between
            0.0 and 1.0, where 0.0 means no mask is applied to every sample and 1.0 means that
            masks are applied to every sample. (default: :obj:`0.5`)
        apply_to_baseline (bool): Whether to act on the baseline signal at the same time, if the
            baseline is passed in when calling. (default: :obj:`False`)

    .. automethod:: __call__
    """

    def __init__(
        self,
        n_components: int = 2,
        series_dim: int = -1,
        mean: float = 0.0,
        std: float = 1.0,
        p: float = 0.5,
        apply_to_baseline: bool = False,  # noqa: FBT001, FBT002
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Initialize RandomPCANoise."""
        super().__init__(p=p, apply_to_baseline=apply_to_baseline)
        self.n_components = n_components
        self.series_dim = series_dim
        self.mean = mean
        self.std = std

    def __call__(
        self,
        *args,  # noqa: ANN002
        eeg: torch.Tensor,
        baseline: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Apply PCA-based noise augmentation to the EEG signal.

        Args:
            *args: Additional positional arguments.
            eeg (torch.Tensor): The input EEG signal.
            baseline (torch.Tensor, optional) : The corresponding baseline signal,
                if apply_to_baseline is set to True and baseline is passed, the baseline signal
                will be transformed with the same way as the experimental signal.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output EEG signal after applying a random PCA noise.
        """
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def random_apply(self, eeg: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        """Apply PCA-based noise augmentation to the EEG signal.

        Expects `eeg` to be of shape `(batch, channels, time)` or `(channels, time)`. PCA is
        performed on the time dimension.
        """
        if len(eeg.shape) != 2 and len(eeg.shape) != 3:
            raise ValueError(
                "RandomPCANoise is only supported on 2D arrays or 3D arrays. For both cases, "
                f"series_dim={self.series_dim} should be the time dimension."
            )

        if self.series_dim != (len(eeg.shape) - 1):
            transpose_dims = list(range(len(eeg.shape)))
            transpose_dims.pop(self.series_dim)
            transpose_dims = [*transpose_dims, self.series_dim]
            # Based on transpose_dims, create undo_transpose_dims
            undo_transpose_dims = [0] * len(eeg.shape)
            for i, dim in enumerate(transpose_dims):
                undo_transpose_dims[dim] = i
            eeg = eeg.permute(*transpose_dims)
        else:
            undo_transpose_dims = []

        try:
            # --- 1. Prepare data for PCA ---
            is_2d = eeg.dim() == 2
            if is_2d:
                eeg = eeg.unsqueeze(0)  # (channels, time) -> (1, channels, time)

            batch_size, num_channels, num_timesteps = eeg.shape

            if self.n_components > min(num_channels, num_timesteps):
                logger.warning(
                    f"n_components ({self.n_components}) is greater than the smaller dimension "
                    f"of the data ({min(num_channels, num_timesteps)}). "
                    "Skipping RandomPCANoise augmentation."
                )
                return eeg.squeeze(0) if is_2d else eeg

            # PCA works on centered data. We center each channel's time series.
            # The original torcheeg implementation transposes and centers along features/channels.
            # Let's align with that: (B, C, T) -> (B, T, C)
            eeg_transposed = eeg.transpose(1, 2)  # Shape: (B, T, C)
            mean = torch.mean(eeg_transposed, dim=1, keepdim=True)  # Mean across time
            centered_eeg = eeg_transposed - mean

            # Reshape for PCA: (B*T, C)
            centered_eeg_flat = centered_eeg.reshape(-1, num_channels)

            # --- 2. Perform PCA using SVD on the flattened data ---
            # U: (B*T, B*T), S: (min(B*T, C),), Vh: (C, C)
            _, S, Vh = torch.linalg.svd(centered_eeg_flat, full_matrices=False)

            # Principal components (eigenvectors) are the rows of Vh
            components = Vh[: self.n_components, :]  # Shape: (n_components, C)

            # Explained variance is proportional to the square of singular values
            explained_variance = S[: self.n_components] ** 2
            explained_variance_ratio = explained_variance / torch.sum(S**2)

            # --- 3. Generate and Scale Noise ---
            # Generate random coefficients for the noise for each sample in the batch
            noise_coeffs = torch.randn(
                batch_size, self.n_components, device=eeg.device, dtype=eeg.dtype
            )
            noise_coeffs = (noise_coeffs * self.std) + self.mean

            # Scale noise by the explained variance of each component
            # Shape: (B, n_components) * (n_components,) -> (B, n_components)
            scaled_noise_coeffs = noise_coeffs * explained_variance_ratio

            # --- 4. Reconstruct Noise Signal ---
            # Create the noise signal by taking a weighted sum of the principal components.
            # (B, n_components) @ (n_components, C) -> (B, C)
            noise_signal_per_batch = scaled_noise_coeffs @ components

            # --- 5. Add Noise and Return ---
            # The noise signal is a single vector per channel for each batch item.
            # We add it to the whole time series of that batch item.
            # noise_signal_per_batch shape: (B, C)
            # Unsqueeze to (B, 1, C) to broadcast over the time dim of eeg_transposed (B, T, C)
            augmented_eeg_transposed = eeg_transposed + noise_signal_per_batch.unsqueeze(1)

            # Transpose back to original (B, C, T) shape
            augmented_eeg = augmented_eeg_transposed.transpose(1, 2)

            # Squeeze back if input was 2D
            if is_2d:
                augmented_eeg = augmented_eeg.squeeze(0)
        except Exception as e:
            logger.warning(
                f"Failed to apply RandomPCANoise due to an error: {e}. Returning original EEG."
            )
            # Return the original tensor in case of any error
            return eeg
        else:
            if self.series_dim != (len(eeg.shape) - 1):
                augmented_eeg = augmented_eeg.permute(*undo_transpose_dims)

            return augmented_eeg

    @property
    def repr_body(self) -> dict:
        """Return a dictionary of the class attributes for representation."""
        return dict(
            super().repr_body,
            mean=self.mean,
            std=self.std,
            n_components=self.n_components,
            series_dim=self.series_dim,
        )


# Mapping of names in config to the actual transform classes (local or torcheeg)
TRANSFORM_MAP = {
    # Custom GPU-compatible versions
    "RandomWindowSlice": RandomWindowSlice,
    "RandomWindowWarp": RandomWindowWarp,
    "RandomFrequencyShift": RandomFrequencyShift,
    "RandomPCANoise": RandomPCANoise,
    # Direct from torcheeg (already GPU-compatible)
    "RandomNoise": transforms.RandomNoise,
    "RandomMask": transforms.RandomMask,
    "RandomShift": transforms.RandomShift,
    "RandomSignFlip": transforms.RandomSignFlip,
    "RandomChannelShuffle": transforms.RandomChannelShuffle,
}


def build_augmentation_pipeline(
    aug_config: dict[str, Any],
    sfreq: float | None = None,
) -> transforms.Compose | None:
    """Build a torcheeg augmentation pipeline from a configuration dictionary.

    Args:
        aug_config (dict[str, Any]): The augmentation configuration, typically from a YAML file.
            It should contain an 'apply' key and a 'transforms' list.
        sfreq (float | None): The sampling frequency of the EEG data, required by some transforms
            like RandomFrequencyShift.

    Returns:
        (transforms.Compose | None): A composed torcheeg transform object if augmentation is
            enabled, otherwise None.
    """
    if not aug_config.get("apply", False):
        return None

    transform_list = []
    config_transforms = aug_config.get("transforms", [])
    if not config_transforms:
        logger.warning("Augmentation is enabled but no transforms are specified in the config.")
        return None

    logger.debug("Building EEG augmentation pipeline...")

    for transform_info in config_transforms:
        # Use a copy to avoid modifying the original config dict during popping
        params = transform_info.copy()
        transform_name = params.pop("name", None)
        if not transform_name:
            logger.warning("A transform in the config is missing a 'name'. Skipping.")
            continue

        if transform_name not in TRANSFORM_MAP:
            logger.error(
                f"Transform '{transform_name}' is not a supported GPU-compatible "
                "transform. Skipping."
            )
            continue

        transform_class = TRANSFORM_MAP[transform_name]

        # Inject sampling_rate if the transform requires it and it's not provided
        if (
            "sampling_rate" in inspect.signature(transform_class).parameters
            and "sampling_rate" in params
            and not params["sampling_rate"]
            and sfreq
        ):
            params["sampling_rate"] = sfreq
            logger.debug(f"  - Injecting sampling_rate={sfreq} into {transform_name}")

        try:
            # Instantiate the transform with its parameters
            transform_instance = transform_class(**params)
            transform_list.append(transform_instance)
            logger.debug(f"  + Added transform: {transform_name}({params})")
        except Exception as e:
            logger.error(
                f"Failed to instantiate transform '{transform_name}' with params {params}: {e}"
            )

    return transforms.Compose(transform_list)
