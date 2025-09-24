# ruff: noqa: S101
"""Tests for custom GPU-compatible augmentation transforms."""

import pytest
import torch

from pretrain_braindecode_models.config import DEVICE
from pretrain_braindecode_models.modeling.augmentations import (
    RandomFrequencyShift,
    RandomPCANoise,
    RandomWindowSlice,
    RandomWindowWarp,
)

DTYPE = torch.float32

# --- Test Data Fixtures ---


@pytest.fixture(scope="module")
def sample_eeg_3d() -> torch.Tensor:
    """Generate a sample 3D of shape `(batch, channels, time)` tensor on the target device."""
    # A simple sine wave per channel with different frequencies
    batch_size, n_channels, n_timesteps = 4, 8, 1000
    t = torch.linspace(0, 1, n_timesteps, device=DEVICE, dtype=DTYPE)
    freqs = torch.linspace(5, 50, n_channels, device=DEVICE, dtype=DTYPE).view(1, -1, 1)
    eeg = torch.sin(2 * torch.pi * freqs * t.unsqueeze(0).unsqueeze(0))
    # Add batch dimension
    return eeg.repeat(batch_size, 1, 1)


@pytest.fixture(scope="module")
def sample_eeg_2d() -> torch.Tensor:
    """Generate a sample 2D of shape `(channels, time)` tensor on the target device."""
    n_channels, n_timesteps = 8, 1000
    t = torch.linspace(0, 1, n_timesteps, device=DEVICE, dtype=DTYPE)
    freqs = torch.linspace(5, 50, n_channels, device=DEVICE, dtype=DTYPE).view(-1, 1)
    return torch.sin(2 * torch.pi * freqs * t.unsqueeze(0))


# --- Tests for RandomWindowSlice ---


def test_rws_applies_transform(sample_eeg_3d: torch.Tensor) -> None:
    """Test that RandomWindowSlice changes the input tensor when p=1.0."""
    transform = RandomWindowSlice(p=1.0, window_size_ratio=0.5)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]

    assert augmented_eeg.shape == sample_eeg_3d.shape
    assert not torch.allclose(augmented_eeg, sample_eeg_3d)


def test_rws_does_not_apply_transform(sample_eeg_3d: torch.Tensor) -> None:
    """Test that RandomWindowSlice does not change the input tensor when p=0.0."""
    transform = RandomWindowSlice(p=0.0)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    assert torch.allclose(augmented_eeg, sample_eeg_3d)


def test_rws_correct_series_dim_2d(sample_eeg_2d: torch.Tensor) -> None:
    """Test RandomWindowSlice with a non-default series_dim on a 2D tensor."""
    # (Time, Channels) instead of (Channels, Time)
    eeg_tc = sample_eeg_2d.T
    transform = RandomWindowSlice(p=1.0, series_dim=0, window_size_ratio=0.5)
    augmented_eeg = transform(eeg=eeg_tc)["eeg"]

    assert augmented_eeg.shape == eeg_tc.shape
    assert not torch.allclose(augmented_eeg, eeg_tc)


def test_rws_window_size_larger_than_signal(sample_eeg_2d: torch.Tensor) -> None:
    """Test that if window size is too large, the original signal is returned."""
    transform = RandomWindowSlice(p=1.0, window_size_ratio=1.5)
    augmented_eeg = transform(eeg=sample_eeg_2d)["eeg"]
    assert torch.allclose(augmented_eeg, sample_eeg_2d)


def test_rws_not_keep_input_shape(sample_eeg_3d: torch.Tensor) -> None:
    """Test that RandomWindowSlice can return a different shape when keep_input_shape is False."""
    transform = RandomWindowSlice(p=1.0, window_size_ratio=0.5, keep_input_shape=False)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]

    expected_timesteps = int(sample_eeg_3d.shape[2] * 0.5)
    assert augmented_eeg.shape == (
        sample_eeg_3d.shape[0],
        sample_eeg_3d.shape[1],
        expected_timesteps,
    )

    # Test with the time dimension being in the middle
    eeg_nct = sample_eeg_3d.permute(0, 2, 1)  # (Batch, Time, Channels)
    transform = RandomWindowSlice(
        p=1.0,
        series_dim=1,
        window_size_ratio=0.5,
        keep_input_shape=False,
    )
    augmented_eeg_nct = transform(eeg=eeg_nct)["eeg"]
    expected_timesteps = int(eeg_nct.shape[1] * 0.5)
    assert augmented_eeg_nct.shape == (
        eeg_nct.shape[0],
        expected_timesteps,
        eeg_nct.shape[2],
    )

    # Test with window size larger than signal. The original signal should be returned, due to:
    # `if window_size >= num_timesteps: return eeg``
    transform = RandomWindowSlice(p=1.0, window_size_ratio=1.5, keep_input_shape=False)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    assert augmented_eeg.shape == sample_eeg_3d.shape
    assert torch.allclose(augmented_eeg, sample_eeg_3d)


# --- Tests for RandomWindowWarp ---


def test_rww_applies_transform(sample_eeg_3d: torch.Tensor) -> None:
    """Test that RandomWindowWarp changes the input tensor when p=1.0."""
    transform = RandomWindowWarp(p=1.0, window_size_ratio=0.2, warp_size_ratio=2.0)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]

    assert augmented_eeg.shape == sample_eeg_3d.shape
    assert not torch.allclose(augmented_eeg, sample_eeg_3d)


def test_rww_speed_up_and_slow_down(sample_eeg_3d: torch.Tensor) -> None:
    """Test both speeding up and slowing down works."""
    # Speed up
    transform_fast = RandomWindowWarp(p=1.0, window_size_ratio=0.5, warp_size_ratio=0.5)
    aug_fast = transform_fast(eeg=sample_eeg_3d)["eeg"]
    assert not torch.allclose(aug_fast, sample_eeg_3d)

    # Slow down
    transform_slow = RandomWindowWarp(p=1.0, window_size_ratio=0.5, warp_size_ratio=2.0)
    aug_slow = transform_slow(eeg=sample_eeg_3d)["eeg"]
    assert not torch.allclose(aug_slow, sample_eeg_3d)


def test_rww_speed_up_and_slow_down_not_keep_input_shape_3d(sample_eeg_3d: torch.Tensor) -> None:
    """Test that if keep_input_shape is False, the output shape is different on 3D input."""
    # sample_eeg_3d has shape (4, 8, 1000)
    # Slow down (warp_size_ratio > 1.0)
    transform = RandomWindowWarp(
        window_size_ratio=0.5,  # Pick a window of size=500
        warp_size_ratio=2.0,  # Warp it to size=1000
        keep_input_shape=False,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_3d.shape[0],
        sample_eeg_3d.shape[1],
        sample_eeg_3d.shape[2] + int(sample_eeg_3d.shape[2] * 0.5),  # 1000 + 500 = 1500
    )

    transform = RandomWindowWarp(
        window_size_ratio=1.0,  # Pick a window of size=1000
        warp_size_ratio=2.0,  # Warp it to size=2000
        keep_input_shape=False,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_3d.shape[0],
        sample_eeg_3d.shape[1],
        sample_eeg_3d.shape[2] * 2,  # 1000 * 2 = 2000
    )

    # Speed up (warp_size_ratio < 1.0)
    transform = RandomWindowWarp(
        window_size_ratio=0.5,  # Pick a window of size=500
        warp_size_ratio=0.5,  # Warp it to size=250
        keep_input_shape=False,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_3d.shape[0],
        sample_eeg_3d.shape[1],
        sample_eeg_3d.shape[2] - int(sample_eeg_3d.shape[2] * 0.25),  # 1000 - 250 = 750
    )

    transform = RandomWindowWarp(
        window_size_ratio=1.0,  # Pick a window of size=1000
        warp_size_ratio=0.5,  # Warp it to size=500
        keep_input_shape=False,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_3d.shape[0],
        sample_eeg_3d.shape[1],
        sample_eeg_3d.shape[2] // 2,  # 1000 // 2 = 500
    )


def test_rww_speed_up_and_slow_down_not_keep_input_shape_ntc_3d(
    sample_eeg_3d: torch.Tensor,
) -> None:
    """Test when keep_input_shape is False and 3D input with time in middle."""
    # sample_eeg_3d has shape (4, 8, 1000)
    sample_eeg_3d_ntc = sample_eeg_3d.permute(0, 2, 1)  # (Batch, Time, Channels)

    # Slow down (warp_size_ratio > 1.0)
    transform = RandomWindowWarp(
        window_size_ratio=0.5,  # Pick a window of size=500
        warp_size_ratio=2.0,  # Warp it to size=1000
        keep_input_shape=False,
        series_dim=1,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_3d_ntc)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_3d_ntc.shape[0],
        sample_eeg_3d_ntc.shape[1] + int(sample_eeg_3d_ntc.shape[1] * 0.5),  # 1000 + 500 = 1500
        sample_eeg_3d_ntc.shape[2],
    )

    transform = RandomWindowWarp(
        window_size_ratio=1.0,  # Pick a window of size=1000
        warp_size_ratio=2.0,  # Warp it to size=2000
        keep_input_shape=False,
        series_dim=1,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_3d_ntc)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_3d_ntc.shape[0],
        sample_eeg_3d_ntc.shape[1] * 2,  # 1000 * 2 = 2000
        sample_eeg_3d_ntc.shape[2],
    )

    # Speed up (warp_size_ratio < 1.0)
    transform = RandomWindowWarp(
        window_size_ratio=0.5,  # Pick a window of size=500
        warp_size_ratio=0.5,  # Warp it to size=250
        keep_input_shape=False,
        series_dim=1,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_3d_ntc)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_3d_ntc.shape[0],
        sample_eeg_3d_ntc.shape[1] - int(sample_eeg_3d_ntc.shape[1] * 0.25),  # 1000 - 250 = 750
        sample_eeg_3d_ntc.shape[2],
    )

    transform = RandomWindowWarp(
        window_size_ratio=1.0,  # Pick a window of size=1000
        warp_size_ratio=0.5,  # Warp it to size=500
        keep_input_shape=False,
        series_dim=1,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_3d_ntc)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_3d_ntc.shape[0],
        sample_eeg_3d_ntc.shape[1] // 2,  # 1000 // 2 = 500
        sample_eeg_3d_ntc.shape[2],
    )


def test_rww_speed_up_and_slow_down_not_keep_input_shape_2d(sample_eeg_2d: torch.Tensor) -> None:
    """Test that if keep_input_shape is False, the output shape is different on 2D input."""
    # sample_eeg_2d has shape (8, 1000)
    # Slow down (warp_size_ratio > 1.0)
    transform = RandomWindowWarp(
        window_size_ratio=0.5,  # Pick a window of size=500
        warp_size_ratio=2.0,  # Warp it to size=1000
        keep_input_shape=False,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_2d)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_2d.shape[0],
        sample_eeg_2d.shape[1] + int(sample_eeg_2d.shape[1] * 0.5),  # 1000 + 500 = 1500
    )

    transform = RandomWindowWarp(
        window_size_ratio=1.0,  # Pick a window of size=1000
        warp_size_ratio=2.0,  # Warp it to size=2000
        keep_input_shape=False,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_2d)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_2d.shape[0],
        sample_eeg_2d.shape[1] * 2,  # 1000 * 2 = 2000
    )

    # Speed up (warp_size_ratio < 1.0)
    transform = RandomWindowWarp(
        window_size_ratio=0.5,  # Pick a window of size=500
        warp_size_ratio=0.5,  # Warp it to size=250
        keep_input_shape=False,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_2d)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_2d.shape[0],
        sample_eeg_2d.shape[1] - int(sample_eeg_2d.shape[1] * 0.25),  # 1000 - 250 = 750
    )

    transform = RandomWindowWarp(
        window_size_ratio=1.0,  # Pick a window of size=1000
        warp_size_ratio=0.5,  # Warp it to size=500
        keep_input_shape=False,
        p=1.0,
    )
    augmented_eeg = transform(eeg=sample_eeg_2d)["eeg"]
    assert augmented_eeg.shape == (
        sample_eeg_2d.shape[0],
        sample_eeg_2d.shape[1] // 2,  # 1000 // 2 = 500
    )


def test_rww_no_warp(sample_eeg_3d: torch.Tensor) -> None:
    """Test that if warp_size_ratio is 1.0, the signal is unchanged."""
    transform = RandomWindowWarp(p=1.0, window_size_ratio=0.5, warp_size_ratio=1.0)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    # Should be close, but floating point errors from interpolation might occur
    assert torch.allclose(augmented_eeg, sample_eeg_3d, atol=1e-6)


# --- Tests for RandomFrequencyShift ---


def test_rfs_applies_transform(sample_eeg_3d: torch.Tensor) -> None:
    """Test that RandomFrequencyShift changes the input tensor when p=1.0."""
    transform = RandomFrequencyShift(p=1.0, sampling_rate=100, shift_min=5.0, shift_max=10.0)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]

    assert augmented_eeg.shape == sample_eeg_3d.shape
    assert not torch.allclose(augmented_eeg, sample_eeg_3d)


def test_rfs_zero_shift(sample_eeg_3d: torch.Tensor) -> None:
    """Test that a shift of zero leaves the signal largely unchanged."""
    transform = RandomFrequencyShift(p=1.0, sampling_rate=100, shift_min=0.0, shift_max=0.0)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    # Due to FFT padding/unpadding and floating point math, a small tolerance is needed
    assert torch.allclose(augmented_eeg, sample_eeg_3d, atol=1e-5)


def test_rfs_correct_series_dim_3d(sample_eeg_3d: torch.Tensor) -> None:
    """Test RandomFrequencyShift with a non-default series_dim on a 3D tensor."""
    # Transpose to (Batch, Time, Channels)
    eeg_ntc = sample_eeg_3d.transpose(1, 2)
    transform = RandomFrequencyShift(p=1.0, sampling_rate=100, series_dim=1)
    augmented_eeg = transform(eeg=eeg_ntc)["eeg"]

    assert augmented_eeg.shape == eeg_ntc.shape
    assert not torch.allclose(augmented_eeg, eeg_ntc)


# --- Tests for RandomPCANoise ---


def test_rpn_applies_transform(sample_eeg_3d: torch.Tensor) -> None:
    """Test that RandomPCANoise changes the input tensor when p=1.0."""
    transform = RandomPCANoise(p=1.0, n_components=4, std=0.1)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]

    assert augmented_eeg.shape == sample_eeg_3d.shape
    assert not torch.allclose(augmented_eeg, sample_eeg_3d)


def test_rpn_zero_std(sample_eeg_3d: torch.Tensor) -> None:
    """Test that zero standard deviation results in no added noise."""
    transform = RandomPCANoise(p=1.0, n_components=4, std=0.0)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]
    assert torch.allclose(augmented_eeg, sample_eeg_3d)


def test_rpn_handles_2d_input(sample_eeg_2d: torch.Tensor) -> None:
    """Test that RandomPCANoise correctly processes a 2D input tensor."""
    transform = RandomPCANoise(p=1.0, n_components=2, std=0.1)
    augmented_eeg = transform(eeg=sample_eeg_2d)["eeg"]

    assert augmented_eeg.shape == sample_eeg_2d.shape
    assert not torch.allclose(augmented_eeg, sample_eeg_2d)


def test_rpn_handles_insufficient_components(sample_eeg_3d: torch.Tensor) -> None:
    """Test that RandomPCANoise gracefully handles n_components > num_channels."""
    n_channels = sample_eeg_3d.shape[1]
    transform = RandomPCANoise(p=1.0, n_components=n_channels + 1)
    augmented_eeg = transform(eeg=sample_eeg_3d)["eeg"]

    # It should log a warning and return the original tensor
    assert torch.allclose(augmented_eeg, sample_eeg_3d)
