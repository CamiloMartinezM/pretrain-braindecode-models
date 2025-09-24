# ruff: noqa: S101, D401
"""Tests for the `reorder_dims` function in `pretrain_braindecode_models.utils.misc`."""

import numpy as np
import pytest
import torch

from pretrain_braindecode_models.utils.misc import reorder_dims

# --- Test Data Fixtures ---
# Using fixtures makes it easy to reuse test data across multiple tests.


@pytest.fixture
def sample_numpy_3d_ntc() -> np.ndarray:
    """A sample 3D NumPy array in NTC (Batch, Time, Channels) format."""
    # Shape: (Batch=2, Time=3, Channels=4)
    return np.arange(24).reshape(2, 3, 4)


@pytest.fixture
def sample_torch_3d_ntc(sample_numpy_3d_ntc: np.ndarray) -> torch.Tensor:
    """A sample 3D PyTorch Tensor in NTC format."""
    return torch.from_numpy(sample_numpy_3d_ntc)


@pytest.fixture
def sample_numpy_2d_tc() -> np.ndarray:
    """A sample 2D NumPy array in TC (Time, Channels) format."""
    # Shape: (Time=3, Channels=4)
    return np.arange(12).reshape(3, 4)


@pytest.fixture
def sample_torch_2d_tc(sample_numpy_2d_tc: np.ndarray) -> torch.Tensor:
    """A sample 2D PyTorch Tensor in TC format."""
    return torch.from_numpy(sample_numpy_2d_tc)


# --- Core Functionality Tests ---


def test_reorder_ntc_to_nct_numpy(sample_numpy_3d_ntc: np.ndarray) -> None:
    """Tests reordering from (N, T, C) to (N, C, T) for a NumPy array."""
    original_data = sample_numpy_3d_ntc
    reordered_data = reorder_dims(original_data, current_order="NTC", target_order="NCT")

    # Expected shape: (Batch=2, Channels=4, Time=3)
    assert reordered_data.shape == (2, 4, 3)

    # Check if a specific element moved correctly
    # original_data[batch=1, time=2, channel=3] should now be at
    # reordered_data[batch=1, channel=3, time=2]
    assert original_data[1, 2, 3] == reordered_data[1, 3, 2]
    assert isinstance(reordered_data, np.ndarray)


def test_reorder_ntc_to_nct_torch(sample_torch_3d_ntc: torch.Tensor) -> None:
    """Tests reordering from (N, T, C) to (N, C, T) for a PyTorch Tensor."""
    original_data = sample_torch_3d_ntc
    reordered_data = reorder_dims(original_data, current_order="NTC", target_order="NCT")

    assert reordered_data.shape == (2, 4, 3)
    assert original_data[1, 2, 3] == reordered_data[1, 3, 2]
    assert isinstance(reordered_data, torch.Tensor)


def test_reorder_nct_to_ntc_numpy(sample_numpy_3d_ntc: np.ndarray) -> None:
    """Tests reordering from (N, C, T) back to (N, T, C) for a NumPy array."""
    # First, convert our fixture to NCT format
    nct_data = sample_numpy_3d_ntc.transpose(0, 2, 1)
    assert nct_data.shape == (2, 4, 3)

    # Now, reorder it back to NTC
    reordered_data = reorder_dims(nct_data, current_order="NCT", target_order="NTC")

    # The result should be identical to the original fixture
    assert reordered_data.shape == (2, 3, 4)
    np.testing.assert_array_equal(reordered_data, sample_numpy_3d_ntc)


def test_reorder_2d_tc_to_ct_numpy(sample_numpy_2d_tc: np.ndarray) -> None:
    """Tests reordering a 2D array from (T, C) to (C, T)."""
    original_data = sample_numpy_2d_tc
    reordered_data = reorder_dims(original_data, current_order="TC", target_order="CT")

    assert reordered_data.shape == (4, 3)
    assert original_data[2, 3] == reordered_data[3, 2]
    assert isinstance(reordered_data, np.ndarray)


def test_reorder_2d_tc_to_ct_torch(sample_torch_2d_tc: torch.Tensor) -> None:
    """Tests reordering a 2D tensor from (T, C) to (C, T)."""
    original_data = sample_torch_2d_tc
    reordered_data = reorder_dims(original_data, current_order="TC", target_order="CT")

    assert reordered_data.shape == (4, 3)
    assert original_data[2, 3] == reordered_data[3, 2]
    assert isinstance(reordered_data, torch.Tensor)


def test_no_reorder_needed(
    sample_numpy_3d_ntc: np.ndarray, sample_torch_3d_ntc: torch.Tensor
) -> None:
    """Tests that the function returns the original object if no reordering is needed."""
    # NumPy
    reordered_np = reorder_dims(sample_numpy_3d_ntc, "NTC", "NTC")
    assert reordered_np is sample_numpy_3d_ntc  # Should be the same object, not a copy

    # PyTorch
    reordered_torch = reorder_dims(sample_torch_3d_ntc, "NTC", "NTC")
    assert reordered_torch is sample_torch_3d_ntc


# --- Edge Case and Error Handling Tests ---


def test_raises_error_on_dim_mismatch(sample_numpy_3d_ntc: np.ndarray) -> None:
    """Tests that a ValueError is raised if data.ndim does not match len(current_order)."""
    with pytest.raises(
        ValueError,
        match="Input data has 3 dimensions, but current_order 'TC' implies 2 dimensions",
    ):
        reorder_dims(sample_numpy_3d_ntc, "TC", "CT")

    with pytest.raises(
        ValueError, match="Input data has 3 dimensions, but current_order 'N' implies 1 dimensions"
    ):
        # Although 'N' isn't a valid DimOrder, the ndim check should happen first
        reorder_dims(sample_numpy_3d_ntc, "N", "C")  # type: ignore[reportArgumentType, arg-type]


def test_raises_error_on_order_string_length_mismatch(sample_numpy_3d_ntc: np.ndarray) -> None:
    """Tests that a ValueError is raised if order strings have different lengths."""
    with pytest.raises(ValueError, match="Dimension order strings must have the same length"):
        reorder_dims(sample_numpy_3d_ntc, "NTC", "TC")


def test_raises_error_on_unsupported_order_string() -> None:
    """Tests that a ValueError is raised for unsupported order strings."""
    data = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="Unsupported order"):
        reorder_dims(data, "XYZ", "NTC")  # type: ignore[reportArgumentType, arg-type]
    with pytest.raises(ValueError, match="Unsupported order"):
        reorder_dims(data, "NTC", "ABC")  # type: ignore[reportArgumentType, arg-type]


def test_raises_error_on_key_mismatch(sample_numpy_2d_tc: np.ndarray) -> None:
    """Tests that a ValueError is raised if target_order contains dims not in current_order."""
    with pytest.raises(ValueError, match="contains dimensions not present"):
        reorder_dims(sample_numpy_2d_tc, "CT", "NC")  # type: ignore[reportArgumentType, arg-type]


def test_raises_error_on_invalid_input_type() -> None:
    """Tests that a TypeError is raised for unsupported data types."""
    data = [[1, 2], [3, 4]]  # A plain list
    with pytest.raises(
        TypeError, match="Input data must be a NumPy array or PyTorch Tensor, got <class 'list'>"
    ):
        reorder_dims(data, "TC", "CT")  # type: ignore[reportArgumentType, arg-type]
