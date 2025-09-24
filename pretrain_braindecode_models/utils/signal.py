"""Signal processing utilities for EEG and audio data."""

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfilt


def compute_audio_event_envelope(
    audio_waveform: np.ndarray,
    sample_rate: int,
    *,
    high_pass_cutoff_hz: float = 4000.0,
    window_sec: float = 0.05,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an energy envelope of an audio signal, emphasizing sharp events.

    This is done by applying a high-pass filter to remove low-frequency noise/voice
    and then calculating the Root Mean Square (RMS) energy in sliding windows.

    Args:
        audio_waveform (np.ndarray): The raw audio signal (1D).
        sample_rate (int): The sample rate of the audio in Hz.
        high_pass_cutoff_hz (float): The cutoff frequency for the high-pass filter.
            Sounds below this frequency will be attenuated. Defaults to 2000.0.
        window_sec (float): The duration of the sliding window for RMS calculation.
        normalize (bool): If True, normalize the final envelope to the [0, 1] range.

    Returns:
        tuple[np,ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - The filtered audio waveform after high-pass filtering.
            - The calculated energy envelope.
            - The corresponding time axis for the envelope.
    """
    # 1. Design and apply a high-pass filter to isolate sharp sounds
    sos = butter(5, high_pass_cutoff_hz, btype="high", fs=sample_rate, output="sos")
    filtered_audio = sosfilt(sos, audio_waveform)

    if isinstance(filtered_audio, tuple):
        # If the output is a tuple, it means the filter returned multiple outputs
        filtered_audio = filtered_audio[0]

    # 2. Calculate the RMS energy in sliding windows to get the envelope
    window_size = int(window_sec * sample_rate)
    # Use convolution for an efficient sliding window RMS calculation
    squared_signal = np.power(filtered_audio, 2)
    window = np.ones(window_size) / float(window_size)
    rms_envelope = np.sqrt(np.convolve(squared_signal, window, "same"))

    envelope_time_axis = np.linspace(
        0, len(audio_waveform) / sample_rate, len(rms_envelope)
    )

    if normalize:
        max_val = np.max(rms_envelope)
        if max_val > 0:
            rms_envelope /= max_val

    return filtered_audio, rms_envelope, envelope_time_axis


def exponential_moving_average(
    data: np.ndarray,
    alpha: float,
    *,
    use_pandas_ewm: bool = True,
    preserve_min_max_points: bool = False,
    preserve_mode: Literal["first", "all"] = "first",
) -> np.ndarray:
    """Calculate the Exponential Moving Average (EMA) of a 1D array.

    Optionally preserves the original minimum and maximum data points in the smoothed output
    to maintain key features of the signal, which is useful for visualization.

    Args:
        data (np.ndarray): The 1D numpy array of data to smooth.
        alpha (float): The smoothing factor, between 0 and 1. A smaller alpha results in
            more smoothing, while alpha = 1 means no smoothing.
        use_pandas_ewm (bool): If True, use pandas' `ewm` for calculation, which can be more
            numerically stable. Otherwise, use a manual iterative method.
        preserve_min_max_points (bool): If True, the original minimum and maximum values
            from the input data will be restored at their original positions in the smoothed
            output array. Defaults to False.
        preserve_mode (Literal["first", "all"]): Governs which occurrences of the min/max
            values are preserved if `preserve_min_max_points` is True.
            - "first": Only restores the first occurrence of the minimum and the first
                       occurrence of the maximum.
            - "all": Restores all occurrences of the minimum and maximum values.
            Defaults to "first".

    Returns:
        np.ndarray: The smoothed data as a 1D numpy array.

    Raises:
        ValueError: If input data is not a 1D numpy array, if alpha is not in the
                    valid range (0, 1], or if `preserve_mode` is invalid.
    """
    # --- Input Validation ---
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a 1D numpy array.")
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be in the range (0, 1].")
    if data.size == 0:
        return np.array([])
    if preserve_mode not in ["first", "all"]:
        raise ValueError("`preserve_mode` must be either 'first' or 'all'.")

    # --- Core EMA Calculation ---
    if use_pandas_ewm:
        s = pd.Series(data)
        # The span is related to alpha by: span = 2/alpha - 1
        ema = s.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    else:
        ema = np.zeros_like(data)
        ema[0] = data[0]  # The first EMA value is the first data point
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    # --- Optional: Preserve Min/Max Points ---
    if preserve_min_max_points:
        # Find the original min/max values and their indices
        min_val = np.min(data)
        max_val = np.max(data)
        min_indices = np.where(data == min_val)[0]
        max_indices = np.where(data == max_val)[0]

        if preserve_mode == "first":
            # If there are any min/max values, keep only the first index
            if min_indices.size > 0:
                min_indices = min_indices[:1]
            if max_indices.size > 0:
                max_indices = max_indices[:1]

        # Restore the original values at the identified indices in the smoothed array
        if min_indices.size > 0:
            ema[min_indices] = min_val
        if max_indices.size > 0:
            ema[max_indices] = max_val

    return ema


def mean_confidence_interval(
    data: list[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Calculate the mean, lower, and upper confidence intervals for the data.

    Args:
        data (list[float]): A list of numerical data points.
        confidence (float): The confidence level (e.g., 0.95 for 95% CI).

    Returns:
        (tuple[float, float, float]): A tuple containing (mean, lower_bound, upper_bound).
            Returns `(nan, nan, nan)` for insufficient data.
    """
    if len(data) < 2:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(data))
    sem = stats.sem(data)
    if sem == 0:  # Handle cases with no variance
        return mean, mean, mean

    lower, upper = stats.t.interval(
        confidence=confidence, df=len(data) - 1, loc=mean, scale=sem
    )
    return mean, lower, upper
