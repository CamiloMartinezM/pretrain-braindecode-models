"""Provides utility functions for managing the EEG data."""

import math
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import interpolate
from scipy.signal import resample, welch
from sklearn.preprocessing import StandardScaler

from pretrain_braindecode_models.config import logger
from pretrain_braindecode_models.utils.misc import is_strictly_increasing

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


def match_eeg_with_samples(
    eeg_dict: dict[str, np.ndarray],
    num_samples: int,
    method: Literal[
        "linear",
        "nearest",
        "nearest-up",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
        "resample",
    ]
    | int = "linear",
    hz: float = 30.0,
    fill_value: Literal["extrapolate", "raise", "clamp"] = "extrapolate",
) -> dict[str, np.ndarray]:
    """Synchronize EEG data with the given number of samples.

    This function resamples the EEG data to match the number of samples with a given frame rate.

    Args:
        eeg_dict (dict): Dictionary containing EEG 'data' and 'timestamps', where the data is
            a 2D array of shape `(n_channels, n_samples)` and timestamps is a 1D array of shape
            `(n_samples,)`
        num_samples (int): Number of samples to match with the EEG data
        method (str):  Specifies the kind of interpolation as a string or as an integer specifying
            the order of the spline interpolator to use. The string has to be one of 'linear',
            'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next' or
            'resample'. 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation
            of zeroth, first, second or third order; 'previous' and 'next' simply return the
            previous or next value of the point; 'nearest-up' and 'nearest' differ when
            interpolating half-integers (e.g. 0.5, 1.5) in that 'nearest-up' rounds up and
            'nearest' rounds down.

            - 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
              'previous', 'next' are used in `scipy.interpolate.interp1d` as the `kind` argument.
            - 'resample' is used to resample the data using `scipy.signal.resample`.

            Default is 'linear'.
        hz (float): Frame rate of the video (default: 30.0)
        fill_value (str): Specifies how to handle out-of-bounds values during interpolation.
            - 'raise': Raise an error if the requested value is outside the bounds of input data.
            - 'clamp': Clamp the values to the nearest valid value within the bounds of input data.
            - 'extrapolate': Extrapolate values outside the bounds of the input data.

    Returns:
        (dict[str, np.ndarray]): Dictionary containing the synchronized EEG data, which has been
            resampled to match the number of given `num_samples`, timestamps and a mask of
            valid indices in the case of truncation.
    """
    # Extract EEG data and timestamps
    eeg_data = eeg_dict["data"]  # Shape: (67, 3000)
    eeg_timestamps = eeg_dict["timestamps"]  # Shape: (3000,)

    # Calculate video frame timestamps (assuming constant frame rate)
    # First, determine start time from the EEG data
    eeg_start_time = eeg_timestamps[0]
    video_duration = (num_samples - 1) / hz  # in seconds

    # Create evenly spaced timestamps for video frames
    # We align the first and last frame with the EEG timing
    frame_timestamps = np.linspace(eeg_start_time, eeg_start_time + video_duration, num_samples)

    # Ensure frame timestamps are within EEG timestamp range
    if frame_timestamps[-1] > eeg_timestamps[-1]:
        logger.warning(
            f"Duration ({frame_timestamps[-1]}) exceeds "
            f"EEG recording ({eeg_timestamps[-1]}). Truncating to EEG duration.",
        )
        valid_frame_indices = frame_timestamps <= eeg_timestamps[-1]
        frame_timestamps = frame_timestamps[valid_frame_indices]
        num_samples = len(frame_timestamps)
    else:
        valid_frame_indices = np.ones(num_samples, dtype=bool)

    logger.info(
        f"Creating {num_samples} EEG samples from {eeg_timestamps.shape[0]} original "
        f"samples ({eeg_timestamps[:5]}... {eeg_timestamps[-5:]}) into "
        f"frames ({frame_timestamps[:5]}... {frame_timestamps[-5:]})"
    )

    # Method 1: Linear interpolation (most common approach)
    synchronized_eeg = np.zeros((eeg_data.shape[0], num_samples))

    for channel in range(eeg_data.shape[0]):
        if method == "resample":
            synchronized_eeg[channel] = resample(eeg_data[channel], num_samples)
        else:
            # Create interpolation function for this EEG channel
            interp_func = interpolate.interp1d(
                eeg_timestamps,
                eeg_data[channel],
                kind=method,  # type: ignore[PylancereportArgumentType]
                bounds_error=fill_value == "raise",
                fill_value=(eeg_data[channel, 0], eeg_data[channel, -1])
                if fill_value != "extrapolate"
                else fill_value,  # type: ignore[PylancereportArgumentType]
            )
            # Apply interpolation to get EEG values at frame timestamps
            synchronized_eeg[channel] = interp_func(frame_timestamps)

    return {
        "data": synchronized_eeg,  # Shape: (67, num_frames)
        "timestamps": frame_timestamps,  # Shape: (num_frames,)
        "valid_indices": valid_frame_indices,  # Shape: (num_frames,)
    }


def create_dig_montage(
    electrodes: set[str] | list[str],
    montage_name: str = "standard_1020",
    *,
    uppercase: bool = False,
) -> mne.channels.DigMontage:
    """Create an MNE montage object with the specified `electrodes`.

    This function checks for missing electrodes and removes them from the montage to match the
    provided set of `electrodes`.

    Args:
        electrodes (set[str] | list[str]): Set of electrode names to include in the montage.
        montage_name (str): Name of the standard montage to use. Default is 'standard_1020'.
        uppercase (bool): Whether to convert electrode names to uppercase in the resulting montage.
            Default is False.

    Returns:
        mne.channels.DigMontage: MNE montage object with the specified electrodes.

    Raises:
        TypeError: If `electrodes` is not a set.
        ValueError: If the new montage keys do not match the provided `electrodes`.
    """
    if not isinstance(electrodes, (set, list)):
        raise TypeError(f"Expected 'electrodes' to be a set or list, got {type(electrodes)}")

    logger.info(f"Creating a new MNE montage for {electrodes} using montage '{montage_name}'")
    eeg_montage = mne.channels.make_standard_montage(montage_name)
    eeg_montage_keys = list(eeg_montage.get_positions()["ch_pos"].keys())

    montage_dict = {k.lower() for k in eeg_montage_keys}
    electrodes_lowered = {k.lower() for k in electrodes}

    # Get the keys that are not in the intersection of electrodes_lowered and montage_dict
    # This is the outer join of the two sets
    missing_keys = electrodes_lowered.symmetric_difference(montage_dict)
    logger.info(f"Length of electrodes: {len(electrodes)}")
    logger.info(f"Length of montage ({montage_name}): {len(montage_dict)}")

    # List the electrodes you want to remove
    # Map the missing_keys back to their original names
    electrodes_to_remove = []
    eeg_montage_keys = list(eeg_montage.get_positions()["ch_pos"].keys())
    for key in missing_keys:
        for k in eeg_montage_keys:
            if key == k.lower():
                electrodes_to_remove.append(k)
                break

    logger.info(f"Electrodes to remove ({len(electrodes_to_remove)}): {electrodes_to_remove}")

    # Get the electrode names and positions
    all_electrode_names = eeg_montage.ch_names
    all_positions = eeg_montage.get_positions()["ch_pos"]

    # Find the indices of the electrodes to remove
    indices_to_remove = [
        all_electrode_names.index(elec)
        for elec in electrodes_to_remove
        if elec in all_electrode_names
    ]

    # Create a boolean mask to keep the electrodes you want
    keep_mask = np.ones(len(all_electrode_names), dtype=bool)
    keep_mask[indices_to_remove] = False

    # Filter the electrode names and positions
    kept_positions = OrderedDict()
    for i, name in enumerate(all_electrode_names):
        if keep_mask[i]:
            kept_positions[name] = all_positions[name]

    # Create a new montage with the kept electrodes
    new_montage = mne.channels.make_dig_montage(
        ch_pos=kept_positions
        if not uppercase
        else {k.upper(): v for k, v in kept_positions.items()},
        # coord_frame="head",  # Assuming the original was in head coordinates
    )

    # Check that new_montage.ch_names has only the electrode_dict keys
    new_montage_keys = {ch.lower() for ch in new_montage.ch_names}
    if new_montage_keys != set(electrodes_lowered):
        raise ValueError(
            f"New montage keys ({len(new_montage_keys)}) do not match the electrodes keys "
            f"({len(set(electrodes_lowered))}): {new_montage_keys} != {set(electrodes_lowered)}.",
        )

    return new_montage


def plot_mne_montage(
    eeg_montage: mne.channels.DigMontage,
    *,
    title: str = "EEG Montage",
    kind: Literal["topomap", "3d"] = "topomap",
    sphere: str | int = "auto",
    filepath: Path | str = "eeg_montage.png",
    show: bool = True,
) -> None:
    """Plot the MNE EEG montage in 2d (topomap) or 3d."""
    fig = eeg_montage.plot(kind=kind, sphere=sphere, show=False)
    if fig is not None and isinstance(fig, Figure):
        if kind == "3d":
            fig.gca().view_init(elev=30, azim=-60)  # type: ignore[reportAttributeAccessIssue]

        if title:
            if kind == "3d":
                fig.gca().set_title(title, y=1.025)
            else:
                fig.gca().set_title(title)

        fig.tight_layout()

        # Save the plot as an image
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving EEG montage plot to {filepath}")
            fig.savefig(
                filepath,
                bbox_inches="tight",
                dpi=300,
            )

        # Show the plot
        if show:
            plt.show()
        plt.close(fig)
    else:
        logger.warning("No figure was returned from `eeg_montage.plot(...)`")


def plot_eeg_montage_3d(
    eeg_montage: mne.channels.DigMontage,
    *,
    color_fp1_fp2: bool = True,
    color_highest_z: bool = True,
    figsize: tuple[int, int] = (16, 8),
    title: str = "EEG Montage 3D",
    filepath: Path | str = "eeg_montage_3d.png",
    show: bool = True,
    dpi: int = 300,
) -> None:
    """Plot the MNE EEG montage in 3D with top-down view and normal 3D view.

    Args:
        eeg_montage (mne.channels.DigMontage): The EEG montage to plot.
        color_fp1_fp2 (bool): Whether to color Fp1 and Fp2 electrodes in green.
        color_highest_z (bool): Whether to color the electrode with the highest Z coord. in red.
        figsize (tuple[int, int]): Size of the figure in inches.
        title (str): Title of the plot.
        filepath (Path | str): Path to save the plot image.
        show (bool): Whether to show the plot.
        dpi (int): Dots per inch for the saved image.
    """
    # Get electrode coordinates and channel names from the montage
    positions = eeg_montage.get_positions()
    ch_pos = positions.get("ch_pos", None)
    if ch_pos is None or not isinstance(ch_pos, dict):
        raise ValueError(
            "Montage does not contain valid 'ch_pos' dictionary for electrode positions."
        )

    electrode_coords_3d = np.array(list(ch_pos.values()))
    channel_names = eeg_montage.ch_names

    # Identify indices for specific electrodes
    highest_z_index = np.argmax(electrode_coords_3d[:, 2])
    fp1_index = channel_names.index("Fp1")
    fp2_index = channel_names.index("Fp2")

    # Create a figure with 1 row and 2 columns
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # --- Left subplot: top-down view ---
    ax1: Axes3D = fig.add_subplot(121, projection="3d")  # type: ignore[reportAssignmentType]

    # Filter electrode_coords_3d to not include fp1 index and fp2 index
    remove_idxs = []
    if color_fp1_fp2:
        remove_idxs = [fp1_index, fp2_index]
    if color_highest_z:
        remove_idxs.append(highest_z_index)

    if remove_idxs:
        filtered_coords = np.delete(
            electrode_coords_3d,
            remove_idxs,
            axis=0,
        )
    else:
        filtered_coords = electrode_coords_3d

    ax1.scatter(
        filtered_coords[:, 0],
        filtered_coords[:, 1],
        filtered_coords[:, 2],  # type: ignore[reportArgumentType]
        c="blue",
        marker="o",
        s=50,
    )
    ax1.scatter(
        electrode_coords_3d[highest_z_index, 0],
        electrode_coords_3d[highest_z_index, 1],
        electrode_coords_3d[highest_z_index, 2],
        c="red",
        marker="o",
        s=100,
    )
    ax1.scatter(
        electrode_coords_3d[fp1_index, 0],
        electrode_coords_3d[fp1_index, 1],
        electrode_coords_3d[fp1_index, 2],
        c="green",
        marker="o",
        s=100,
    )
    ax1.scatter(
        electrode_coords_3d[fp2_index, 0],
        electrode_coords_3d[fp2_index, 1],
        electrode_coords_3d[fp2_index, 2],
        c="green",
        marker="o",
        s=100,
    )

    # Add labels for each electrode
    for name, coord in zip(channel_names, electrode_coords_3d, strict=True):
        ax1.text(coord[0], coord[1], coord[2], name, size=9, color="black")

    # Remove all ticks and labels as well as the axis outline
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_axis_off()

    ax1.set_title("Top-Down View (XY Plane)", y=0.95)

    # Set view from above: look down along the Z axis
    ax1.view_init(elev=-90, azim=90)

    # --- Right subplot: normal 3D view ---
    ax2: Axes3D = fig.add_subplot(122, projection="3d")  # type: ignore[reportAssignmentType]
    ax2.scatter(
        filtered_coords[:, 0],
        filtered_coords[:, 1],
        filtered_coords[:, 2],  # type: ignore[reportArgumentType]
        c="blue",
        marker="o",
        s=50,
    )
    if color_highest_z:
        ax2.scatter(
            electrode_coords_3d[highest_z_index, 0],
            electrode_coords_3d[highest_z_index, 1],
            electrode_coords_3d[highest_z_index, 2],
            c="red",
            marker="o",
            s=100,
        )
    if color_fp1_fp2:
        ax2.scatter(
            electrode_coords_3d[fp1_index, 0],
            electrode_coords_3d[fp1_index, 1],
            electrode_coords_3d[fp1_index, 2],
            c="green",
            marker="o",
            s=100,
        )
        ax2.scatter(
            electrode_coords_3d[fp2_index, 0],
            electrode_coords_3d[fp2_index, 1],
            electrode_coords_3d[fp2_index, 2],
            c="green",
            marker="o",
            s=100,
        )

    # Add labels for each electrode
    for name, coord in zip(channel_names, electrode_coords_3d, strict=True):
        ax2.text(coord[0], coord[1], coord[2], name, size=9, color="black")

    ax2.set_title("3D View", y=0.95)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")

    if title:
        fig.suptitle(title, y=0.9)

    fig.tight_layout()

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving EEG montage 3D plot to {filepath}")
        fig.savefig(
            filepath,
            bbox_inches="tight",
            dpi=dpi,
        )
    if show:
        plt.show()

    plt.close(fig)


def plot_channel(
    data: np.ndarray,
    timestamps: np.ndarray | None = None,
    channel_index: int = 0,
    events: pd.DataFrame | None = None,
    first_timestamp: float | None = None,
    overlay_data: np.ndarray | None = None,
    *,
    data_label: str = "Downsampled",
    overlay_data_label: str = "High-Res",
) -> None:
    """Plot `data` for the specified `channel_index` and (if provided) overlay the event markers.

    Args:
        data (np.ndarray): EEG data of shape (n_channels, n_samples)
        timestamps (np.ndarray, optional): 1D (or 2D with one row) array of timestamps
            (in seconds). If None, the function will generate a time axis based on the number of
            samples.
        channel_index (int, optional): Index of the channel to plot (default is 0)
        events (pd.DataFrame, optional): DataFrame with an event column 'sample' indicating the
            sample number of the event.
        first_timestamp (float, optional): If provided, the first timestamp of the recording and
            will be used to normalize the timestamps with. If not provided, `timestamps[0]`
            / `timestamps.min()` will be used to normalize the timestamps and will thus start
            from 0.
        overlay_data (np.ndarray, optional): Optional (perhaps high-resolution) signal of same
            shape `(n_channels, n_samples_high)`.
        data_label (str, optional): Label for the downsampled data in the plot legend.
            Defaults to "Downsampled".
        overlay_data_label (str, optional): Label for the overlay data in the plot legend.
            Defaults to "High-Res".
    """
    if timestamps is None:
        # Generate timestamps based on the number of samples
        num_samples = data.shape[1]
        timestamps = np.arange(num_samples)
        xaxis = "Samples"
    else:
        xaxis = "Time (s)"

    plt.figure(figsize=(15, 5))

    # Ensure we have a 1D array for the timestamps
    times = timestamps.flatten() if timestamps.ndim > 1 else timestamps

    # Normalize the timestamps to start from 0
    times = times - times[0] if first_timestamp is None else times - first_timestamp
    plt.plot(
        times,
        data[channel_index],
        label=data_label if overlay_data is not None else None,
        linewidth=2,
    )

    # Overlay high-resolution signal if provided
    if overlay_data is not None:
        n_high = overlay_data.shape[1]
        # Interpolate same time range as the downsampled one
        high_res_times = np.linspace(times[0], times[-1], n_high)
        plt.plot(
            high_res_times,
            overlay_data[channel_index],
            label=overlay_data_label,
            alpha=0.5,
        )

    # If event information is available, mark the events on the plot
    if events is not None:
        for _, row in events.iterrows():
            sample_idx = int(row["sample"])
            if sample_idx < len(times):
                event_time = times[sample_idx]
                plt.axvline(event_time, color="red", linestyle="--", alpha=0.7)
                plt.text(
                    event_time,
                    np.max(data[channel_index]),
                    row["id"],
                    rotation=90,
                    verticalalignment="top",
                    color="red",
                )

    plt.xlabel(xaxis)
    plt.ylabel("Amplitude")
    plt.title(f"EEG Data - Channel {channel_index}")
    plt.legend()
    plt.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_segmented_eeg(
    eeg_data: dict[str, np.ndarray],
    segmented_data: dict[str, dict[str, np.ndarray]],
    channel_idx: int = 0,
    time_window: tuple[float, float] | None = None,
    figsize: tuple = (12, 6),
    alpha: float = 0.3,
    colors: dict | None = None,
    title: str | None = None,
    highlight_segments: list[str] | None = None,
) -> None:
    """Plot EEG data with highlighted segments for different expressions.

    Args:
        eeg_data (dict): Dictionary containing 'data' and 'timestamps' arrays for the full EEG
            signal.
        segmented_data (dict): Dictionary with segment names as keys and segmented EEG data as
            values (which are dictionaries with 'data' and 'timestamps').
        channel_idx (int, optional): Index of the channel to plot. Defaults to 0.
        time_window (tuple, optional): Time window to display in seconds (start, end).
                                       If None, shows the entire signal. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 6).
        alpha (float, optional): Transparency of the highlighted segments. Defaults to 0.3.
        colors (dict, optional): Dictionary mapping expression names to colors.
                                 If None, uses default color cycle. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        highlight_segments (list, optional): List of segments to highlight. These strings must
            match the keys in `segmented_data`. If None, all segments are highlighted.
    """
    # Extract data
    data = eeg_data["data"]
    timestamps = eeg_data["timestamps"]

    # Check if timestamps are 1D or 2D
    if timestamps.ndim == 2 and timestamps.shape[0] != 1:
        raise ValueError("Timestamps should be either 1D or 2D with one row.")

    # If timestamps are 2D, take the first row
    if timestamps.ndim == 2:
        timestamps = timestamps[0, :]

    # Normalize timestamps to start from 0
    start_time = timestamps[0]
    relative_timestamps = timestamps - start_time

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Apply time window filter if specified
    if time_window is not None:
        start_sec, end_sec = time_window
        mask = (relative_timestamps >= start_sec) & (relative_timestamps <= end_sec)
        plot_data = data[channel_idx, mask]
        plot_times = relative_timestamps[mask]
    else:
        start_sec, end_sec = None, None  # Make it throw an error if used later
        plot_data = data[channel_idx, :]
        plot_times = relative_timestamps[:]

    logger.info(
        f"Plotting EEG channel {channel_idx} from {plot_times[0]:.2f}s to {plot_times[-1]:.2f}s "
        f"with {len(plot_data)} data points."
    )

    # Plot the full signal
    ax.plot(plot_times, plot_data, color="black", alpha=0.7, label="EEG Signal")

    # Generate colors if not provided
    if colors is None:
        # Use default color cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        default_colors = prop_cycle.by_key()["color"]
        colors = {
            expr: default_colors[i % len(default_colors)]
            for i, expr in enumerate(segmented_data.keys())
        }

    # Create legend patches
    legend_patches = []

    # Highlight segments for each expression
    for expr_name, segment in segmented_data.items():
        if highlight_segments is not None and expr_name not in highlight_segments:
            continue

        expr_timestamps = segment["timestamps"] - start_time

        # Skip if segment is outside the time window
        if time_window is not None and (
            expr_timestamps[-1] < start_sec or expr_timestamps[0] > end_sec
        ):
            continue

        # Plot the segment
        ax.axvspan(
            expr_timestamps[0],
            expr_timestamps[-1],
            color=colors[expr_name],
            alpha=alpha,
        )

        # Add to legend
        legend_patches.append(mpatches.Patch(color=colors[expr_name], label=expr_name))

    # Dynamically calculate number of columns in legend to fit horizontally
    max_legend_width = figsize[0] * fig.dpi  # width in pixels
    avg_label_width = 300  # estimated average label width in pixels
    n_cols = max(1, math.floor(max_legend_width / avg_label_width))

    # Place the legend underneath the plot
    ax.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=n_cols,
        frameon=False,
    )

    # Set labels and title
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(f"Amplitude (Channel {channel_idx})")
    ax.set_title(title if title else f"EEG Channel {channel_idx}")
    ax.grid(visible=True, alpha=0.3)

    fig.tight_layout(rect=(0, 0.05, 1, 1))  # leave space at bottom for legend
    plt.show()
    plt.close(fig)


def plot_eeg_around_event(
    eeg_data_segment: np.ndarray,
    eeg_timestamps_segment: np.ndarray,
    video_path: Path,
    *,
    relative_event_time: float,
    eeg_time_range: tuple[float, float] | None = None,
    video_time_range: tuple[float, float] | None = None,
    channel_to_plot: int | Literal["average"] = "average",
    num_frames_around_event: int = 2,
    frame_offset: int = 15,
    display_relative_time: bool = True,
    output_path: Path | None = None,
    plt_style: str = "default",
    dpi: int = 100,
    show: bool = True,
) -> None:
    """Create a static plot with three video frames around an event, aligned with the EEG signal.

    Args:
        eeg_data_segment (np.ndarray): The cropped EEG data for the segment.
        eeg_timestamps_segment (np.ndarray): The timestamps for the EEG segment.
        video_path (Path): Path to the original, full-length video file.
        relative_event_time (float): The time (in seconds) of the main event relative to the start
            of the `time_range` (or 0 if not provided).
        eeg_time_range (tuple[float, float] | None, optional): The time range in **seconds** of the
            EEG data to consider. The event must be within this range. If None, the entire segment
            is used. Defaults to None.
        video_time_range (tuple[float, float] | None, optional): The time range of the
            video to consider. The event must be within this range. Defaults to None.
        channel_to_plot (int | Literal["average"], optional): The EEG data to plot.
        num_frames_around_event (int): Number of frames to show on EACH side of the
            central event frame. E.g., `2` will show 5 frames total (T-2, T-1, T, T+1, T+2).
            Defaults to 2.
        frame_offset (int, optional): The number of frames to look before and after the main event
            frame. This works on top of the `time_range` parameter. If `time_range` is provided,
            this will be used to extract frames relative to the `event_time_in_video`. If not
            provided, the frames will be extracted from the `video_path` at the specified
            `event_time_in_video` +/- `frame_offset` (in seconds). Defaults to 15.
        display_relative_time (bool, optional): If True, display the time in the frames title and
            in the xlim as relative (from `0` always), otherwise use absolute time in the video.
        output_path (Path | None, optional): If provided, save the plot to this path.
        plt_style (str, optional): Matplotlib style to use. Defaults to "default".
        dpi (int, optional): DPI for the saved plot. Defaults to 150.
        show (bool, optional): Whether to show the plot interactively. Defaults to True.
    """
    if not video_path.exists():
        logger.error(f"Video file not found at: {video_path}")
        return

    # Check that eeg_timestamps is strictly increasing and raise an error if not
    _ = is_strictly_increasing(eeg_timestamps_segment, from_zero=False, raise_error=True)

    # Keep the original start time for absolute calculations
    segment_start_in_video = eeg_timestamps_segment[0]

    # Warn that eeg timestamps are strictly increasing but not from zero
    if eeg_timestamps_segment[0] != 0:
        logger.warning(
            "EEG timestamps must be strictly increasing and start from zero. "
            "Will make them relative to the first timestamp."
        )
        # Make timestamps relative to the first timestamp
        eeg_timestamps_segment = eeg_timestamps_segment - eeg_timestamps_segment[0]

    # --- 1. Prepare EEG Signal ---
    if channel_to_plot == "average":
        eeg_signal = np.mean(eeg_data_segment, axis=0)
        y_label = "Average Amplitude (μV)"
    else:
        if not 0 <= channel_to_plot < eeg_data_segment.shape[0]:
            raise ValueError(f"Channel index {channel_to_plot} is out of bounds.")
        eeg_signal = eeg_data_segment[channel_to_plot, :]
        y_label = f"Channel {channel_to_plot} Amplitude (μV)"

    # Crop EEG signal to the specified time range if provided
    if eeg_time_range is not None:
        start_time, end_time = eeg_time_range
        if not (0 <= relative_event_time + start_time <= end_time):
            logger.error(
                f"Event time {relative_event_time + start_time} is outside the EEG time range "
                f"{start_time} to {end_time}.",
            )
            return

        # Filter EEG timestamps and signal to the specified range
        mask = (eeg_timestamps_segment >= start_time) & (eeg_timestamps_segment <= end_time)
        eeg_timestamps_segment = eeg_timestamps_segment[mask]
        eeg_signal = eeg_signal[mask]

    # --- 2. Extract Video Frames based on absolute time ---
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps > 0:
        logger.error("Could not determine video FPS.")
        cap.release()
        return

    time_per_frame = 1.0 / video_fps

    clip_start_time = video_time_range[0] if video_time_range else 0.0
    absolute_event_time = clip_start_time + relative_event_time
    center_frame_num = round(absolute_event_time / time_per_frame)

    frame_indices = range(-num_frames_around_event, num_frames_around_event + 1)
    frame_numbers_to_get = [center_frame_num + (i * frame_offset) for i in frame_indices]

    frames = {}
    for frame_num in frame_numbers_to_get:
        target_msec = (frame_num / video_fps) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
        ret, frame = cap.read()
        if ret:
            frames[frame_num] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    # --- 3. Create the Plot Layout ---
    total_frames_to_plot = len(frame_numbers_to_get)
    with plt.style.context(plt_style):
        fig, axes = plt.subplots(
            2,
            total_frames_to_plot,
            dpi=dpi,
            figsize=(4 * total_frames_to_plot, 8),
            gridspec_kw={"height_ratios": [1, 1]},
        )
        # Combine the bottom axes into a single one for the EEG plot
        gs = axes[0, 0].get_gridspec()
        for ax in axes[1, :]:
            ax.remove()
        ax_eeg = fig.add_subplot(gs[1, :])

        # --- 4. Plot Video Frames and EEG Alignment Lines ---
        for i, frame_num in enumerate(frame_numbers_to_get):
            ax_img = axes[0, i]
            current_frame: np.ndarray | None = frames.get(frame_num)

            time_at_frame = frame_num * time_per_frame
            display_time = time_at_frame - (absolute_event_time - relative_event_time)

            if current_frame is not None:
                ax_img.imshow(current_frame)
                if display_relative_time:
                    ax_img.set_title(f"T = {display_time:.2f}s")
                else:
                    ax_img.set_title(f"T = {time_at_frame:.2f}s")
            else:
                ax_img.set_title(f"Frame {frame_num} not found")
            ax_img.axis("off")

            # Each frame's time window is from T - 0.5*duration to T + 0.5*duration
            frame_start_time = display_time - (time_per_frame / 2.0)
            frame_end_time = display_time + (time_per_frame / 2.0)
            ax_eeg.axvline(x=frame_start_time, color="r", linestyle="--", alpha=0.7)
            ax_eeg.axvline(x=frame_end_time, color="r", linestyle="--", alpha=0.7)

        # --- 5. Plot EEG Signal ---
        ax_eeg.plot(eeg_timestamps_segment, eeg_signal, color="#3498db")
        ax_eeg.set_ylabel(y_label)

        if display_relative_time:
            ax_eeg.set_xlabel("Relative Time (s)")
            ax_eeg.set_xlim(eeg_timestamps_segment[0], eeg_timestamps_segment[-1])
        else:
            # If displaying absolute time, the x-axis shows the original video timeline
            ax_eeg.set_xlabel("Absolute Time (s)")
            # We need to create new tick positions and labels
            # Get the current relative ticks
            relative_ticks = ax_eeg.get_xticks()
            # Calculate the absolute time for these tick positions
            absolute_tick_labels = [
                f"{tick + segment_start_in_video:.2f}" for tick in relative_ticks
            ]
            # Set the new labels
            ax_eeg.set_xticklabels(absolute_tick_labels)
            ax_eeg.set_xlim(eeg_timestamps_segment[0], eeg_timestamps_segment[-1])

        # Crop to eeg_time_range if provided (applied to the relative timeline)
        if eeg_time_range is not None:
            ax_eeg.set_xlim(eeg_time_range[0], eeg_time_range[1])

        ax_eeg.grid(visible=True, alpha=0.3)
        ax_eeg.axvline(
            x=relative_event_time,
            color="g",
            linestyle="-",
            linewidth=2,
            label=f"Event at {relative_event_time:.2f}s",
            alpha=0.5,
        )
        ax_eeg.legend()

    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.success(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def compute_signal_energy(
    eeg_data: np.ndarray,
    window_size: int = 100,
    overlap: int = 50,
    *,
    normalize: bool = True,
    compute_power: bool = False,
) -> np.ndarray:
    """Compute the energy (or power) of EEG signals over sliding windows.

    This function calculates the signal energy or power for each EEG channel (electrode)
    using overlapping windows, and optionally normalizes the result per channel to the [0, 1]
    range.

    Args:
        eeg_data (np.ndarray): EEG signal of shape (n_channels, n_samples).
        window_size (int): Size of the sliding window (in samples).
        overlap (int): Number of samples that windows overlap. Must be less than window_size.
        normalize (bool): If True, normalize the energy/power per channel to the [0, 1] range.
        compute_power (bool): If True, computes average power (mean of squared values per window)
                              instead of total energy (sum of squares).

    Returns:
        np.ndarray: Array of shape (n_channels, n_windows) containing the energy or power
                    values per window, optionally normalized per channel.

    Raises:
        ValueError: If overlap is greater than or equal to window_size.
    """
    n_channels, n_samples = eeg_data.shape
    step = window_size - overlap

    if step <= 0:
        raise ValueError("Overlap must be smaller than window_size")

    n_windows = (n_samples - overlap) // step
    energy = np.zeros((n_channels, n_windows))

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > n_samples:
            end = n_samples
            if end <= start:
                continue  # Skip invalid window

        window = eeg_data[:, start:end]
        if window.shape[1] == 0:
            continue

        if compute_power:
            energy[:, i] = np.mean(window**2, axis=1)
        else:
            energy[:, i] = np.sum(window**2, axis=1)

    if normalize:
        energy_min = np.min(energy, axis=1, keepdims=True)
        energy_max = np.max(energy, axis=1, keepdims=True)
        range_ = energy_max - energy_min + 1e-10  # avoid div-by-zero
        energy_norm = (energy - energy_min) / range_
        energy_norm = np.clip(energy_norm, 0, 1)

        # Optional: Zero-out channels with nearly zero range
        zero_var = (range_ <= 1e-10).flatten()
        energy_norm[zero_var, :] = 0.0
        return energy_norm

    return energy


def calculate_global_energy_range(
    segmented_data: dict[str, dict[str, np.ndarray]],
    sfreq: float,
    window_duration: float = 0.1,
    overlap_ratio: float = 0.5,
) -> tuple[StandardScaler | None, tuple[float, float]]:
    """Calculate the global min and max of standardized signal energy across all EEG segments.

    This is useful for creating a consistent color scale (vmin, vmax) for
    animations or comparative plots.

    Args:
        segmented_data (dict): A dictionary of all EEG segments to consider.
        sfreq (float): The sampling frequency of the EEG data.
        window_duration (float): Duration of the sliding window for energy calculation.
        overlap_ratio (float): Overlap ratio between consecutive windows.

    Returns:
        tuple[StandardScaler | None, tuple[float, float]]: A tuple containing the global scaler,
            global minimum (vmin), and global maximum (vmax) standardized energy values.
    """
    logger.info("Calculating global energy range across all segments...")
    all_energy_maps = []
    window_size = int(window_duration * sfreq)
    overlap = int(window_size * overlap_ratio)

    for segment in segmented_data.values():
        if segment["data"].shape[1] < window_size:
            continue
        energy_map = compute_signal_energy(segment["data"], window_size, overlap)
        # We need to stack them for the scaler, so transpose to (n_windows, n_channels)
        all_energy_maps.append(energy_map.T)

    if not all_energy_maps:
        logger.error("No energy data could be calculated. Cannot determine range.")
        return None, (-1.0, 1.0)

    # Concatenate all windows from all expressions into one big array
    concatenated_energies = np.vstack(all_energy_maps)

    # Standardize the entire dataset of energy windows
    scaler = StandardScaler()
    scaled_energies = scaler.fit_transform(concatenated_energies)

    vmin, vmax = scaled_energies.min(), scaled_energies.max()
    logger.success(f"Global energy range (vmin, vmax): ({vmin:.2f}, {vmax:.2f})")

    return scaler, (vmin, vmax)


# TODO: Replace with https://yasa-sleep.org/generated/yasa.bandpower.html
# https://raphaelvallat.com/bandpower.html
def compute_spectral_power(
    eeg_data: np.ndarray,
    *,
    fs: float = 500,
    freq_band: tuple[float, float] = (8, 12),
    window_size: int = 250,
    overlap: int = 125,
) -> np.ndarray:
    """Compute the spectral power in a specific frequency band.

    Normalizes power per channel to 0-1 range.

    Args:
        eeg_data (np.ndarray): EEG signal of shape (n_channels, n_samples).
        fs (float): Sampling frequency of the EEG data in Hz.
        freq_band (tuple[float, float]): Frequency band to compute power for (low, high).
        window_size (int): Size of the sliding window (in samples).
        overlap (int): Number of samples that windows overlap. Must be less than window_size.

    Returns:
        np.ndarray: Array of shape (n_channels, n_windows) containing the power values per
                    window, normalized per channel to the [0, 1] range.

    Raises:
        ValueError: If overlap is greater than or equal to window_size.
        ValueError: If the frequency band is invalid (low >= high).
    """
    n_channels, n_samples = eeg_data.shape
    step = window_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than window_size")
    n_windows = (n_samples - overlap) // step  # Adjusted calculation

    power = np.zeros((n_channels, n_windows))

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > n_samples:  # Ensure window doesn't exceed data length
            end = n_samples
            if start >= end:
                continue  # Skip if window is empty/invalid

        window = eeg_data[:, start:end]
        if window.shape[1] < window_size:  # Pad if window is too short for welch nperseg
            padding = window_size - window.shape[1]
            window = np.pad(window, ((0, 0), (0, padding)), "constant")
        if window.shape[1] == 0:
            continue  # Skip empty windows

        for ch in range(n_channels):
            # Use nperseg=window.shape[1] if window is shorter than original window_size
            current_nperseg = min(window_size, window.shape[1])
            if current_nperseg == 0:
                continue  # Skip if segment is empty

            try:
                freqs, psd = welch(
                    window[ch], fs=fs, nperseg=current_nperseg, noverlap=0
                )  # Use no overlap within welch
            except ValueError as e:
                logger.error(f"Welch error on ch {ch}, window {i}: {e}. Skipping.")
                continue  # Skip this window/channel if welch fails

            # Find indices corresponding to the frequency band
            idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])

            if np.any(idx_band):
                # Compute mean power in the band
                power[ch, i] = np.mean(psd[idx_band])
            else:
                power[ch, i] = 0  # No power if band is empty

    # Normalize power to 0-1 range per channel
    power_min = np.min(power, axis=1, keepdims=True)
    power_max = np.max(power, axis=1, keepdims=True)
    power_norm = (power - power_min) / (power_max - power_min + 1e-10)
    power_norm = np.clip(power_norm, 0, 1)

    # Handle channels with zero variance
    zero_variance_channels = (power_max - power_min) <= 1e-10
    power_norm[zero_variance_channels[:, 0], :] = 0.0

    return power_norm


def map_activity_to_frames(
    activity: np.ndarray, n_windows: int, step: int, fs: float, fps: float, n_samples: int
) -> tuple[np.ndarray, int]:
    """Map activity values (n_channels, n_windows) to video frames (n_channels, total_frames).

    Args:
        activity (np.ndarray): Activity values with shape (n_channels, n_windows).
        n_windows (int): Number of windows in the activity data.
        step (int): Step size between windows in samples.
        fs (float): Sampling frequency of the EEG data in Hz.
        fps (float): Frames per second for the video.
        n_samples (int): Total number of samples in the EEG data.

    Returns:
        (tuple[np.ndarray, int]): A tuple containing:
            - activity_per_frame: Activity values mapped to each frame with shape
                `(n_channels, total_frames)`.
            - total_frames: Total number of frames in the video.
    """
    # Compute the total duration of the EEG data in seconds
    duration = n_samples / fs

    # Compute the total number of frames needed for the video
    total_frames = int(np.ceil(duration * fps))  # Use ceil to cover the full duration

    # Compute the time point (in seconds) for each frame
    frame_times = np.arange(total_frames) / fps

    # Compute the start time (in seconds) for each activity window
    window_start_times = (np.arange(n_windows) * step) / fs

    # Find which window index corresponds to each frame time
    # Use searchsorted to find the insertion points
    # 'right' ensures that a frame time belongs to the window that started just before or at it
    window_indices = np.searchsorted(window_start_times, frame_times, side="right") - 1
    window_indices = np.clip(window_indices, 0, n_windows - 1)  # Clip indices to valid range

    # Map activity values to each frame
    activity_per_frame = activity[:, window_indices]

    return activity_per_frame, total_frames
