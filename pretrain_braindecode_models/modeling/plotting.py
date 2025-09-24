"""Utility functions for plotting training history and metrics."""

import math
import re
import string
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textalloc as ta
import torch
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import (
    FuncFormatter,
    LogFormatterMathtext,
    LogLocator,
)
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pretrain_braindecode_models.config import DEVICE, logger
from pretrain_braindecode_models.modeling.models import extract_model_name
from pretrain_braindecode_models.utils.colors import darken_color, get_n_colors
from pretrain_braindecode_models.utils.custom_types import (
    PredictorProtocol,
    TrainingHistoryClassification,
)
from pretrain_braindecode_models.utils.loading import load_json
from pretrain_braindecode_models.utils.misc import (
    get_nested_value,
    metadata_is_filtered_out,
)
from pretrain_braindecode_models.utils.signal import (
    exponential_moving_average,
)


def _get_dynamic_ylim_and_outliers(
    *data_arrays: np.ndarray, iqr_multiplier: float = 2.0
) -> tuple[tuple[float, float] | None, list[np.ndarray], list[np.ndarray]]:
    """Calculate a dynamic y-axis limit based on the IQR method to handle outliers.

    Args:
        *data_arrays: One or more 1D NumPy arrays of data to analyze (e.g., train_loss, val_loss).
        iqr_multiplier: How many IQRs above the 3rd quartile (Q3) to set the
                        upper limit. A larger value is more permissive.
                        Common values are 1.5 (standard outlier) to 3.0.

    Returns:
        A tuple containing:
        - ylim (tuple | None): The calculated (min, max) for the y-axis. None if no valid data.
        - clipped_data (list): A list of the input arrays, clipped to the new ylim.
        - outlier_indices (list): A list of boolean arrays, one for each input array,
                                  where True indicates an outlier.
    """
    # Combine all data to find a common scale, ignoring NaNs and Infs
    combined_data = np.concatenate(data_arrays)
    valid_data = combined_data[np.isfinite(combined_data)]

    if valid_data.size < 2:  # Not enough data to calculate stats
        return (
            None,
            list(data_arrays),
            [np.zeros_like(d, dtype=bool) for d in data_arrays],
        )

    # Calculate IQR statistics
    q1 = np.percentile(valid_data, 25)
    q3 = np.percentile(valid_data, 75)
    iqr = q3 - q1

    # Define the upper bound for the y-axis
    # Lower bound can be the data min or slightly below
    upper_bound = q3 + iqr * iqr_multiplier
    lower_bound = np.min(valid_data)

    # Add a small padding to the y-axis for better visualization
    padding = (upper_bound - lower_bound) * 0.05
    ylim = (lower_bound - padding, upper_bound + padding)

    # Find outliers and clip data for each input array
    clipped_data = []
    outlier_indices = []
    for arr in data_arrays:
        # We only care about upper-bound outliers for visualization purposes
        outliers = arr > upper_bound
        outlier_indices.append(outliers)
        clipped_data.append(np.clip(arr, a_min=None, a_max=upper_bound))

    return ylim, clipped_data, outlier_indices


def _format_value(
    value: float,
    *,
    plt_style: str | list[str],
    is_percent_like: bool = False,
) -> str:
    """Format a float value, escaping '%' for LaTeX if needed."""
    # Check if LaTeX is likely active by checking the style context
    is_latex = any("science" in s for s in plt_style) if isinstance(plt_style, list) else False

    if is_percent_like:
        if is_latex:
            # Manual percentage formatting with escaped percent sign
            return f"{value * 100:.2f}\\%"
        # Standard percentage formatting
        return f"{value:.2%}"
    return f"{value:.4f}"


def _annotate_texts(
    ax: Axes,
    texts_to_adjust: list[Text],
    lines_to_avoid: list[Line2D] | None = None,
    text_adjuster: Literal["adjusttext", "textalloc"] | None = "textalloc",
    text_fontsize: str | float = 8,
) -> None:
    """Handle text annotation using the specified adjuster.

    Args:
        ax (Axes): The matplotlib axes to draw on.
        texts_to_adjust (list[plt.Text]): A list of matplotlib Text objects to be placed.
        lines_to_avoid (list[plt.Line2D] | None, optional): A list of Line2D objects for
            collision avoidance. Defaults to None.
        text_adjuster (Literal["adjusttext", "textalloc"] | None, optional): The text adjustment
            strategy to use. Defaults to "textalloc".
        text_fontsize (float, optional): The font size for the text annotations. Defaults to 8.
    """
    if not lines_to_avoid:
        lines_to_avoid = []

    if text_adjuster == "adjusttext" and texts_to_adjust:
        adjust_text(texts_to_adjust, add_objects=lines_to_avoid, ax=ax)
    elif text_adjuster == "textalloc":
        # We need to extract existing lines for collision avoidance for all calls
        x_lines = [line.get_xdata(orig=False) for line in lines_to_avoid]
        y_lines = [line.get_ydata(orig=False) for line in lines_to_avoid]

        # Group texts by their x-coordinate
        texts_by_x = defaultdict(list)
        for t in texts_to_adjust:
            texts_by_x[t.get_position()[0]].append(t)

        # Remove the original, unadjusted text objects from the plot first
        for t in texts_to_adjust:
            t.remove()

        # Process each group of texts
        for text_group in texts_by_x.values():
            if len(text_group) > 1:
                # Stacked points: process them one by one with different directions
                # Sort by y-value to have a consistent order (e.g., top one is always east)
                text_group.sort(key=lambda t: t.get_position()[1], reverse=True)

                # Assign alternating directions
                directions = ["east", "west", "northeast", "southwest"]
                for i, t in enumerate(text_group):
                    ta.allocate(
                        ax=ax,
                        x=[t.get_position()[0]],
                        y=[t.get_position()[1]],
                        text_list=[t.get_text()],
                        x_lines=x_lines,  # type: ignore[reportArgumentType, arg-type]
                        y_lines=y_lines,  # type: ignore[reportArgumentType, arg-type]
                        textsize=text_fontsize,  # type: ignore[reportArgumentType, arg-type]
                        textcolor=mcolors.to_hex(t.get_color()),
                        direction=directions[i % len(directions)],
                        # margin=0.01,
                        # min_distance=0.015,
                        draw_lines=True,
                        linewidth=0.5,
                        linecolor="gray",
                        verbose=False,
                    )
            else:
                # Single point: let textalloc choose the best direction
                t = text_group[0]
                ta.allocate(
                    ax=ax,
                    x=[t.get_position()[0]],
                    y=[t.get_position()[1]],
                    text_list=[t.get_text()],
                    x_lines=x_lines,  # type: ignore[reportArgumentType, arg-type]
                    y_lines=y_lines,  # type: ignore[reportArgumentType, arg-type]
                    textsize=text_fontsize,  # type: ignore[reportArgumentType, arg-type]
                    textcolor=mcolors.to_hex(t.get_color()),
                    # Let textalloc decide
                    direction=None,  # type: ignore[reportArgumentType, arg-type]
                    # margin=0.01,
                    # min_distance=0.015,
                    draw_lines=True,
                    linewidth=0.5,
                    linecolor="gray",
                    verbose=False,
                )


def _setup_plot_layout(
    history: TrainingHistoryClassification,
    loss_types: list[str],
    metrics: list[str],
    *,
    figsize: tuple[float, float] = (12, 4),
    sharex: bool = True,
) -> tuple[Figure, dict[str, Any]]:
    """Determine the plot layout, creating a special grid if LR is present.

    Args:
        history (TrainingHistoryClassification): The training history dictionary.
        loss_types (list[str]): A list of loss names to be plotted.
        metrics (list[str]): A list of metric names to be plotted.
        figsize (tuple[float, float], optional): The base size (width, height) for a single
            subplot slot. The total figure height is scaled by the number of slots.
            Defaults to (12, 4).
        sharex (bool, optional): Whether the subplots should share the x-axis.
            Defaults to True.

    Returns:
        tuple[Figure, dict[str, Any], GridSpec | None]: A tuple containing:
            - The matplotlib Figure object.
            - A dictionary of axes ('loss', 'lr', 'metrics').
    """
    history_dict = history.to_dict(ignore_empty=False)  # Important: keep empty lists
    has_lr = "lr" in history_dict and history_dict["lr"]

    # If LR is present, the first "plot slot" is a 2x1 grid for Loss and LR.
    # Otherwise, it's just a single plot for Loss.
    n_loss_plots = len(loss_types)
    n_metric_plots = len(metrics)
    n_lr_plots = 1 if has_lr else 0
    total_plot_slots = n_loss_plots + n_metric_plots

    # Calculate figure height based on the number of main plots + a smaller one for LR
    total_fig_height = figsize[1] * (total_plot_slots + (0.6 * n_lr_plots))
    fig = plt.figure(figsize=(figsize[0], total_fig_height), constrained_layout=True)

    # Create a GridSpec that accommodates all plots
    num_gs_rows = total_plot_slots + n_lr_plots
    height_ratios = [1] * total_plot_slots + ([0.6] * n_lr_plots if has_lr else [])
    gs = GridSpec(num_gs_rows, 1, figure=fig, height_ratios=height_ratios)

    # Create axes for each plot type
    axes_dict: dict[str, Any] = {"losses": [], "metrics": [], "lr": None}

    # Loss axes
    ref_ax = None  # The first axis created will be the reference for sharex
    for i in range(n_loss_plots):
        ax = fig.add_subplot(gs[i, 0], sharex=ref_ax if sharex else None)
        if ref_ax is None:
            ref_ax = ax
        axes_dict["losses"].append(ax)

    # Metric axes
    for i in range(n_metric_plots):
        ax = fig.add_subplot(gs[i + n_loss_plots, 0], sharex=ref_ax if sharex else None)
        axes_dict["metrics"].append(ax)

    # LR axis
    if has_lr:
        ax = fig.add_subplot(gs[-1, 0], sharex=ref_ax if sharex else None)
        axes_dict["lr"] = ax

    return fig, axes_dict


def _format_xaxis_epoch_ticks(
    ax: Axes,
    max_epoch: int,
    proximity_threshold_ratio: float = 0.05,
) -> None:
    """Ensure x-axis ticks are integers, include the last epoch, and avoid crowding.

    Args:
        ax (Axes): The matplotlib axis object to format.
        max_epoch (int): The total number of epochs (1-based).
        proximity_threshold_ratio (float): A ratio of the total epoch span. Ticks within this
            distance of `max_epoch` (and not `max_epoch` itself) will be removed to prevent
            label overlap. Defaults to 0.05 (i.e., 5% of the total range).
    """
    # Get the current ticks that matplotlib thinks are reasonable
    ticks = ax.get_xticks()

    # Filter for integer ticks within the data range [1, max_epoch]
    integer_ticks = {round(tick) for tick in ticks if 1 <= tick <= max_epoch}

    # Ensure the first and last epoch numbers are always included
    integer_ticks.add(1)
    integer_ticks.add(max_epoch)

    # --- Heuristic to prevent crowding near the final tick ---
    # Calculate the minimum distance to avoid crowding
    epoch_span = max_epoch - 1
    min_distance = epoch_span * proximity_threshold_ratio

    # Create a new set for ticks to keep, to avoid modifying the set while iterating
    final_ticks = set(integer_ticks)

    # Check for ticks (other than max_epoch itself) that are too close to max_epoch
    for tick in integer_ticks:
        if tick != max_epoch and (max_epoch - tick) < min_distance:
            # If a tick is too close, remove it from our final set
            final_ticks.discard(tick)
            logger.debug(
                f"Removed crowded tick {tick} because it was too close to the "
                f"final epoch {max_epoch} (threshold: {min_distance:.1f} epochs)."
            )

    # Set the final, cleaned-up ticks
    ax.set_xticks(sorted(final_ticks))

    # Ensure labels are formatted as clean integers
    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x)}")  # noqa: ARG005
    )


def _plot_lr_curve(
    ax_lr: Axes,
    history: TrainingHistoryClassification,
    epochs: list[int] | np.ndarray | None = [],
    **kwargs,
) -> None:
    """Plot the learning rate curve on its dedicated subplot."""
    history_dict = history.to_dict(ignore_empty=False)  # Important: keep empty lists
    if "lr" not in history_dict or not history_dict["lr"]:
        logger.warning("No learning rate data found in history; skipping LR plot.")
        ax_lr.text(
            0.5,
            0.5,
            "LR Data Not Found",
            ha="center",
            va="center",
            transform=ax_lr.transAxes,
        )
        return

    # Create an explicit x-axis for epoch numbers
    if not epochs:
        history_len = len(history_dict["lr"])
        epochs = np.arange(1, history_len + 1)

    ax_lr.plot(
        epochs,
        history_dict["lr"],
        label="Learning Rate",
        color="tab:purple",
        marker=".",
        linestyle="-",
    )
    ax_lr.set_ylabel("Learning Rate", fontsize=kwargs.get("axis_fontsize", "medium"))
    ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_lr.legend(fontsize=kwargs.get("legend_fontsize", "small"))
    ax_lr.grid(visible=True, linestyle="--", alpha=0.6)


def __format_log_scale(ax: Axes) -> Axes:
    """Format the y-axis of a plot to use logarithmic scale with custom ticks."""
    # 1. Set the major ticks to be at every power of 10 (e.g., 1e-1, 1e0, 1e1)
    #    This uses a standard logarithmic formatter.
    major_locator = LogLocator(base=10.0, numticks=15)
    ax.yaxis.set_major_locator(major_locator)

    # 2. Set the minor ticks to appear at intermediate values (e.g., 2e-1, 3e-1, ...)
    #    'subs' specifies the multiples of the lower power of 10.
    #    (0.2, 0.4, 0.6, 0.8) will create ticks at 2x, 4x, 6x, 8x for each decade.
    minor_locator = LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1).tolist(), numticks=15)
    ax.yaxis.set_minor_locator(minor_locator)

    # 3. Use a specific formatter for the minor ticks to ensure they are readable.
    #    Without this, they might not show up or might be formatted poorly.
    #    LogFormatterMathtext will format them as, e.g., "2x10^{-1}".
    ax.yaxis.set_minor_formatter(LogFormatterMathtext(base=10.0))

    # 4. Optional: Customize the appearance of the tick labels.
    #    This makes the minor labels smaller so they don't visually compete
    #    with the major labels.
    ax.tick_params(axis="y", which="minor", labelsize="small")
    return ax


def __safe_save_fig(
    fig: Figure,
    save_plot_path: str | Path | None,
    *,
    dpi: int = 300,
    name: str = "",
    verbose: bool = False,
) -> None:
    """Save a matplotlib figure to a file, creating directories if needed.

    Args:
        fig (plt.Figure): The figure to save.
        save_plot_path (str | Path | None): The path to save the figure.
        dpi (int): The DPI to use for the saved figure.
        name (str): A name for the plot, used in logging.
        verbose (bool): If True, log the save action.
    """
    if not save_plot_path:
        return

    # --- Convert to Path object for robust handling ---
    save_path = Path(save_plot_path)

    # --- Create parent directories ---
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Validate file extension ---
    supported_formats = ["png", "jpg", "jpeg", "pdf", "svg"]

    # .suffix includes the dot, so we slice from the 1st character
    file_ext = save_path.suffix[1:].lower()

    if file_ext not in supported_formats:
        logger.warning(f"Unsupported save format '.{file_ext}'. Defaulting to PNG.")

        # Change the suffix of the path object
        save_path = save_path.with_suffix(".png")
        file_ext = "png"  # Update the format for savefig

    # --- Save the figure ---
    try:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=file_ext)
        if verbose:
            name = name if name else "Plot"
            logger.info(f"{name} saved to {save_path} with DPI={dpi}.")
    except Exception as e:
        logger.error(f"Failed to save figure to {save_path}: {e}")


def __build_legend(
    metadata: dict,
    key_map: dict[str, list[str] | tuple[list[str], Callable | None]],
    template: str,
) -> str:
    """Build a legend label from metadata using a template string and a key map.

    Example:
    >>> key_map = {
            "name": ["model", "model_kwargs", "name"],
            "model_dp": ["model", "model_kwargs", "additional_kwargs", "drop_prob"],
            "mlp_dims": ["model", "model_kwargs", "mlp_hidden_dims"],
            "mlp_dp": ["model", "model_kwargs", "mlp_dropout_prob"],
        }
    >>> template = "{name} (dp={model_dp:.0f}%), MLP (dp={mlp_dp:.0f}%) : {mlp_dims}"
    >>> __build_legend(metadata, key_map, template)
    'ShallowFBCSPNet (dp=50%), MLP (dp=70%): [2048, 4096, 2048]'

    >>> key_map = {
            "name": ["model_kwargs", "name"],
            "dtype": (["dataset_params", "dim_order"], str.upper),
        }
    >>> template = "{name} ({dtype})"
    >>> __build_legend(metadata, key_map, template)
    'ShallowFBCSPNet (CHANNELS_FIRST)'

    Args:
        metadata (dict): The metadata dictionary.
        key_map (dict[str, list[str]]): A dictionary mapping placeholder names in the template to
            their full key path in the metadata dict.
        template (str): An f-string-like template for the final legend label.

    Returns:
        str: The formatted legend label string.
    """
    format_values = {}

    # 1. Extract and transform all required values from metadata
    for placeholder, path_info in key_map.items():
        transform_func = None
        if isinstance(path_info, tuple):
            key_path, transform_func = path_info
        else:
            key_path = path_info  # Assumes it's a list

        value = get_nested_value(metadata, key_path)

        # Pre-process certain values for cleaner formatting
        key_name = key_path[-1]
        if "model_class" in key_name and isinstance(value, str):
            value = value.strip().replace("<class '", "").replace("'>", "").split(".")[-1]

        # Automatically handle percentage conversion for dropout probabilities
        if ("dropout_prob" in key_name or "drop_prob" in key_name) and isinstance(
            value, (int, float)
        ):
            value *= 100  # Convert 0.7 to 70.0 for formatting with {:.0f}

        # --- Apply the user-defined transformation ---
        if transform_func and value is not None:
            try:
                # Always apply transform if provided and value is not None
                value = transform_func(value)
            except Exception as e:
                logger.warning(
                    f"Could not apply transform '{transform_func.__name__}' to value '{value}'. "
                    f"Error: {e}"
                )
        elif transform_func and value is None:
            # Also apply transform to None values
            # This allows a transform function to convert None to a meaningful string.
            try:
                value = transform_func(value)
            except Exception as e:
                logger.warning(
                    f"Could not apply transform '{transform_func.__name__}' to None. Error: {e}"
                )

        format_values[placeholder] = value

    # --- 2. Format the template string ---
    try:
        # We use a custom formatter to handle missing keys gracefully
        class SafeFormatter(string.Formatter):
            def format_field(self, value, format_spec) -> str:  # noqa: ANN001
                # If the value is None, we cannot apply a format spec like ':.0f'.
                # In this case, we return a default string (e.g., 'N/A') and ignore the spec.
                if value is None:
                    return "N/A"
                return super().format_field(value, format_spec)

            def get_value(
                self,
                key: int | str,
                args: Any,  # noqa: ANN401
                kwargs: Any,  # noqa: ANN401
            ) -> Any:  # noqa: ANN401
                if isinstance(key, str):
                    return kwargs.get(key, f"{{{key}}}")  # Return placeholder if key not found

                return super().get_value(key, args, kwargs)

            def convert_field(self, value: str | float | None, conversion: str | None) -> str:
                if conversion == "upper":
                    return str(value).upper()
                if conversion == "capitalize":
                    return str(value).capitalize()
                # Fallback to default behavior for other flags (like 'r' or 's')
                return super().convert_field(value, conversion)

        formatter = SafeFormatter()

        # --- Template Cleaning ---
        # This handles missing keys by replacing the full placeholder and common surrounding
        # patterns with an empty string.
        final_template = template
        for placeholder, value in format_values.items():
            if value is None:
                # Regex to find the placeholder and optional surrounding patterns
                # like ", key={...}", " (key={...})", "key: {...}"
                # This makes the cleaning much more robust.
                patterns_to_remove = [
                    r",\s*" + placeholder + r"\s*=\s*\{" + placeholder + r"[^}]*\}",
                    r"\s*\([^)]*" + placeholder + r"[^)]*\)",
                    r"[,\s]*\w+:\s*\{" + placeholder + r"[^}]*\}",
                ]
                for pattern in patterns_to_remove:
                    final_template = re.sub(pattern, "", final_template)

                # A final simple replacement for any remaining cases
                final_template = final_template.replace(f"{{{placeholder}}}", "N/A")

        # Clean up any resulting double spaces, commas, or empty parentheses
        final_template = re.sub(r"\s{2,}", " ", final_template).strip()
        final_template = re.sub(r",\s*,", ",", final_template).strip(" ,")
        final_template = re.sub(r"\(\s*\)", "", final_template).strip()  # Removes empty ()

        return formatter.format(final_template, **format_values)

    except (ValueError, KeyError) as e:
        logger.error(f"Failed to format legend template. Error: {e}")
        # Fallback to a simple join of found values
        return ", ".join(f"{k}={v}" for k, v in format_values.items() if v is not None)


def generate_plot_paths(
    base_path: str | Path | None,
    suffixes: dict[str, str],
    middle_suffix: str = "",
) -> dict[str, Path | None]:
    """Generate a dictionary of new file paths based on a base path and given suffixes.

    If the base_path is None, all generated paths will also be None.

    Args:
        base_path (str | Path | None): The base file path (e.g., "plots/my_run/main_loss.png").
        suffixes (dict[str, str]): A dictionary where keys are identifiers for the new plots and
            values are the suffixes to append to the base filename's stem.
            Example: {'line': '_per_expr_line', 'bar': '_per_expr_bar'}
        middle_suffix (str): An optional suffix to insert into the middle of the filename,
            e.g., "_optim_loss".

    Returns:
        (dict[str, Path | None]): A dictionary with the same keys as `suffixes`, where each value
            is the newly generated Path object, or None if base_path was None.
            Example: `{'line': Path('plots/my_run/main_loss_per_expr_line.png'), ...}`
    """
    if base_path is None:
        # If no base path is provided, return a dictionary of Nones
        return dict.fromkeys(suffixes, None)

    base_path_obj = Path(base_path)
    parent = base_path_obj.parent
    stem = base_path_obj.stem
    ext = base_path_obj.suffix  # Includes the dot, e.g., ".png"

    plot_paths: dict[str, Path | None] = {}
    for key, suffix in suffixes.items():
        # Create new filename: e.g., "losses" + "_optim_loss" + "_per_expr_bar" + ".png"
        new_filename = f"{stem}{middle_suffix}{suffix}{ext}"
        plot_paths[key] = parent / new_filename

    return plot_paths


def plot_loss_comparison(
    metadata_files: list[Path],
    output_path: Path,
    *,
    loss_to_plot: str = "loss",
    metrics_to_plot: list[str] | None = None,
    layout: Literal["vertical", "horizontal"] = "vertical",
    smoothing_factor: float | None = None,
    best_epoch_metric: str = "loss",
    plot_best_epoch_vlines: bool = True,
    top_k: int | None = None,
    from_epoch: int | None = 0,
    to_epoch: int | None = None,
    filters: dict[tuple[str, ...], set] | None = None,
    label_keys: dict[str, list[str] | tuple[list[str], Callable | None]] | None = None,
    label_template: str | dict[str, str] | None = None,
    label_template_selector: tuple[list[str], str] | None = None,
    allowed_eeg_windows: set[float] | None = None,
    allowed_flame_frames: set[int] | None = None,
    allowed_model_types: set[str] | None = None,
    simplify_model_names: dict[str, str] | None = None,
    title: str = "",
    title_suffix: str = "",
    legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.15),
    legend_fontsize: int | str = "small",
    log_scale: bool = False,
    plt_style: str | list[str] = "default",
    figsize: tuple = (12, 10),
    dpi: int = 300,
    text_adjuster: Literal["adjusttext", "textalloc"] | None = "textalloc",
    **kwargs,
) -> None:
    """Plot a comparison of training and testing losses from multiple model metadata files.

    Args:
        metadata_files (list[Path]): A list of paths to metadata files containing model training
            history and configuration.
        output_path (Path): The path where the plot will be saved.
        loss_to_plot (str): The base name of the loss to plot (e.g., "loss", "param_loss",
            "vertex_loss"). The function will look for `train_{loss_to_plot}` and
            `val_{loss_to_plot}` in the history. Defaults to "loss".
        metrics_to_plot (list[str] | None): A list of metric names to plot in addition to the loss.
            Each metric will get its own subplot. The names must match the keys in the 'history'
            dict (without train_/val_ prefix). Example: ['Balanced Accuracy', 'F1 Score'].
        layout (Literal["vertical", "horizontal"]): Arrangement of subplots.
            - "vertical": Train/Val plots for each metric are paired in a row.
                (e.g., Row 1: Train Loss, Val Loss; Row 2: Train Acc, Val Acc). Default.
            - "horizontal": All training plots are in the left column, all validation plots are in
                the right column.
        smoothing_factor (float | None): A value between 0.0 and 1.0 to control EMA smoothing. 0.0
            means no smoothing (raw data). Values closer to 1.0 (e.g., 0.9) result in heavier
            smoothing. If provided, a faint raw line and a solid smoothed line are plotted.
        best_epoch_metric (str): The validation metric used to determine the "best" epoch. Can be
            'loss' or any metric name from `metrics_to_plot`. Defaults to 'loss'.
        plot_best_epoch_vlines (bool): If True, draws a vertical line and annotates the values at
            the best epoch for each run.
        top_k (int | None): If specified, only the top K models (based on best validation
            loss) will be plotted. If None, all valid models are plotted.
        from_epoch (int | None): The epoch from which to start plotting the loss data. If None,
            uses the first epoch.
        to_epoch (int | None): The epoch to stop plotting the loss data. If None, plots all epochs.
        filters (dict[tuple[str, ...], set] | None): A dictionary to filter which metadata files
            are plotted. Keys are tuples representing the key path
            (e.g., ('kwargs', 'norm_layer')). Values are sets of allowed values
            (e.g., {'batch', 'layer'}). If None, no filtering is applied. For example,
            `filters={("kwargs", "norm_layer"): {"layer"}}` would only include metadata files
            where the 'norm_layer' key in 'kwargs' is set to 'layer'.
        label_keys (dict[str, list[str] | tuple[list[str], Callable | None]] | None): Maps
            placeholder names to their config path. The path can be a simple list of keys, or a
            tuple of (key_path, transform_function).
        label_template (str | dict[str, str] | None): A template string for the legend labels.
            If None, a default label will be generated based on the model class and dataset.
            The template can use placeholders for metadata keys, e.g.,
            `"{name} (dp={model_dp:.0f}%) MLP (dp: {mlp_dp:.0f}%): {mlp_dims}, bs: {bs}"`.
            Can be a single f-string-like template, or a dictionary of templates for conditional
            formatting.
        label_template_selector (tuple[list[str], str] | None): Required if `label_template` is a
            dict. A tuple containing `(key_path, default_value)`. The value found at `key_path` in
            the metadata will be used to select the template from the `label_template` dict.
        allowed_eeg_windows (set[float] | None): A set of allowed EEG window lengths in seconds.
            If None, all EEG window lengths are allowed.
        allowed_flame_frames (set[int] | None): A set of allowed FLAME window lengths in frames.
            If None, all FLAME window lengths are allowed.
        allowed_model_types (set[str] | None): A set of model type names (e.g., {'MLP', 'LSTM'})
            to include. If None, all model types are allowed.
        simplify_model_names (dict[str, str] | None): A dictionary to simplify model names in the
            plot legends. Keys are the original model names, and values are the simplified names.
        title (str): The title of the plot.
        title_suffix (str): A suffix to append to the title for additional context.
        legend_bbox_to_anchor (tuple[float, float]): The position of the legend in the plot.
            For example, `(0.5, -0.15)` positions it horizontally centered (0.5) and slightly
            below the figure area (-0.15).
        legend_fontsize (int | str): The font size for the legend. Can be an integer or a string
            like 'small', 'x-small', etc.
        log_scale (bool): If True, the y-axis will be set to logarithmic scale.
        plt_style (str | list[str]): The style(s) to use for the plot. It will use
            `with plt.style.context(plt_style):`.
        figsize (tuple): The size of the figure to create.
        dpi (int): The resolution of the figure in dots per inch.
        text_adjuster (Literal["adjusttext", "textalloc"] | None): The library to use for
            preventing text annotation overlap. Can be 'adjusttext', 'textalloc', or
            None for no adjustment. Defaults to "textalloc".
        **kwargs: Additional keyword arguments to pass to `matplotlib.pyplot.plot()`.
    """

    def plot_line(
        ax_train: Axes,
        ax_val: Axes,
        epochs: np.ndarray | list | range,
        train_data: np.ndarray | list,
        val_data: np.ndarray | list,
        train_color: str,
        val_color: str,
        label: str | None = None,
        *,
        is_pct: bool = False,
        best_idx: int | None = None,
        va_train: str = "top",
        va_val: str = "top",
    ) -> None:
        """Plot a line with optional smoothing and faint raw data in the background."""
        if not train_data or not val_data:
            return

        raw_train_data = np.array(train_data)
        raw_val_data = np.array(val_data)

        if ax_train not in lines_per_axis:
            lines_per_axis[ax_train] = []
        if ax_val not in lines_per_axis:
            lines_per_axis[ax_val] = []
        if ax_train not in texts_per_axis:
            texts_per_axis[ax_train] = []
        if ax_val not in texts_per_axis:
            texts_per_axis[ax_val] = []

        if smoothing_factor is not None and smoothing_factor > 0:
            # Plot raw data faintly in the background
            (raw_train_line,) = ax_train.plot(
                epochs, train_data, color=train_color, alpha=0.2, linewidth=0.8
            )
            (raw_val_line,) = ax_val.plot(
                epochs, val_data, color=val_color, alpha=0.2, linewidth=0.8
            )

            lines_per_axis[ax_train].append(raw_train_line)
            lines_per_axis[ax_val].append(raw_val_line)

            # Calculate EMA. Alpha is the inverse of the user-friendly factor.
            alpha = 1.0 - smoothing_factor
            train_data = exponential_moving_average(
                raw_train_data,
                alpha,
                preserve_min_max_points=True,
                preserve_mode="first",
            )
            val_data = exponential_moving_average(
                raw_val_data,
                alpha,
                preserve_min_max_points=True,
                preserve_mode="first",
            )

        (train_line,) = ax_train.plot(epochs, train_data, label=label, color=train_color, **kwargs)
        (val_line,) = ax_val.plot(epochs, val_data, label=label, color=val_color, **kwargs)

        lines_per_axis[ax_train].extend([train_line])
        lines_per_axis[ax_val].extend([val_line])

        from_epoch_i = from_epoch if from_epoch is not None else 0

        if plot_best_epoch_vlines and best_idx is not None:
            # Draw vlines
            vline_train = ax_train.axvline(
                x=best_idx + 1,  # +1 because it's 1-based
                color=train_color,
                linestyle=":",
                alpha=0.7,
            )
            vline_val = ax_val.axvline(
                x=best_idx + 1,  # +1 because it's 1-based
                color=val_color,
                linestyle=":",
                alpha=0.7,
            )
            lines_per_axis[ax_train].append(vline_train)
            lines_per_axis[ax_val].append(vline_val)

            # Add text with a slight initial offset
            texts_per_axis[ax_train].append(
                ax_train.text(
                    best_idx,
                    float(raw_train_data[best_idx - from_epoch_i]),
                    _format_value(
                        float(raw_train_data[best_idx - from_epoch_i]),
                        plt_style=plt_style,
                        is_percent_like=is_pct,
                    ),
                    color=train_color,
                    fontsize=5,
                    ha="center",
                    va=va_train,
                )
            )
            texts_per_axis[ax_val].append(
                ax_val.text(
                    best_idx,
                    float(raw_val_data[best_idx - from_epoch_i]),
                    _format_value(
                        float(raw_val_data[best_idx - from_epoch_i]),
                        plt_style=plt_style,
                        is_percent_like=is_pct,
                    ),
                    color=val_color,
                    fontsize=5,
                    ha="center",
                    va=va_val,
                )
            )

    if not metadata_files:
        logger.warning("No metadata files provided to plot. Exiting.")
        return

    if metrics_to_plot is None:
        metrics_to_plot = []

    # Define the loss keys to be used throughout the function
    train_loss_key = f"train_{loss_to_plot}"
    val_loss_key = f"val_{loss_to_plot}"

    # The metric for finding the best epoch still needs the val_ prefix.
    best_epoch_metric_key = (
        f"val_{best_epoch_metric}" if "val_" not in best_epoch_metric else best_epoch_metric
    )

    # Determine if lower values are better for the best epoch metric
    lower_is_better = "loss" in best_epoch_metric.lower()

    runs_data = []
    for metadata_path in metadata_files:
        try:
            metadata = load_json(metadata_path)

            # --- Generic Filtering Logic ---
            if filters is not None and metadata_is_filtered_out(
                metadata, filters, metadata_name=metadata_path.parent.name
            ):
                logger.debug(f"Skipping {metadata_path.parent.name}: filters did not match.")
                continue

            # --- Parsing Logic (with model class extraction) ---
            dataset_meta = metadata.get("dataset", metadata.get("dataset_params", {}))
            eeg_win_sec = dataset_meta.get("eeg_window_seconds")
            flame_win_sec = dataset_meta.get("flame_window_seconds")
            video_fps = dataset_meta.get("video_fps", 29.97)

            # Extract model class name
            model_class_name = extract_model_name(metadata)

            # Simplify name for filtering and labeling
            if simplify_model_names is not None:
                model_class_name = simplify_model_names.get(model_class_name, model_class_name)

            # --- Filtering Logic (now includes model type) ---
            if any(v is None for v in [eeg_win_sec, flame_win_sec, video_fps]):
                logger.debug(
                    f"Skipping {metadata_path.parent.name}: Missing required metadata fields, "
                    f"eeg_window_seconds={eeg_win_sec}, flame_window_seconds={flame_win_sec}, "
                    f"video_fps={video_fps}"
                )
                continue

            flame_frames = round(flame_win_sec * video_fps)

            # Filter by model type
            if allowed_model_types and model_class_name not in allowed_model_types:
                logger.debug(
                    f"Skipping {metadata_path.parent.name}: model type '{model_class_name}' "
                    "not in allowed set."
                )
                continue

            if allowed_eeg_windows and eeg_win_sec not in allowed_eeg_windows:
                logger.debug(
                    f"Skipping {metadata_path.parent.name}: "
                    f"eeg_window_seconds={eeg_win_sec} not in allowed set."
                )
                continue

            if allowed_flame_frames and flame_frames not in allowed_flame_frames:
                logger.debug(
                    f"Skipping {metadata_path.parent.name}: "
                    f"flame_frames={flame_frames} not in allowed set."
                )
                continue

            history = metadata.get("history", {})

            # Try "history" first, then "loss" if not found
            train_loss = metadata.get("history", metadata.get("loss", {})).get(train_loss_key, [])
            val_loss = metadata.get("history", metadata.get("loss", {})).get(val_loss_key, [])

            if not train_loss or not val_loss:
                logger.warning(f"Skipping {metadata_path.parent.name}: Missing loss data.")
                continue

            # --- Determine Best Epoch based on `best_epoch_metric` ---
            # Get the history for the target metric
            best_metric_history = history.get(best_epoch_metric_key, [])

            if not best_metric_history:
                logger.warning(
                    f"Skipping {metadata_path.parent.name}: "
                    f"Best epoch metric '{best_epoch_metric_key}' not found. "
                    f"Available metrics: {list(history.keys())}"
                )
                continue

            # Find the best epoch index and value
            best_epoch_idx = (
                np.argmin(best_metric_history)
                if lower_is_better
                else np.argmax(best_metric_history)
            )
            best_metric_value = best_metric_history[best_epoch_idx]
            best_train_loss_at_that_epoch = train_loss[best_epoch_idx]
            best_val_loss_at_that_epoch = val_loss[best_epoch_idx]

            # --- Store data for this run ---
            runs_data.append(
                {
                    "path": metadata_path,
                    "metadata": metadata,
                    "history": history,
                    # NOTE: best_metric_value is from the ranking metric, which might be different
                    # from the plotted loss
                    "best_metric_value": best_metric_value,
                    "best_train_loss": best_train_loss_at_that_epoch,
                    "best_val_loss": best_val_loss_at_that_epoch,
                    "best_epoch": best_epoch_idx + 1,
                    "model_class_name": model_class_name,
                    "eeg_win_sec": eeg_win_sec,
                    "flame_win_sec": flame_win_sec,
                    "flame_frames": flame_frames,
                }
            )

        except Exception as e:
            logger.error(f"Failed to process or plot {metadata_path}: {e}")

    if not runs_data:
        logger.error("No valid models found to plot after filtering. No plot will be generated.")
        return

    # Sort runs by best validation loss (ascending)
    runs_data.sort(key=lambda x: x["best_metric_value"], reverse=not lower_is_better)

    # Determine which runs to plot
    runs_to_plot = runs_data[:top_k] if top_k is not None else runs_data

    # --- Plot Setup: Dynamic and Layout-Aware ---
    with plt.style.context(plt_style):
        n_metrics = len(metrics_to_plot)
        n_rows = 1 + n_metrics

        if layout == "horizontal":
            # Horizontal layout: Train plots on top of Val plots
            n_cols = 2
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(figsize[0], figsize[1] * n_rows),
                dpi=dpi,
                sharex=True,
            )
            # Ensure axes is always 2D array
            if n_rows == 1:
                axes = np.expand_dims(axes, axis=0)
        elif layout == "vertical":
            # Vertical layout: One metric per row, Train/Val side-by-side
            n_cols = 1
            # We need 2 rows for Loss (Train/Val) + 2 rows for each metric
            fig, axes = plt.subplots(
                2 * n_rows,
                n_cols,
                figsize=(figsize[0], figsize[1] * n_rows),
                dpi=dpi,
                sharex=True,
            )
            # Ensure axes is a 2D array even with 1 column
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
        else:
            raise ValueError("`layout` must be 'vertical' or 'horizontal'.")

        from_epoch = 0 if from_epoch is None else from_epoch
        colors = get_n_colors(len(runs_to_plot), style=plt_style)

        # --- Prepare lists for adjustText for each subplot ---
        # We create a dictionary where keys are the axis objects
        texts_per_axis: dict[Axes, list] = {}
        lines_per_axis: dict[Axes, list] = {}

        # A counter for each axis to alternate the nudge
        nudge_counter_per_axis = defaultdict(int)

        # --- Plotting Loop ---
        for i, run in enumerate(runs_to_plot):
            # --- Build Label ---
            base_label = ""
            if label_keys and label_template:
                # --- Conditional Template Selection ---
                final_template_str = ""
                if isinstance(label_template, dict):
                    if not label_template_selector:
                        raise ValueError(
                            "If `label_template` is a dictionary, `label_template_selector` "
                            "must be provided."
                        )

                    selector_key_path, default_value = label_template_selector
                    # Get the value from metadata that will decide which template to use
                    selector_value = get_nested_value(
                        run["metadata"], selector_key_path, default=default_value
                    )

                    # Choose the template. Fallback to the default if the specific one isn't found.
                    final_template_str = label_template.get(
                        selector_value,
                        label_template.get(default_value, ""),
                    )
                    if not final_template_str:
                        logger.error(
                            f"No valid template found for selector value '{selector_value}' or "
                            f"default '{default_value}'."
                        )
                        final_template_str = "Template_Error"

                elif isinstance(label_template, str):
                    final_template_str = label_template

                base_label = __build_legend(
                    metadata=run["metadata"],
                    key_map=label_keys,
                    template=final_template_str,
                )
            else:
                base_label = (
                    f"{run['model_class_name']}: "
                    f"EEG={run['eeg_win_sec']}s, "
                    f"FLAME={run['flame_win_sec']}s ({run['flame_frames']}f)"
                )

            # Enhance label with rank and performance
            best_epoch_metric_title = (
                best_epoch_metric.replace("_", " ").title().replace("Val", "Val.")
            )
            best_epoch_metric_value = _format_value(
                float(run["best_metric_value"]),
                plt_style=plt_style,
                is_percent_like="loss" not in best_epoch_metric.lower(),
            )
            final_label = (
                f"{base_label} (Best {best_epoch_metric_title}: {best_epoch_metric_value} "
                f"@ep. {run['best_epoch']})"
            )

            logger.info(f"Plotted: '{final_label}' for {run['path']}")

            to_epoch_run = (
                to_epoch if to_epoch is not None else len(run["history"].get("train_loss", []))
            )
            epochs = range(from_epoch + 1, to_epoch_run + 1)

            # Plot Loss
            train_loss = run["history"].get(train_loss_key, [])[from_epoch:to_epoch_run]
            val_loss = run["history"].get(val_loss_key, [])[from_epoch:to_epoch_run]

            plot_line(
                axes[0, 0],
                axes[0, 1] if layout == "horizontal" else axes[1, 0],
                epochs,
                train_loss,
                val_loss,
                label=final_label,
                train_color=colors[i],
                val_color=colors[i],
                is_pct=False,
                best_idx=run["best_epoch"] - 1,  # "best_epoch": best_epoch_idx + 1,
            )

            # Plot Metrics
            for metric_idx, metric_name in enumerate(metrics_to_plot):
                train_metric = run["history"].get(f"train_{metric_name}", [])[
                    from_epoch:to_epoch_run
                ]
                val_metric = run["history"].get(f"val_{metric_name}", [])[from_epoch:to_epoch_run]

                row_idx = metric_idx + 1
                row_idx_train = 2 * (metric_idx + 1)
                row_idx_val = row_idx_train + 1

                # --- Alternating Nudge Logic ---
                if layout == "horizontal":
                    if axes[row_idx, 0] not in nudge_counter_per_axis:
                        nudge_counter_per_axis[axes[row_idx, 0]] = 0
                    if axes[row_idx, 1] not in nudge_counter_per_axis:
                        nudge_counter_per_axis[axes[row_idx, 1]] = 0
                    va_train = (
                        "bottom" if nudge_counter_per_axis[axes[row_idx, 0]] % 2 == 0 else "top"
                    )
                    va_val = (
                        "top" if nudge_counter_per_axis[axes[row_idx, 1]] % 2 == 0 else "bottom"
                    )
                    # Increment for the next line on this axis
                    nudge_counter_per_axis[axes[row_idx, 0]] += 1
                    nudge_counter_per_axis[axes[row_idx, 1]] += 1
                else:
                    if axes[row_idx_train, 0] not in nudge_counter_per_axis:
                        nudge_counter_per_axis[axes[row_idx_train, 0]] = 0
                    if axes[row_idx_val, 1] not in nudge_counter_per_axis:
                        nudge_counter_per_axis[axes[row_idx_val, 1]] = 0
                    va_train = (
                        "bottom"
                        if nudge_counter_per_axis[axes[row_idx_train, 0]] % 2 == 0
                        else "top"
                    )
                    va_val = (
                        "top"
                        if nudge_counter_per_axis[axes[row_idx_val, 1]] % 2 == 0
                        else "bottom"
                    )
                    # Increment for the next line on this axis
                    nudge_counter_per_axis[axes[row_idx_train, 0]] += 1
                    nudge_counter_per_axis[axes[row_idx_val, 1]] += 1

                if not train_metric:
                    logger.warning(f"Missing data for training metric: {metric_name}")

                if not val_metric:
                    logger.warning(f"Missing data for validation metric: {metric_name}")

                plot_line(
                    (axes[row_idx, 0] if layout == "horizontal" else axes[row_idx_train, 0]),
                    (axes[row_idx, 1] if layout == "horizontal" else axes[row_idx_val, 1]),
                    epochs,
                    train_metric,
                    val_metric,
                    train_color=colors[i],
                    val_color=colors[i],
                    label=final_label,
                    is_pct=True,
                    best_idx=run["best_epoch"] - 1,  # "best_epoch": best_epoch_idx + 1,
                    # Determine vertical alignment based on an alternating counter for this axis
                    va_train=va_train,
                    va_val=va_val,
                )

        # --- Formatting and Finalization ---
        logger.info(f"Plotted {len(runs_to_plot)} models from {len(metadata_files)} total files.")

        if title:
            plot_title = f"{title} (top-{len(runs_to_plot)})" if top_k is not None else title
            fig.suptitle(f"{plot_title} {title_suffix}", fontsize="large")

        loss_title_str = loss_to_plot.replace("_", " ").title()

        if layout == "horizontal":
            axes[0, 0].set_title(f"Training {loss_title_str}")
            axes[0, 1].set_title(f"Validation {loss_title_str}")
            axes[0, 0].set_ylabel(loss_title_str)
            for metric_idx, metric_name in enumerate(metrics_to_plot):
                metric_name_title = metric_name.replace("_", " ").title()
                row = metric_idx + 1
                axes[row, 0].set_title(f"Training {metric_name_title}")
                axes[row, 1].set_title(f"Validation {metric_name_title}")
                axes[row, 0].set_ylabel(metric_name_title)
            # Set xlabel only on the bottom-most row
            axes[-1, 0].set_xlabel("Epoch")
            axes[-1, 1].set_xlabel("Epoch")
        else:  # vertical
            axes[0, 0].set_title(f"Training {loss_title_str}")
            axes[1, 0].set_title(f"Validation {loss_title_str}")
            axes[0, 0].set_ylabel(loss_title_str)
            axes[1, 0].set_ylabel(loss_title_str)
            for metric_idx, metric_name in enumerate(metrics_to_plot):
                metric_name_title = metric_name.replace("_", " ").title()
                row_train = 2 * (metric_idx + 1)
                axes[row_train, 0].set_title(f"Training {metric_name_title}")
                axes[row_train + 1, 0].set_title(f"Validation {metric_name_title}")
                axes[row_train, 0].set_ylabel(metric_name_title)
                axes[row_train + 1, 0].set_ylabel(metric_name_title)
            # Set xlabel only on the very last subplot
            axes[-1, 0].set_xlabel("Epoch")

        for ax in axes.flatten():
            ax.grid(visible=True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)

        # Apply log scale ONLY to the loss axes
        if log_scale:
            if layout == "horizontal":
                axes[0, 0].set_yscale("log")  # Training Loss
                axes[0, 1].set_yscale("log")  # Validation Loss
            else:  # vertical
                axes[0, 0].set_yscale("log")  # Training Loss
                axes[1, 0].set_yscale("log")  # Validation Loss

        # --- Legend and Layout Adjustment ---
        # The legend logic works for both layouts as it pulls from one of the validation axes
        handles, labels = (
            axes[0, 1] if layout == "horizontal" else axes[1, 0]
        ).get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=legend_bbox_to_anchor,
                ncol=min(len(labels), 1),
                frameon=False,
                fontsize=legend_fontsize,
            )

        # --- Call adjust_text on each axis ---
        for ax, texts in texts_per_axis.items():
            if texts and text_adjuster:
                _annotate_texts(
                    ax=ax,
                    texts_to_adjust=texts,
                    lines_to_avoid=lines_per_axis.get(ax, []),
                    text_adjuster=text_adjuster,
                    text_fontsize=5,
                )
                # adjust_text(
                #     texts,
                #     ax=ax,
                #     objects=lines_per_axis[ax],
                #     only_move={
                #         "text": "y",
                #         "static": "y",
                #         "explode": "xy",
                #         "pull": "xy",
                #     },
                #     # force_text : tuple[float, float] | float, default (0.1, 0.2)
                #     # the repel force from texts is multiplied by this value
                #     force_text=(10, 10),
                #     # force_static : tuple[float, float] | float, default (0.1, 0.2)
                #     # the repel force from points and objects is multiplied by this value
                #     force_static=(5, 5),
                #     # force_explode : tuple[float, float] | float, default (0.1, 0.5)
                #     # same as other forces, but for the forced move of texts away from nearby
                #     # texts and static positions before iterative adjustment
                #     force_explode=(2, 2),
                #     expand_axes=True,
                #     ensure_inside_axes=True,
                #     # arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
                # )

        fig.tight_layout()
        __safe_save_fig(fig, output_path, dpi=dpi, name="Loss comparison", verbose=True)
        plt.show()
        plt.close(fig)


def plot_final_loss_bars(
    metadata_files: list[Path],
    output_path: Path,
    *,
    baselines: list[dict[str, Any]] | None = None,
    loss_to_plot: str = "loss",
    best_metric: str = "val_loss",
    top_k: int | None = None,
    add_param_count_bar: bool = False,
    param_type: Literal["total", "trainable"] = "total",
    filters: dict[tuple[str, ...], set] | None = None,
    label_keys: dict[str, list[str] | tuple[list[str], Callable | None]] | None = None,
    label_template: str | dict[str, str] | None = None,
    label_template_selector: tuple[list[str], str] | None = None,
    allowed_eeg_windows: set[float] | None = None,
    allowed_flame_frames: set[int] | None = None,
    allowed_model_types: set[str] | None = None,
    simplify_model_names: dict[str, str] | None = None,
    y_label: str = "Loss (MSE)",
    y_axis_scale_factor: float = 1.0,
    bar_label_decimals: int = 4,
    bar_label_fontsize: int | str = 8,
    xticklabels_fontsize: int | str | None = None,
    figsize: tuple = (12, 8),
    ylim: tuple[float, float] | None = None,
    dpi: int = 600,
    plt_style: str | list[str] = "default",
) -> None:
    """Create a grouped bar plot comparing the train and validation loss at the best epoch.

    The "best" epoch is determined by finding the minimum or maximum of the `best_metric` in the
    validation history. The bars then show the `train_loss` and `val_loss` at that specific epoch.

    Args:
        metadata_files (list[Path]): A list of paths to metadata files containing model training
            history and configuration.
        output_path (Path): The path where the plot will be saved.
        baselines (list[dict[str, Any]] | None): An optional list of dictionaries, each
            representing a baseline to add to the plot. Each dictionary must contain:
            - 'name' (str): The label for the baseline.
            - 'train' (float): The training metric value.
            - 'val' (float): The validation metric value (used for sorting).
            Example: `[{"name": "Chance", "train": 0.5, "val": 0.5}]`. Defaults to None.
        loss_to_plot (str): The base name of the loss to plot (e.g., "loss", "param_loss",
            "vertex_loss"). The function will look for `train_{loss_to_plot}` and
            `val_{loss_to_plot}` in the history. Defaults to "loss".
        best_metric (str): The metric to use for determining the best epoch (e.g., 'val_loss',
            'val_balanced_accuracy'). The values at this epoch will be used for sorting.
        top_k (int | None): If specified, only the top K models (based on `best_metric`)
            will be plotted. If None, all valid models are plotted.
        add_param_count_bar (bool): If True, adds a third bar for parameter count on a secondary
            y-axis.
        param_type (Literal["total", "trainable"]): Specifies which parameter count to display.
        filters (dict[tuple[str, ...], set] | None): A dictionary to filter which metadata files
            are plotted.
        label_keys (dict[str, list[str] | tuple[list[str], Callable | None]] | None): Maps
            placeholder names to their config path for dynamic label generation.
        label_template (str | dict[str, str] | None): A template string for the x-axis labels,
            e.g., {name} <BACKSLASH><BACKSLASH>n(bs={bs}, dp={model_dp:.0f}%)".
        label_template_selector (tuple[list[str], str] | None): Required if `label_template` is a
            dict. Specifies which metadata key to use to select a template.
        allowed_eeg_windows (set[float] | None): A set of allowed EEG window lengths in seconds.
        allowed_flame_frames (set[int] | None): A set of allowed FLAME window lengths in frames.
        allowed_model_types (set[str] | None): A set of model type names to include.
        simplify_model_names (dict[str, str] | None): A dictionary to simplify model names.
        y_label (str): The base label for the y-axis. Defaults to "Loss (MSE)".
        y_axis_scale_factor (float): A factor to scale the bar label values by. The y-axis
            label will be updated to reflect this scaling (e.g., "Loss (MSE) (x100)").
            Defaults to 1.0 (no scaling).
        bar_label_decimals (int): The number of decimal places to show on the bar labels.
            Defaults to 4.
        bar_label_fontsize (int): The font size for the bar labels.
        xticklabels_fontsize (int | str | None): The font size for the x-axis tick labels.
        figsize (tuple): The size of the figure to create.
        ylim (tuple[float, float] | None): The y-axis limits for the plot.
        dpi (int): The resolution of the figure in dots per inch.
        plt_style (str | list[str]): The style(s) to use for the plot.
    """

    def formatter(v: float) -> str:
        """Format the bar label with scaling and specified decimals."""
        return f"{(v * y_axis_scale_factor):.{bar_label_decimals}f}"

    if not metadata_files:
        logger.warning("No metadata files provided to plot. Exiting.")
        return

    # Define the loss keys to be used throughout the function
    train_loss_key = f"train_{loss_to_plot}"
    val_loss_key = f"val_{loss_to_plot}"

    # The metric for finding the best epoch still needs the val_ prefix.
    best_metric_key = f"val_{best_metric}" if "val_" not in best_metric else best_metric

    lower_is_better = "loss" in best_metric.lower()
    results = []

    for metadata_path in metadata_files:
        try:
            metadata = load_json(metadata_path)

            if filters and metadata_is_filtered_out(
                metadata, filters, metadata_name=metadata_path.parent.name
            ):
                continue

            dataset_meta = metadata.get("dataset", metadata.get("dataset_params", {}))
            eeg_win_sec = dataset_meta.get("eeg_window_seconds")
            flame_win_sec = dataset_meta.get("flame_window_seconds")
            video_fps = dataset_meta.get("video_fps", 29.97)
            model_class_name = extract_model_name(metadata)

            if simplify_model_names:
                model_class_name = simplify_model_names.get(model_class_name, model_class_name)

            if any(v is None for v in [eeg_win_sec, flame_win_sec, video_fps]):
                continue

            flame_frames = round(flame_win_sec * video_fps)

            if (
                (allowed_model_types and model_class_name not in allowed_model_types)
                or (allowed_eeg_windows and eeg_win_sec not in allowed_eeg_windows)
                or (allowed_flame_frames and flame_frames not in allowed_flame_frames)
            ):
                continue

            history = metadata.get("history", metadata.get("loss", {}))
            train_loss_history = history.get(train_loss_key, [])
            val_loss_history = history.get(val_loss_key, [])
            best_metric_history = history.get(best_metric_key, [])

            if not train_loss_history or not val_loss_history or not best_metric_history:
                logger.warning(
                    f"Skipping {metadata_path.parent.name}: Missing required history data."
                )
                continue

            best_epoch_idx = (
                np.argmin(best_metric_history)
                if lower_is_better
                else np.argmax(best_metric_history)
            )
            best_metric_value = best_metric_history[best_epoch_idx]
            train_loss_at_best = train_loss_history[best_epoch_idx]
            val_loss_at_best = val_loss_history[best_epoch_idx]
            best_epoch = best_epoch_idx + 1

            # --- Dynamically Build the Label ---
            if label_keys and label_template:
                final_template_str = ""
                if isinstance(label_template, dict):
                    if not label_template_selector:
                        raise ValueError(
                            "`label_template_selector` must be provided "
                            "if `label_template` is a dict."
                        )
                    selector_key_path, default_value = label_template_selector
                    selector_value = get_nested_value(
                        metadata, selector_key_path, default=default_value
                    )
                    final_template_str = label_template.get(
                        selector_value, label_template.get(default_value, "")
                    )
                elif isinstance(label_template, str):
                    final_template_str = label_template

                label = __build_legend(
                    metadata=metadata, key_map=label_keys, template=final_template_str
                )
            else:
                label = f"{model_class_name}\n(Best @epoch {best_epoch})"

            # Extract parameter count
            param_count = metadata.get("num_params", {}).get(param_type, 0)

            results.append(
                {
                    "label": label,
                    "train_loss": train_loss_at_best,
                    "val_loss": val_loss_at_best,
                    "best_metric_value": best_metric_value,
                    "param_count": param_count,
                }
            )

        except Exception as e:
            logger.error(f"Failed to process {metadata_path}: {e}")

    # --- Inject Baselines ---
    if baselines:
        logger.info(f"Adding {len(baselines)} baseline(s) to the plot.")
        for baseline in baselines:
            if not all(k in baseline for k in ["name", "train", "val"]):
                logger.error(f"Baseline dictionary is malformed, skipping: {baseline}")
                continue
            results.append(
                {
                    "label": baseline["name"],
                    "train_loss": baseline["train"],
                    # For regression, the metric value is the loss. For classification,
                    # it's the score. We assume the user provides the correct value for the
                    # metric being used.
                    "val_loss": (
                        baseline["val"] if "loss" in best_metric.lower() else 0.0
                    ),  # val_loss is plotted, val is for sorting
                    "best_metric_value": baseline["val"],
                    # Baselines can optionally have a param count
                    "param_count": baseline.get("num_params", {}).get(param_type, 0),
                }
            )

    if not results:
        logger.error("No valid models found to plot after filtering. No plot will be generated.")
        return

    df = pd.DataFrame(results).dropna()
    if df.empty:
        logger.error("DataFrame is empty after processing. No plot will be generated.")
        return

    # Sort by the best metric value and take the top K
    df_sorted = df.sort_values("best_metric_value", ascending=lower_is_better)
    if top_k is not None:
        df_sorted = df_sorted.head(top_k)

    # --- Plotting with Matplotlib ---
    with plt.style.context(plt_style):
        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

        labels = df_sorted["label"]

        # Scale data values before plotting
        train_values = df_sorted["train_loss"] * y_axis_scale_factor
        val_values = df_sorted["val_loss"] * y_axis_scale_factor
        param_values = df_sorted["param_count"]

        x = np.arange(len(labels))
        num_bars = 3 if add_param_count_bar else 2
        width = 0.8 / num_bars

        # Bar positions for [Train, Val, Params] order
        pos1 = x - width
        pos2 = x
        pos3 = x + width

        # Plot primary bars (loss/metric)
        loss_title_str = loss_to_plot.replace("_", " ").title()
        rects1 = ax1.bar(
            pos1 if num_bars == 3 else x - width / 2,
            train_values,
            width,
            label=f"Train {loss_title_str}",
            alpha=0.8,
        )
        rects2 = ax1.bar(
            pos2 if num_bars == 3 else x + width / 2,
            val_values,
            width,
            label=f"Validation {loss_title_str}",
            alpha=0.8,
        )

        ax1.set_title(f"Top {len(df_sorted)} Model Performance (ranked by {best_metric_key})")

        # --- Updated Y-axis and Bar Label Logic ---
        scale_indicator_str = ""
        if y_axis_scale_factor != 1.0:
            try:
                # Check if it's a clean power of 10
                exponent = math.log10(y_axis_scale_factor)
                if np.isclose(exponent, round(exponent)):
                    scale_indicator_str = f" ($\\times 10^{{{round(exponent)}}})$"
                else:  # Not a power of 10, use standard multiplication
                    factor_str = (
                        f"{int(y_axis_scale_factor)}"
                        if y_axis_scale_factor == int(y_axis_scale_factor)
                        else f"{y_axis_scale_factor}"
                    )
                    scale_indicator_str = f" ($\\times {factor_str}$)"
            except (
                ValueError,
                TypeError,
            ):  # Fallback for non-positive or invalid scale factors
                factor_str = (
                    f"{int(y_axis_scale_factor)}"
                    if y_axis_scale_factor == int(y_axis_scale_factor)
                    else f"{y_axis_scale_factor}"
                )
                scale_indicator_str = f" (x{factor_str})"

        if loss_to_plot.lower() != "loss":
            y_label = f"{loss_title_str} (MSE)"

        ax1.set_ylabel(y_label + scale_indicator_str)
        ax1.set_xlabel("Model Configuration")
        ax1.set_xticks(x)

        xtick_kwargs: dict[str, int | str] = {
            "rotation": 30,
        }
        if xticklabels_fontsize is not None:
            xtick_kwargs["size"] = xticklabels_fontsize

        # ax1.set_xticklabels(labels, **xtick_kwargs)
        ax1.set_xticklabels(labels, ha="right", fontdict=xtick_kwargs)

        # --- Bar Label Formatting ---
        # The value `v` passed to the formatter is now already scaled.
        ax1.bar_label(
            rects1,
            padding=3,
            fmt=f"%.{bar_label_decimals}f",
            fontsize=bar_label_fontsize,
        )
        ax1.bar_label(
            rects2,
            padding=3,
            fmt=f"%.{bar_label_decimals}f",
            fontsize=bar_label_fontsize,
        )

        # --- Y-axis Limit Scaling ---
        if ylim is None:
            # Use the scaled values for automatic limit calculation
            max_value_plotted = max(train_values.max(), val_values.max())
            ax1.set_ylim(0, max_value_plotted * 1.25)
        else:
            # Scale the user-provided ylim
            scaled_ylim = (ylim[0] * y_axis_scale_factor, ylim[1] * y_axis_scale_factor)
            ax1.set_ylim(scaled_ylim)

        # --- Secondary Axis for Parameter Count ---
        if add_param_count_bar:
            ax2 = ax1.twinx()

            max_params = param_values.max()
            if max_params >= 1_000_000_000:
                param_scale, param_unit, exponent = 1e9, "B", 9
            elif max_params >= 1_000_000:
                param_scale, param_unit, exponent = 1e6, "M", 6
            elif max_params >= 1_000:
                param_scale, param_unit, exponent = 1e3, "K", 3
            else:
                param_scale, param_unit, exponent = 1, "", 0

            scaled_param_values = param_values / param_scale

            rects3 = ax2.bar(
                pos3,
                scaled_param_values,
                width,
                label=f"Params ({param_type})",
                color="tab:grey",
                alpha=0.6,
            )
            ax2.set_ylabel(f"Parameters ({param_type.capitalize()}) ($\\times 10^{{{exponent}}}$)")
            ax2.set_ylim(0, scaled_param_values.max() * 1.25)

            def param_formatter(v: float) -> str:
                if v == 0:
                    return ""
                if param_unit == "B":
                    return f"{v:.1f}{param_unit}"
                if param_unit == "M":
                    return f"{v:.1f}{param_unit}"
                if param_unit == "K":
                    return f"{v:.1f}{param_unit}"
                return str(int(v))

            ax2.bar_label(rects3, padding=3, fmt=param_formatter, fontsize=bar_label_fontsize)

            ax1.grid(visible=False)
            ax2.grid(visible=False)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                handles1 + handles2,
                labels1 + labels2,
                frameon=True,
                fontsize="small",
                loc="upper left",
            )
        else:
            ax1.legend(frameon=True, fontsize="small")

        fig.tight_layout()
        __safe_save_fig(fig, output_path, dpi=dpi, name="Final loss bars", verbose=True)
        plt.show()
        plt.close(fig)


def plot_history(
    history: TrainingHistoryClassification,
    metrics: list[str] | None = None,
    filepath: str | None = None,
    overall_best_epoch: int | None = None,
    early_stop_epoch: int | None = None,
    phase_info: list[tuple[str, int]] | None = None,
    phase_bg_colors: list[str] | None = None,
    *,
    figsize: tuple[float, float] = (12, 4),
    plt_style: list[str] | str = "default",
    text_adjuster: Literal["adjusttext", "textalloc"] | None = "textalloc",
    dynamic_clip_iqr_multiplier: float | None = None,
    dynamic_clip_on_metrics: bool = False,
    log_scale: bool = False,
    log_scale_loss: bool = False,
    log_scale_metrics: bool = False,
    figtitle: str | None = None,
    y_figtitle: float = 1.0,
    axis_fontsize: str | float = "medium",
    legend_fontsize: str | float = "small",
    text_fontsize: str | float = 8,
    dpi: int = 300,
    show: bool = True,
    sharex: bool = False,
    verbose: bool = True,
    is_combined: bool = False,
    plot_phase_best: bool = True,
) -> None:
    """Plot training and validation history from a `history` dictionary.

    This function generates a multi-panel plot showing the evolution of training and validation
    loss, any specified metrics, and the learning rate schedule if it is present in the history
    dictionary. It supports visualization of multi-phase training and dynamic clipping of outliers.

    Args:
        history (TrainingHistoryClassification): Dictionary containing the
        training history. Must include 'train_loss' and 'val_loss'. Can optionally include
        'lr' and metrics like 'train_acc', 'val_acc'.
        metrics (list[str], optional): List of metric names (without 'train_'/'val_'
            prefix) to plot. If None, all available metrics in `history` are plotted.
            Defaults to None.
        filepath (str, optional): If provided, the plot will be saved to this file path.
            Defaults to None.
        overall_best_epoch (int, optional): If provided, a vertical line indicating the
            overall best model epoch will be plotted in each subplot. Defaults to None.
        early_stop_epoch (int, optional): An alias for `overall_best_epoch` for clarity in
            non-combined training. Defaults to None.
        phase_info (list[tuple[str, int]], optional): Required if `is_combined` is True.
            A list of tuples, where each tuple contains the phase name (str) and its
            duration in epochs (int). Defaults to None.
        phase_bg_colors (list[str], optional): List of colors for the background shading of
            each phase. If not provided, a default color map will be used. Defaults to None.
        figsize (tuple[float, float], optional): Base size of a single subplot in inches
            (width, height). The final figure height is scaled by the number of plots.
            Defaults to (12, 4).
        plt_style (list[str] | str, optional): Matplotlib style(s) to use for the plot.
            Defaults to "default".
        text_adjuster (Literal["adjusttext", "textalloc"] | None, optional): The library to
            use for preventing text annotation overlap. Can be 'adjusttext', 'textalloc', or
            None for no adjustment. Defaults to "textalloc".
        dynamic_clip_iqr_multiplier (float, optional): If provided, enables dynamic y-axis
            clipping based on the Interquartile Range (IQR) to handle outliers. Points
            above the calculated limit are marked. Set to None to disable. Defaults to 1.0.
        dynamic_clip_on_metrics (bool, optional): If True, applies dynamic clipping to metric
            plots as well as the loss plot. Defaults to False.
        log_scale (bool, optional): **Deprecated** argument.
        log_scale_loss (bool, optional): If True, the y-axis for the loss plot will be
            logarithmic. Defaults to False.
        log_scale_metrics (bool, optional): If True, the y-axis for all metric plots will be
            logarithmic. Defaults to False.
        figtitle (str, optional): Title for the entire figure. Defaults to None.
        y_figtitle (float, optional): Vertical position adjustment of the figure title.
            Defaults to 1.0.
        axis_fontsize (str | float, optional): Font size for the axis labels.
            Defaults to "medium".
        legend_fontsize (str | float, optional): Font size for the legends.
            Defaults to "small".
        text_fontsize (str | float, optional): Font size for text annotations on the plot.
            Defaults to 8.
        dpi (int, optional): Dots per inch for the saved plot. Defaults to 300.
        show (bool, optional): If True, the plot will be displayed interactively.
            Defaults to True.
        sharex (bool, optional): If True, all subplots will share the same x-axis.
            Defaults to True.
        verbose (bool, optional): If True, prints information about the plot creation process.
            Defaults to True.
        is_combined (bool, optional): If True, treats the history as a combination of multiple
            training phases defined by `phase_info`. Defaults to False.
        plot_phase_best (bool, optional): If True and `is_combined` is True, plots vertical
            lines indicating the best epoch within each training phase. Defaults to True.
    """

    class Phase(TypedDict):
        name: str
        start: int
        end: int
        len_: int

    history_dict = history.to_dict(ignore_empty=False)  # Important: keep empty lists
    history_len = len(history_dict["train_loss"])  # Get total epochs from loss history
    epochs = np.arange(1, history_len + 1)

    if early_stop_epoch is not None and overall_best_epoch is not None:
        raise ValueError(
            "`overall_best_epoch` and `early_stop_epoch` cannot be provided at the same time.",
        )

    if log_scale:
        logger.warning(
            "`log_scale` is deprecated. Use `log_scale_loss` or `log_scale_metrics` instead."
        )

    if early_stop_epoch is not None:
        overall_best_epoch = early_stop_epoch

    # --- Dynamically detect loss plots ---
    all_keys = history_dict.keys()
    loss_types = np.unique(
        [
            key.replace("train_", "").replace("val_", "")
            for key in all_keys
            if key.endswith("_loss") and history_dict[key]
        ]
    ).tolist()

    # Make sure that 'loss' is always the first loss type if it exists
    if "loss" in loss_types:
        loss_types.remove("loss")
        loss_types.insert(0, "loss")

    # --- Input Validation and Phase Calculation ---
    phases: list[Phase] = []
    phase_colors: dict[str, str] = {}
    if is_combined:
        if not phase_info:
            raise ValueError("`phase_info` must be provided when `is_combined` is True.")

        calculated_len = sum(length for _, length in phase_info)
        if calculated_len != history_len:
            raise ValueError(
                f"Total length of phases ({calculated_len}) defined in `phase_info` "
                f"does not match history length ({history_len}).",
            )

        current_epoch = 0
        for name, length in phase_info:
            if length <= 0:
                # Warn or skip phases with zero length? Let's skip for robustness.
                logger.warning(
                    f"Phase '{name}' has zero or negative length ({length}) and will be skipped.",
                )
                continue
            start_epoch = current_epoch
            end_epoch = current_epoch + length  # Exclusive end epoch
            phases.append(Phase(name=name, start=start_epoch, end=end_epoch, len_=length))
            current_epoch = end_epoch

        # Generate phase colors (once)
        if phase_bg_colors is None:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            default_colors = prop_cycle.by_key()["color"]
            phase_colors = {
                phase["name"]: default_colors[i % len(default_colors)]
                for i, phase in enumerate(phases)
            }
        elif len(phase_bg_colors) != len(phases):
            logger.warning(
                f"Number of provided `phase_bg_colors` ({len(phase_bg_colors)}) does not "
                f"match the number of valid phases ({len(phases)}). Using default colors.",
            )
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            default_colors = prop_cycle.by_key()["color"]
            phase_colors = {
                phase["name"]: default_colors[i % len(default_colors)]
                for i, phase in enumerate(phases)
            }
        else:
            phase_colors = {
                phase["name"]: color for phase, color in zip(phases, phase_bg_colors, strict=True)
            }
    else:
        # If not combined, treat the whole history as a single phase for consistency
        pass  # No phase processing needed

    # --- Metrics Determination ---
    if metrics is None:
        # Auto-detect metrics excluding loss and specific keys like 'early_stop_epoch'
        metrics = sorted(
            [
                key.split("_", 1)[1]  # Get name after train_/val_
                for key in history_dict
                if (key.startswith(("train_", "val_")))
                and "loss" not in key.lower()  # Exclude loss
                and key != "early_stop_epoch"  # Exclude specific non-metric keys if present
            ]
        )
        # Remove duplicates from auto-detection
        metrics = sorted(set(metrics))

    with plt.style.context(plt_style):
        # Lists of objects to avoid overlap with (for the case of the loss plot)
        lines_to_avoid_loss: dict[str, list[Line2D]] = {}

        # --- Plot Setup ---
        fig, axes_dict = _setup_plot_layout(
            history,
            loss_types,
            metrics,
            figsize=figsize,
            sharex=sharex,
        )
        loss_axes, metric_axes, ax_lr = (
            axes_dict["losses"],
            axes_dict["metrics"],
            axes_dict["lr"],
        )

        # --- Plot all loss curves ---
        train_line_loss_color = "tab:blue"
        val_line_loss_color = "tab:orange"
        for i, loss_type in enumerate(loss_types):
            ax_loss = loss_axes[i]
            train_key = f"train_{loss_type}"
            val_key = f"val_{loss_type}"

            if train_key not in history_dict or val_key not in history_dict:
                continue

            train_data = np.array(history_dict[train_key])  # Convert to numpy array
            val_data = np.array(history_dict[val_key])

            # Dynamic Clipping Logic for Loss ---
            if dynamic_clip_iqr_multiplier is not None:
                (
                    ylim,
                    (train_loss_plot, val_loss_plot),
                    (train_outliers, val_outliers),
                ) = _get_dynamic_ylim_and_outliers(
                    train_data, val_data, iqr_multiplier=dynamic_clip_iqr_multiplier
                )

                if ylim is not None:
                    ax_loss.set_ylim(ylim)
            else:
                # Prepare data for plotting without clipping
                train_loss_plot, val_loss_plot = train_data, val_data
                train_outliers, val_outliers = (
                    np.zeros_like(train_data, dtype=bool),
                    np.zeros_like(val_data, dtype=bool),
                )

            (train_line,) = ax_loss.plot(
                epochs,
                train_loss_plot,
                label="Training Loss",  # color="tab:blue"
            )
            (val_line,) = ax_loss.plot(
                epochs,
                val_loss_plot,
                label="Validation Loss",  # color="tab:orange"
            )
            lines_to_avoid_loss[loss_type] = [train_line, val_line]

            # Add markers for detected outliers
            train_line_loss_color = train_line.get_color()
            if np.any(train_outliers):
                ax_loss.scatter(
                    np.where(train_outliers)[0],
                    train_loss_plot[train_outliers],
                    marker="^",
                    color=train_line_loss_color,
                    s=100,
                    zorder=5,
                    label="Train Loss Outliers",
                )
            val_line_loss_color = val_line.get_color()
            if np.any(val_outliers):
                ax_loss.scatter(
                    np.where(val_outliers)[0],
                    val_loss_plot[val_outliers],
                    marker="^",
                    color=val_line_loss_color,
                    s=100,
                    zorder=5,
                    label="Validation Loss Outliers",
                )

            ax_loss.set_ylabel(loss_type.replace("_", " ").title(), fontsize=axis_fontsize)
            ax_loss.set_title(loss_type.replace("_", " ").title(), fontsize=axis_fontsize)

            # --- Plot Learning Rate Curve (if applicable) ---
            if ax_lr and i == 0:  # Make sure to plot LR only once
                _plot_lr_curve(
                    ax_lr,
                    history,
                    legend_fontsize=legend_fontsize,
                    fontsize=axis_fontsize,
                )

                # Update x-ticks for all subplots so that only integer ticks are shown
                _format_xaxis_epoch_ticks(ax_lr, history_len)

            if log_scale_loss:
                ax_loss.set_yscale("log")
                __format_log_scale(ax_loss)

        # Plot Metrics
        for i, metric in enumerate([*loss_types, *metrics]):
            texts_to_adjust: list[Text] = []  # List of text objects to adjust
            lines_to_avoid: list[Line2D] = []  # List of line objects to avoid

            # Choose the right key and the right ax
            ax_metric = metric_axes[i - len(loss_axes)] if i >= len(loss_types) else loss_axes[i]
            is_loss_plot = i < len(loss_types)
            lower_is_better = is_loss_plot

            # Get the key expected by TrainingHistory
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"

            if train_key not in history_dict or val_key not in history_dict:
                logger.warning(
                    f"History is missing keys '{train_key}' and/or '{val_key}'. "
                    f"History has the following keys: {history_dict.keys()}. "
                    f"Skipping plot for metric '{metric}'"
                )
                ax_metric.set_title(f"Metric '{metric}' Data Missing")
                ax_metric.text(
                    0.5,
                    0.5,
                    "Data not found",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax_metric.transAxes,
                )
                continue  # Skip to next metric if keys are missing

            # Convert to numpy array
            train_values = np.array(history_dict[train_key])
            val_values = np.array(history_dict[val_key])

            # --- Dynamic Clipping Logic for Metrics ---
            if dynamic_clip_iqr_multiplier is not None and dynamic_clip_on_metrics:
                (
                    ylim_metric,
                    (train_values_plot, val_values_plot),
                    (train_outliers_metric, val_outliers_metric),
                ) = _get_dynamic_ylim_and_outliers(
                    train_values, val_values, iqr_multiplier=dynamic_clip_iqr_multiplier
                )

                if ylim_metric is not None:
                    ax_metric.set_ylim(ylim_metric)
            else:
                train_values_plot, val_values_plot = train_values, val_values
                train_outliers_metric, val_outliers_metric = (
                    np.zeros_like(train_values, dtype=bool),
                    np.zeros_like(val_values, dtype=bool),
                )

            # Only do this for the metrics (not the loss plots)
            train_metric_color = "tab:blue"
            val_metric_color = "tab:orange"
            if not is_loss_plot:
                metric_label = " ".join(metric.split("_")).title()
                (train_line,) = ax_metric.plot(
                    epochs,
                    train_values_plot,
                    label=f"Training {metric_label}",
                )
                (val_line,) = ax_metric.plot(
                    epochs,
                    val_values_plot,
                    label=f"Validation {metric_label}",
                )
                lines_to_avoid.extend([train_line, val_line])

                # Add markers for detected outliers
                train_metric_color = train_line.get_color()
                if np.any(train_outliers_metric):
                    ax_metric.scatter(
                        np.where(train_outliers_metric)[0],
                        train_values_plot[train_outliers_metric],
                        marker="^",
                        color=train_metric_color,
                        s=100,
                        zorder=5,
                        label=f"Train {metric_label} Outliers",
                    )
                val_metric_color = val_line.get_color()
                if np.any(val_outliers_metric):
                    ax_metric.scatter(
                        np.where(val_outliers_metric)[0],
                        val_values_plot[val_outliers_metric],
                        marker="^",
                        color=val_metric_color,
                        s=100,
                        zorder=5,
                        label=f"Val {metric_label} Outliers",
                    )

                ax_metric.set_ylabel(metric_label, fontsize=axis_fontsize)
                ax_metric.set_title(
                    f"Training and Validation {metric_label}", fontsize=axis_fontsize
                )

            # Annotate last epoch's metric values (optional, can get crowded)
            # Consider only annotating the overall best or removing if too cluttered
            last_epoch_idx = history_len - 1
            if last_epoch_idx >= 0:
                try:
                    if "loss" not in metric.lower():
                        # Format as percentage only if metric seems like it
                        # (e.g., < 1.0 or name implies it)
                        # Simple heuristic: check if max value is <= 1
                        is_percent_like = np.nanmax(val_values) <= 1.0 or any(
                            s in metric.lower()
                            for s in [
                                "accuracy",
                                "iou",
                                "f1",
                                "score",
                                "precision",
                                "recall",
                            ]
                        )
                    else:
                        is_percent_like = False

                    train_fmt_str = _format_value(
                        train_values[last_epoch_idx],
                        plt_style=plt_style,
                        is_percent_like=is_percent_like,
                    )
                    val_fmt_str = _format_value(
                        val_values[last_epoch_idx],
                        plt_style=plt_style,
                        is_percent_like=is_percent_like,
                    )

                    texts_to_adjust.append(
                        ax_metric.text(
                            last_epoch_idx + 1,
                            train_values[last_epoch_idx],
                            train_fmt_str,
                            color=(train_line_loss_color if is_loss_plot else train_metric_color),
                            verticalalignment="bottom",
                            horizontalalignment="right",
                            fontsize=text_fontsize,
                        )
                    )
                    texts_to_adjust.append(
                        ax_metric.text(
                            last_epoch_idx + 1,
                            val_values[last_epoch_idx],
                            val_fmt_str,
                            color=(val_line_loss_color if is_loss_plot else val_metric_color),
                            verticalalignment="top",
                            horizontalalignment="right",
                            fontsize=text_fontsize,
                        )
                    )
                except IndexError:
                    logger.warning(
                        f"Could not annotate last epoch for metric '{metric}'. "
                        "Index out of bounds.",
                    )

            # Add vertical lines for best epochs
            if is_combined and plot_phase_best and val_values is not None:
                for _, phase in enumerate(phases):
                    phase_val_data = val_values[phase["start"] : phase["end"]]
                    if len(phase_val_data) == 0:
                        continue  # Skip if phase has no data points

                    # Find best epoch within the phase
                    best_epoch_in_phase_rel = (
                        np.argmin(phase_val_data) if lower_is_better else np.argmax(phase_val_data)
                    )
                    best_epoch_abs = phase["start"] + best_epoch_in_phase_rel
                    best_val = phase_val_data[best_epoch_in_phase_rel]

                    # Determine format for label
                    is_percent_like_metric = not is_loss_plot and (np.nanmax(val_values) <= 1.0)
                    fmt_str = _format_value(
                        best_val,
                        plt_style=plt_style,
                        is_percent_like=is_percent_like_metric,
                    )

                    # Plot line for phase best
                    line_color = darken_color(phase_colors[phase["name"]], amount=0.3)
                    best_epoch_line = ax_metric.axvline(
                        x=best_epoch_abs + 1,
                        color=line_color,
                        linestyle=":",  # Dotted for phase best
                        label=f"{phase['name']} Best ({fmt_str}) @epoch {best_epoch_abs + 1}",
                        alpha=0.6,
                    )
                    lines_to_avoid.append(best_epoch_line)

            # Add overall best epoch line (works for both combined and non-combined)
            if (
                overall_best_epoch is not None
                and 0 <= overall_best_epoch < history_len
                and val_values is not None
            ):
                overall_best_val = val_values[overall_best_epoch]

                # Determine if the value should be formatted as a percentage
                is_percent_like_metric = not is_loss_plot and (
                    np.nanmax(val_values) <= 1.0
                    or any(
                        s in (metric or "").lower()
                        for s in [
                            "accuracy",
                            "iou",
                            "f1",
                            "score",
                            "precision",
                            "recall",
                        ]
                    )
                )
                fmt_str = _format_value(
                    overall_best_val,
                    plt_style=plt_style,
                    is_percent_like=is_percent_like_metric,
                )

                label_prefix = "Overall Best" if is_combined else "Best Model"
                best_epoch_line = ax_metric.axvline(
                    x=overall_best_epoch + 1,
                    color="r",
                    linestyle="--",  # Dashed red for overall best
                    label=f"{label_prefix} ({fmt_str}) @epoch {overall_best_epoch + 1}",
                    alpha=0.6,
                )
                lines_to_avoid.append(best_epoch_line)

                # Add annotations for the values at the best epoch
                if train_values is not None:
                    train_val_at_best = train_values[overall_best_epoch]

                    train_fmt_at_best = _format_value(
                        train_val_at_best,
                        plt_style=plt_style,
                        is_percent_like=is_percent_like_metric,
                    )

                    # Create the text objects and add them to our list for later adjustment
                    # Only add them if the values are different from train_values[last_epoch_idx],
                    # and val_values[last_epoch_idx]
                    if last_epoch_idx >= 0 and last_epoch_idx != overall_best_epoch:
                        # Only add text if the values are different from the last epoch's values
                        # to avoid cluttering the plot with redundant information
                        texts_to_adjust.append(
                            ax_metric.text(
                                overall_best_epoch + 1,
                                train_val_at_best,
                                train_fmt_at_best,
                                color=(
                                    train_line_loss_color if is_loss_plot else train_metric_color
                                ),
                                fontsize=text_fontsize,
                                ha="center",
                            )
                        )

                # Do the same for validation data
                if val_values is not None:
                    val_val_at_best = val_values[overall_best_epoch]

                    val_fmt_at_best = _format_value(
                        val_val_at_best,
                        plt_style=plt_style,
                        is_percent_like=is_percent_like_metric,
                    )

                    if last_epoch_idx >= 0 and last_epoch_idx != overall_best_epoch:
                        texts_to_adjust.append(
                            ax_metric.text(
                                overall_best_epoch + 1,
                                val_val_at_best,
                                val_fmt_at_best,
                                color=(val_line_loss_color if is_loss_plot else val_metric_color),
                                fontsize=text_fontsize,
                                ha="center",
                            )
                        )

            elif overall_best_epoch is not None and history_len > 0:
                logger.warning(
                    f"overall_best_epoch ({overall_best_epoch}) is outside the valid "
                    f"range [0, {history_len - 1}). Line not plotted.",
                )

            # Annotate the texts for the current axis
            _annotate_texts(
                ax_metric,
                texts_to_adjust,
                (lines_to_avoid + lines_to_avoid_loss[metric] if is_loss_plot else lines_to_avoid),
                text_adjuster=text_adjuster,
                text_fontsize=text_fontsize,
            )

        # --- Add Phase Shading and Best Epoch Lines (Loop through axes again) ---
        all_plot_axes = [*loss_axes, *metric_axes]
        for i, ax in enumerate(all_plot_axes):
            is_loss_plot = i < len(loss_axes)

            # --- Phase Background and Legend Patches ---
            phase_legend_patches = []  # Store patches for THIS axis legend
            if is_combined:
                for i_p, phase in enumerate(phases):
                    # Plot the background segment
                    if phase["len_"] > 0:  # Only shade if phase has length
                        # Calculate shifted boundaries
                        span_start = phase["start"] - 0.5

                        # Make the last phase go over the last epoch a bit
                        if i_p == len(phases) - 1:
                            span_end = phase["end"] + 1.0
                        else:
                            span_end = phase["end"] - 0.5  # phase["end"] is exclusive index

                            # Ensure end doesn't exceed slightly past the last data point's center
                            span_end = min(span_end, history_len - 0.5)

                        # Ensure start doesn't go below -0.5
                        span_start = max(span_start, -0.5)

                        # Ensure start < end before plotting
                        if span_start < span_end:
                            ax.axvspan(
                                span_start,
                                span_end,
                                color=phase_colors[phase["name"]],
                                alpha=0.3,
                                zorder=-10,  # Ensure it's behind lines
                            )
                        else:
                            # This might happen if a phase has length 1 and is at the very
                            # start/end or due to rounding issues with tiny phases. Log a warning.
                            logger.warning(
                                f"Skipping axvspan for phase '{phase['name']}' due to "
                                f"non-positive span width ({span_start=}, {span_end=}).",
                            )

                        # Create a patch for the legend
                        phase_legend_patches.append(
                            mpatches.Patch(
                                color=phase_colors[phase["name"]],
                                label=phase["name"],
                                alpha=0.3,
                            ),  # Make legend patch visible
                        )

            # --- Configure Legend (Combining line handles and phase patches) ---
            # Improve legend placement: try to put it outside plot area if many entries
            handles, labels = ax.get_legend_handles_labels()
            all_handles = handles + phase_legend_patches  # Combine handles
            all_labels = labels + [p.get_label() for p in phase_legend_patches]  # Combine labels
            if all_handles:  # Only add legend if there's something to show
                # Sort legend entries:
                # 1. Training/Val lines first, then
                # 2. Phase backgrounds, then
                # 3. Best lines
                # Combine existing line handles/labels with the phase patches

                # Define the desired order based on labels
                order_preference = {
                    "Training": 0,
                    "Validation": 1,  # Train/Val first
                    # Phase names will come after Train/Val
                    "Best": 10,
                    "Overall Best": 11,  # Best lines last
                }

                def get_sort_key(label: str, order_preference: dict[str, int]) -> int:
                    for prefix, order in order_preference.items():
                        if label.startswith(prefix):
                            return order
                    # Default for phase names or others not matched
                    return 5  # Place phases between Train/Val and Best

                # Create pairs of (handle, label) and sort them
                combined_legend_items = sorted(
                    zip(all_handles, all_labels, strict=True),
                    key=lambda item: get_sort_key(item[1], order_preference),
                )

                # Unzip the sorted pairs
                sorted_handles, sorted_labels = (
                    zip(*combined_legend_items, strict=True) if combined_legend_items else ([], [])
                )

                try:
                    ax.legend(sorted_handles, sorted_labels, fontsize=legend_fontsize)
                except IndexError:
                    logger.warning("Could not reorder legend items. Using default order.")
                    ax.legend(
                        # bbox_to_anchor=(1, 0.5), # Place legend outside plot area to the right
                        fontsize=legend_fontsize,
                    )

            # Update x-ticks for all subplots so that only integer ticks are shown
            _format_xaxis_epoch_ticks(ax, history_len)

            # --- Apply log scale conditionally (only on metrics) ---
            if not is_loss_plot and log_scale_metrics:
                ax.set_yscale("log")
                __format_log_scale(ax)

        # Set common x-label only on the last subplot
        last_ax = ax_lr if ax_lr else loss_axes[-1]
        if metric_axes:
            last_ax = metric_axes[-1]
        last_ax.set_xlabel("Epoch", fontsize=axis_fontsize)

        if figtitle:  # Only add title if it's not None or empty
            fig.suptitle(figtitle, fontsize="large", y=y_figtitle)

        # --- Save or Show Plot ---
        __safe_save_fig(fig, filepath, dpi=dpi, verbose=verbose)

        # If not saving, show the plot (useful in interactive environments)
        if show:
            plt.show()
        plt.close(fig)  # Close figure even if not shown/saved to free memory


def plot_confusion_matrix(
    clf: PredictorProtocol | nn.Module,
    dataset: Dataset | torch.Tensor,
    *,
    batch_size: int = 32,
    classes: Sequence[str] | None = None,
    filepath: str | Path | None = None,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 300,
    show: bool = True,
) -> None:
    """Plot a confusion matrix for a classifier on a given dataset.

    Args:
        clf (PredictorProtocol | nn.Module): The classifier to evaluate. Can be a scikit-learn
            style object with a `.predict()` method or a raw PyTorch `nn.Module`.
        dataset (Dataset | np.ndarray): The dataset to use for predictions. It can be a
            PyTorch Dataset or a NumPy array.
        y_true (np.ndarray): True labels for the dataset.
        batch_size (int, optional): Batch size for predictions. Defaults to 64.
        classes (Sequence[str], optional): List of class names. If None, uses default labels.
        filepath (str | Path, optional): Path to save the plot. If None, the plot will not be
            saved.
        figsize (tuple[float, float], optional): Size of the figure in inches (width, height).
            Defaults to (10, 10).
        dpi (int, optional): Dots per inch for the saved plot. Defaults to 300.
        show (bool, optional): If True, displays the plot interactively. Defaults to True.
    """
    # Use batch prediction creating a dataloader
    dataloader = DataLoader(
        (dataset if isinstance(dataset, Dataset) else torch.utils.data.TensorDataset(dataset)),
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for confusion matrix
    )
    y_pred_l = []
    y_true_l = []
    for batch in dataloader:
        inputs, targets = batch

        if isinstance(clf, PredictorProtocol):
            pred = clf.predict(inputs)
        elif isinstance(clf, nn.Module):
            inputs = inputs.to(DEVICE)
            clf = clf.to(DEVICE)
            with torch.no_grad():
                # The predicted logits (i.e., clf(inputs)) is of shape (batch, n_classes),
                # we need to choose the class with highest probability
                logits = clf(inputs)
                if logits.ndim == 2 and logits.shape[1] > 1:
                    # For multi-class classification
                    pred = logits.argmax(dim=1).cpu()
                elif logits.ndim == 1:
                    # For binary classification, logits is a single value per sample
                    pred = (logits > 0.5).long().cpu()
                else:
                    pred = logits.cpu()  # Fallback, but should not happen
        else:
            logger.warning(
                "The provided classifier/model does not have a 'predict' or forward method"
                "Ensure that the model is compatible with the dataset.",
            )
            return

        y_pred_l.append(
            pred.numpy().astype(int) if isinstance(pred, torch.Tensor) else pred.astype(int)
        )
        y_true_l.append(
            targets.numpy().astype(int)
            if isinstance(targets, torch.Tensor)
            else targets.astype(int)
        )

    y_pred = np.concatenate(y_pred_l, axis=0)
    y_true = np.concatenate(y_true_l, axis=0)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())

    # Save a classification_report.txt using sklearn's classification_report function
    classification_report_str = classification_report(
        y_true.flatten(),
        y_pred.flatten(),
        target_names=classes,
        zero_division=0,  # Handle zero division gracefully
    )
    if (
        filepath is not None
        and classification_report_str
        and isinstance(classification_report_str, str)
    ):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if "train" in filepath.name:
            output_filename = "train_classification_report.txt"
        elif "test" in filepath.name:
            output_filename = "test_classification_report.txt"
        elif "val" in filepath.name:
            output_filename = "val_classification_report.txt"
        else:
            logger.warning(
                "Unknown dataset split. Classification report saved as 'classification_report.txt'"
            )
            output_filename = "classification_report.txt"

        with (filepath.parent / output_filename).open("w") as f:
            f.write(classification_report_str)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Calculate percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            # annot[i, j] = f"{cm[i, j]}\n{cm_percent[i, j]:.1f}%"
            # Avoid displaying 0 counts
            if cm[i, j] > 0:
                annot[i, j] = f"{cm_percent[i, j]:.0f}%"
            else:
                annot[i, j] = ""

    sns.heatmap(
        cm_percent,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=classes if classes is not None else "auto",
        yticklabels=classes if classes is not None else "auto",
        ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"format": "%.0f%%", "shrink": 0.5},
        vmin=0,
        vmax=100,
    )

    # Adjust colorbar font size
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=20)

    ax.set_yticklabels(ax.get_yticklabels(), size=16)
    ax.set_xticklabels(ax.get_xticklabels(), size=16)
    ax.set_xlabel("Predicted label", fontsize=20)
    ax.set_ylabel("True label", fontsize=20)
    ax.set_title("Confusion Matrix", fontsize=22)
    fig.tight_layout()
    __safe_save_fig(fig, save_plot_path=filepath, dpi=dpi, verbose=True)
    if show:
        plt.show()
    plt.close(fig)


def save_classification_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: Sequence[str | int] | None = None,
    cm_filepath: str | Path | None = None,
    report_filepath: str | Path | None = None,
    cm_title: str = "Confusion Matrix",
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 300,
    show: bool = True,
) -> None:
    """Plot a confusion matrix and save a classification report.

    Args:
        y_true (np.ndarray): True labels for the dataset.
        y_pred (np.ndarray): Predicted labels for the dataset.
        classes (Sequence[str | int], optional): List of class names. If None, uses default labels.
        cm_filepath (str | Path, optional): Path to save the confusion matrix plot.
            If None, the plot will not be saved.
        report_filepath (str | Path, optional): Path to save the classification report text file.
            If None, the report will not be saved.
        cm_title (str, optional): Title for the confusion matrix plot.
            Defaults to "Confusion Matrix".
        figsize (tuple[float, float], optional): Size of the figure in inches (width, height).
            Defaults to (10, 10).
        dpi (int, optional): Dots per inch for the saved plot. Defaults to 300.
        show (bool, optional): If True, displays the plot interactively. Defaults to True.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())

    # Save a classification_report.txt using sklearn's classification_report function
    classes_str = [str(c).title() for c in classes] if classes is not None else "auto"
    classification_report_str = classification_report(
        y_true.flatten(),
        y_pred.flatten(),
        target_names=classes_str,
        zero_division=0,  # Handle zero division gracefully
    )
    if report_filepath is not None and isinstance(classification_report_str, str):
        report_filepath = Path(report_filepath)
        report_filepath.parent.mkdir(parents=True, exist_ok=True)

        with report_filepath.open("w") as f:
            f.write(classification_report_str)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Calculate percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            # annot[i, j] = f"{cm[i, j]}\n{cm_percent[i, j]:.1f}%"
            # Avoid displaying 0 counts
            if cm[i, j] > 0:
                annot[i, j] = f"{cm_percent[i, j]:.2f}%"
            else:
                annot[i, j] = ""

    sns.heatmap(
        cm_percent,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=classes_str,
        yticklabels=classes_str,
        ax=ax,
        # annot_kws={"size": 8},
        cbar_kws={"format": "%.0f%%", "shrink": 0.75},
        vmin=0,
        vmax=100,
    )

    # Adjust colorbar font size
    # cbar = ax.collections[0].colorbar
    # if cbar is not None:
    #     cbar.ax.tick_params(labelsize=20)

    ax.set_yticklabels(ax.get_yticklabels())  # , size=16)
    ax.set_xticklabels(ax.get_xticklabels())  # , size=16)
    ax.set_xlabel("Predicted label")  # , fontsize=20)
    ax.set_ylabel("True label")  # , fontsize=20)
    ax.set_title(cm_title)  # , fontsize=22)
    fig.tight_layout()
    __safe_save_fig(fig, save_plot_path=cm_filepath, dpi=dpi, verbose=True)
    if show:
        plt.show()
    plt.close(fig)
