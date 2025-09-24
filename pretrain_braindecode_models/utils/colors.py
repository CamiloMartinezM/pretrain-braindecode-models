"""Utility functions for generating and manipulating colors."""

import colorsys
from collections.abc import Sequence

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap

from pretrain_braindecode_models.config import logger


def _generate_additional_distinct_colors(
    n_new: int,
    *,
    existing_colors: list[str | tuple],
    s: float = 0.8,
    v: float = 0.9,
    as_255: bool = False,
) -> list[tuple[float, float, float] | str]:
    """Generate new colors that are distinct from a list of existing colors.

    This function creates a large pool of candidate colors and selects the ones
    that are maximally distant (in RGB space) from the colors already in use.

    Args:
        n_new (int): The number of new, distinct colors to generate.
        existing_colors (list[str | tuple]): A list of existing colors (hex, names, or RGB tuples).
        s (float): Saturation for the new colors.
        v (float): Value for the new colors.
        as_255 (bool): Whether to return new colors as RGB tuples in [0, 255] range.

    Returns:
        list: A list of the newly generated colors.
    """
    # Convert existing colors to a consistent RGB float format for distance calculation
    existing_rgb = np.array([mcolors.to_rgb(c) for c in existing_colors])

    # Generate a large pool of candidates to choose from
    # Generating more candidates than needed increases the chance of finding a good fit.
    candidate_pool_size = max(20, n_new * 10)
    candidate_colors = generate_distinct_colors(candidate_pool_size, s=s, v=v, as_255=False)
    candidate_rgb = np.array(candidate_colors)

    new_colors = []

    for _ in range(n_new):
        if candidate_rgb.shape[0] == 0:
            # Fallback: if we run out of candidates, just generate a random one
            new_color = generate_distinct_colors(1, s=s, v=v, as_255=False)[0]
            new_colors.append(new_color)
            continue

        # For each candidate, calculate its minimum distance to any of the existing colors
        # (both the original ones and any newly added ones)
        # Distance is calculated in RGB space.
        # Broadcasting allows calculating all distances at once: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
        distances = np.linalg.norm(
            candidate_rgb[:, np.newaxis, :] - existing_rgb[np.newaxis, :, :], axis=2
        )
        min_distances = np.min(distances, axis=1)

        # Select the candidate that is farthest from all existing colors
        best_candidate_idx = np.argmax(min_distances)
        new_color_rgb = candidate_rgb[best_candidate_idx]

        # Add the selected color to our results and to the pool of existing colors
        new_colors.append(tuple(new_color_rgb))
        existing_rgb = np.vstack([existing_rgb, new_color_rgb])

        # Remove the chosen color from the candidate pool
        candidate_rgb = np.delete(candidate_rgb, best_candidate_idx, axis=0)

    if as_255:
        return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in new_colors]

    # Return as hex for consistency with matplotlib cycle if not as_255
    return [mcolors.to_hex(c) for c in new_colors]


def generate_distinct_colors(
    n: int,
    s: float = 0.8,
    v: float = 0.9,
    *,
    as_255: bool = False,
) -> list[tuple[float, float, float]]:
    """Generate `n` distinct colors that are perceptually different.

    Args:
        n (int): Number of colors to generate
        s (float): Saturation value (0-1), by default 0.8
        v (float): Value value (0-1), by default 0.9
        as_255 (bool): If True, the colors will be in the range [0, 255], by default False

    Returns:
        (list[tuple[float, float, float]]): List of RGB colors in the range [0, 1] or [0, 255]
    """
    golden_ratio_conjugate = 0.618033988749895
    hue = 0.0
    colors = []

    for _ in range(n):
        hue += golden_ratio_conjugate
        hue %= 1

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, s, v)

        colors.append((int(r * 255), int(g * 255), int(b * 255)) if as_255 else (r, g, b))

    return colors


def get_different_colors_from_plt_prop_cycle(
    num_colors: int,
    exclude_colors: list[str] | None = None,
    style: str | list[str] = "default",
    *,
    allow_less: bool = False,
) -> list[str]:
    """Get a list of different colors from the default matplotlib color cycle.

    That is, `plt.rcParams["axes.prop_cycle"].by_key()["color"]`.

    Args:
        num_colors (int): The number of different colors to get.
        exclude_colors (list[str] | None, optional): A list of colors to exclude from the list, by
            default None.
        style (str | list[str], optional): The style to use for the plot, by default "default".
        allow_less (bool, optional): If True, the function will return fewer colors if it can't
            find the requested number of colors, by default False.

    Returns:
        list[str]: A list of different colors from the default matplotlib color cycle.

    Raises:
        ValueError: If the number of different colors requested is greater than the number of
            colors in the default matplotlib color cycle and `allow_less` is False.
    """
    colors = []
    with plt.style.context(style):
        for color in plt.rcParams["axes.prop_cycle"].by_key()["color"]:
            if exclude_colors and color in exclude_colors:
                continue
            if color not in colors:
                colors.append(color)
            if len(colors) == num_colors:
                break

    if not allow_less and len(colors) < num_colors:
        raise ValueError(
            f"Could not find {num_colors} different colors in {style} matplotlib color cycle.",
        )

    return colors


def get_n_colors(
    n: int,
    *,
    cmap: str | None = None,
    s: float = 0.8,
    v: float = 0.9,
    as_255: bool = False,
    exclude_colors: list[str] | None = None,
    style: str | list[str] = "default",
) -> list:
    """Get an appropriate list of `n` distinct colors.

    This function first attempts to pull colors from the active Matplotlib color cycle
    (`misc.get_different_colors_from_plt_prop_cycle()`). If more colors are requested than are
    available in the cycle, it intelligently generates additional, perceptually distinct colors
    that are different from the ones already selected.

    Args:
        n (int): Number of colors to retrieve.
        cmap (str | None): Name of a matplotlib colormap to use (qualitative or continuous). If
            None, the default color cycle will be used.
        s (float): Saturation for generated colors (if needed), by default 0.8.
        v (float): Value for generated colors (if needed), by default 0.9.
        as_255 (bool): If True, generated colors are in the range [0, 255], by default False.
        exclude_colors (list[str] | None): Colors to exclude from the matplotlib cycle.
        style (str | list[str]): Matplotlib style to use for the color cycle, by default "default".

    Returns:
        list: List of colors in either hex strings (if using color cycle) or RGB tuples
            (if generated).
    """
    if cmap:
        colormap = cm.get_cmap(cmap)
        if hasattr(colormap, "N") and colormap.N < 256:  # Discrete colormap
            if n > colormap.N:
                raise ValueError(f"Colormap '{cmap}' only supports {colormap.N} discrete colors.")
            return [mcolors.to_hex(colormap(i)) for i in range(n)]

        # Continuous colormap
        return [mcolors.to_hex(colormap(i / (n - 1))) for i in range(n)]

    # --- Hybrid Color Generation ---
    # 1. Get as many unique colors as possible from the specified style's color cycle.
    initial_colors = []
    with plt.style.context(style):
        for color in plt.rcParams["axes.prop_cycle"].by_key()["color"]:
            if exclude_colors and color in exclude_colors:
                continue
            if color not in initial_colors:
                initial_colors.append(color)
            if len(initial_colors) == n:
                break

    # 2. Check if we have enough colors.
    num_missing = n - len(initial_colors)
    if num_missing <= 0:
        return initial_colors[:n]  # Return the exact number requested.

    # 3. If not, generate the remaining number of colors.
    logger.warning(
        f"Matplotlib color cycle exhausted after finding {len(initial_colors)} colors. "
        f"Generating {num_missing} additional distinct color(s)."
    )

    additional_colors = _generate_additional_distinct_colors(
        n_new=num_missing,
        existing_colors=initial_colors,
        s=s,
        v=v,
        as_255=as_255,
    )

    return initial_colors + additional_colors


def darken_color(color, amount: float = 0.3) -> tuple[float, float, float, float]:  # noqa: ANN001
    """Darken the given color by multiplying (1-amount) darken factors.

    Amount should be between 0 and 1. A value of 0 will return the original color,
    a value of 1 will return black.

    Args:
        color (color-like): Color representation (name, hex, RGB, etc.).
        amount (float, optional): Amount to darken (0.0 to 1.0), by default 0.3

    Returns:
        tuple[float, float, float, float]: RGBA tuple of the darkened color.
    """
    try:
        c = mcolors.to_rgba(color)
        # Darken R, G, B components; keep alpha
        darker_rgb = [max(0.0, comp * (1 - amount)) for comp in c[:3]]
        return (darker_rgb[0], darker_rgb[1], darker_rgb[2], c[3])
    except ValueError:
        logger.warning(f"Could not convert color '{color}' to RGBA. Returning grey.", stacklevel=2)
        return mcolors.to_rgba("grey")  # Fallback color


def get_color_from_value(
    value: float,
    cmap_name: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """Map a value (normalized 0-1) to an RGBA color tuple (0-255)."""
    cmap = cm.get_cmap(cmap_name)

    # Normalize value within vmin/vmax (should already be 0-1, but robust)
    norm_value = max(0.0, min(1.0, (value - vmin) / (vmax - vmin + 1e-10)))
    color_float = cmap(norm_value)  # RGBA float 0-1
    return (np.array(color_float) * 255).astype(np.uint8)  # color_uint8


def color_text(
    text: str,
    value: float,
    cmap_name: str | Colormap = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> str:
    """Color a string using ANSI escape codes based on a numeric value and a matplotlib cmap.

    Args:
        text (str): The string to color.
        value (float): The numeric value (e.g., a percentage from 0.0 to 1.0) that determines the
            color.
        cmap_name (str | Colormap): The name of the matplotlib colormap to use.
        vmin (float): The minimum value of the scale for normalization.
        vmax (float): The maximum value of the scale for normalization.

    Returns:
        str: The color-coded string with ANSI escape codes.
    """
    # Normalize the value
    norm_value = max(0.0, min(1.0, (value - vmin) / (vmax - vmin + 1e-10)))

    # Get the color from the colormap
    cmap = cm.get_cmap(cmap_name)
    rgb_float = cmap(norm_value)[:3]  # Get RGB, ignore alpha

    # Convert float RGB (0-1) to integer RGB (0-255)
    r, g, b = [int(c * 255) for c in rgb_float]

    # Create the ANSI escape code for 24-bit color
    # \033[38;2;r;g;bm -> Set foreground color
    # \033[0m -> Reset color
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def create_custom_colormap(name: str, colors: Sequence[str | tuple[float, ...]]) -> Colormap:
    """Create and register a custom linear segmented colormap in matplotlib.

    If a colormap with the given name already exists, it will be overwritten.

    Args:
        name (str): The name for the new colormap (e.g., 'CustomRdYlGn').
        colors (list[str | tuple[float, ...]]): A list of colors that define the colormap. Colors
            can be specified as hex strings, color names ('red'), or RGB(A) tuples.

    Returns:
        Colormap: The created matplotlib colormap object.
    """
    return LinearSegmentedColormap.from_list(name, colors)
