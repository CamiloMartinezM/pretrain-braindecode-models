"""Utility functions for dimensionality reduction and visualization of high-dimensional data."""

import math
from importlib.util import find_spec as importlib_find_spec
from pathlib import Path
from time import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from pretrain_braindecode_models.config import logger
from pretrain_braindecode_models.utils.colors import get_n_colors

# Install with:
# * conda install -c conda-forge umap-learn
# * OR: pip install umap-learn
if importlib_find_spec("umap") is not None:
    from umap import UMAP

    UMAP_INSTALLED = True
else:
    UMAP_INSTALLED = False

# Prefer using opentsne instead of sklearn.manifold.TSNE because it is faster by using FFT method
# Install with:
# * conda install --channel conda-forge opentsne
# * OR: pip install opentsne
# See: https://github.com/pavlin-policar/openTSNE/
if importlib_find_spec("openTSNE") is not None:
    from openTSNE import TSNE
else:
    from sklearn.manifold import TSNE

# If `use_tsnecuda` is specified when calling `apply_tsne`, then we use the CUDA version of t-SNE
# See: https://github.com/CannyLab/tsne-cuda/blob/main/INSTALL.md
# Install with:
# * conda install tsnecuda -c conda-forge
# * OR: pip install tsnecuda
if importlib_find_spec("tsnecuda") is not None:
    from tsnecuda import TSNE as TSNE_CUDA  # Import the CUDA version of t-SNE

    TSNE_CUDA_INSTALLED = True


def scatter_plot(
    X: np.ndarray,
    indices: list[pd.Index | np.ndarray] | None = None,
    labels: list[str] | None = None,
    style: str = "default",
    cmap: str | None = None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    special_points: list | None = None,
    special_points_labels: list | None = None,
    special_points_markers: list | None = None,
    special_points_sizes: list | None = None,
    special_points_colors: list | None = None,
    figsize: tuple[int, int] = (6, 4),
    filepath: str | Path | None = None,
    *,
    remove_ticks: bool = False,
    show: bool = True,
    dpi: int = 300,
) -> None:
    """Scatter plot for dimensionality reduced data.

    Args:
        X (np.ndarray): The dimensionality reduced data.
        indices (list[pd.Index | np.ndarray], optional): A list of indices specifying different
            groups to plot. Useful to color different groups of points differently, by default []
        labels (list of str, optional): List of labels for each group, by default None
        style (str, optional): The plot style to use, by default "default". This is used to get
            different colors from the default matplotlib color cycle.
        cmap (str | None, optional): The colormap to use for the coloring of labels. If this is
            provided, then `style` is ignored for this coloring. By default None.
        title (str, optional): The title of the plot, by default ""
        xlabel (str, optional): The label for the x axis, by default "x"
        ylabel (str, optional): The label for the y axis, by default "y"
        zlabel (str, optional): The label for the z axis, by default "z"
        special_points (list, optional): Special points to plot, by default None
        special_points_labels (list, optional): Labels for special points, by default None
        special_points_markers (list, optional): Markers for special points, by default None
        special_points_sizes (list, optional): Sizes for special points, by default None
        special_points_colors (list, optional): Colors for special points, by default None
        figsize (tuple[int, int], optional): The figure size, by default (6, 4)
        filepath (str | Path | None, optional): The path to save the plot, by default None
        remove_ticks (bool, optional): Whether to remove the ticks from the plot, by default False
        show (bool, optional): Whether to show the plot, by default True
        dpi (int, optional): The figure resolution, by default 300

    Raises:
        ValueError: If the dimensionality of X is not 1, 2, or 3.
    """
    if style is None:
        style = "default"

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        num_axes = X.shape[1]

        if num_axes not in [1, 2, 3]:
            raise ValueError("X must have either 1, 2 or 3 columns for 1D, 2D or 3D plotting.")

        if num_axes in [1, 2]:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")

        # If the indices are not provided, plot all the data points as a single class
        if not indices:
            indices = [pd.Index(range(X.shape[0]))]

        # Use class labels if provided, otherwise use default
        if labels is None:
            labels = [f"Class {i + 1}" for i in range(len(indices))]

        # Get colors for each class
        colors = get_n_colors(len(indices), style=style, cmap=cmap)

        # Sort the indices list based on how many points there are. The last must be the one with
        # less points, so they are most likely to be put on top
        labels, indices = zip(*sorted(zip(labels, indices, strict=True)), strict=True)

        for i, idx in enumerate(indices):
            label = labels[i]
            scatter_kwargs = {"label": label, "alpha": 0.6, "color": colors[i]}
            if num_axes == 1:
                ax.scatter(X[idx], np.zeros_like(X[idx]), **scatter_kwargs)
            elif num_axes == 2:
                ax.scatter(X[idx, 0], X[idx, 1], **scatter_kwargs)
            else:
                ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], **scatter_kwargs)

        # Plot special points if provided
        if special_points is not None:
            # Default values
            if special_points_markers is None:
                special_points_markers = "X"
            if special_points_sizes is None:
                special_points_sizes = 200
            if special_points_colors is None:
                special_points_colors = "red"
            if special_points_labels is None:
                special_points_labels = ["Special Points"] * len(special_points)

            # Convert single values to lists for consistent processing
            if isinstance(special_points_markers, str):
                special_points_markers = [special_points_markers] * len(special_points)
            if isinstance(special_points_sizes, (int, float)):
                special_points_sizes = [special_points_sizes] * len(special_points)
            if isinstance(special_points_colors, str):
                special_points_colors = [special_points_colors] * len(special_points)
            if isinstance(special_points_labels, str):
                special_points_labels = [special_points_labels] * len(special_points)

            # Handle case where we might have a single label for all points
            if len(special_points_labels) == 1 and len(special_points) > 1:
                special_points_labels = special_points_labels * len(special_points)

            # Group points by label for legend consistency
            unique_labels = []
            label_to_points = {}

            for i, label in enumerate(special_points_labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    label_to_points[label] = {
                        "points": [i],
                        "marker": special_points_markers[i],
                        "size": special_points_sizes[i],
                        "color": special_points_colors[i],
                    }
                else:
                    label_to_points[label]["points"].append(i)

            # Plot each group of special points
            for label in unique_labels:
                info = label_to_points[label]
                points_indices = info["points"]

                # For the first point of each label type, add a label for the legend
                if num_axes == 1:
                    ax.scatter(
                        special_points[points_indices, 0],
                        np.zeros_like(special_points[points_indices, 0]),
                        marker=info["marker"],
                        s=info["size"],
                        color=info["color"],
                        label=label,
                    )
                elif num_axes == 2:
                    ax.scatter(
                        special_points[points_indices, 0],
                        special_points[points_indices, 1],
                        marker=info["marker"],
                        s=info["size"],
                        color=info["color"],
                        label=label,
                    )
                else:
                    ax.scatter(
                        special_points[points_indices, 0],
                        special_points[points_indices, 1],
                        special_points[points_indices, 2],
                        marker=info["marker"],
                        s=info["size"],
                        color=info["color"],
                        label=label,
                    )

        ax.set_xlabel(xlabel)
        if num_axes > 1:
            ax.set_ylabel(ylabel)
        if num_axes == 3:
            ax.set_zlabel(zlabel)
        if title:
            ax.set_title(title)

        # Remove ticks
        if remove_ticks:
            if num_axes == 1:
                ax.set_yticks([])
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if num_axes == 3:
                ax.set_zticks([])
                ax.set_box_aspect([1, 1, 1], zoom=0.8)

        ax.set_aspect("auto")

        # Dynamically calculate number of columns in legend to fit horizontally
        max_legend_width = figsize[0] * fig.dpi  # width in pixels
        avg_label_width = 300  # estimated average label width in pixels
        n_cols = max(1, math.floor(max_legend_width / avg_label_width))

        # Place the legend underneath the plot and center it horizontally.
        # loc='upper center': Use the legend's top-center as the anchor point.
        # bbox_to_anchor=(0.5, -0.15): Place this anchor point at the horizontal center (0.5)
        #                              and slightly below (-0.15) the axes area. Adjust the
        #                              Y value (-0.15) if needed for more/less spacing.
        #                              A smaller negative number (e.g., -0.20) gives more space.
        ax.legend(
            loc="upper center",  # Anchor point on the legend box
            bbox_to_anchor=(0.5, -0.15),  # Position relative to the axes box
            ncol=n_cols,
            frameon=False,
            fontsize="small",
        )

        # Adjust layout slightly to prevent title/legend overlap if needed, although
        # bbox_inches='tight' in savefig often handles this.
        # Adjust rect as needed [left, bottom, right, top]
        # fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save the figure if filepath is provided
        if filepath is not None:
            filepath = Path(filepath)
            extension = filepath.suffix[1:]  # Removes the leading dot
            plt.savefig(filepath, format=extension, dpi=dpi, bbox_inches="tight")
            logger.success(f"Figure saved to {filepath}")

        # Show the plot if specified
        if show:
            plt.show()

        plt.close()


def apply_pca(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
    *,
    standardize_first: bool = True,
) -> np.ndarray:
    """Apply PCA to reduce the dimensionality of the given high-dimensional array."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    pipeline = PCA(n_components=n_components, random_state=random_state)
    if standardize_first:
        pipeline = make_pipeline(StandardScaler(), pipeline)

    return pipeline.fit_transform(X), pipeline


def apply_tsne(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    perplexity: float | None = None,
    learning_rate: float | None = None,
    random_state: int = 42,
    *,
    use_tsnecuda: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Apply t-SNE to reduce the dimensionality of the given high-dimensional array.

    The following parameters are set based on the number of samples in the data and the
    recommendations cited from various sources:
    * Uncertain Choices in Method Comparisons: An Illustration with t-SNE and UMAP (2023)
      See: https://epub.ub.uni-muenchen.de/107259/1/BA_Weber_Philipp.pdf
    * New guidance for using t-SNE: Alternative defaults, hyperparameter selection automation,
      and comparative evaluation (2022)
      See: https://www.sciencedirect.com/science/article/pii/S2468502X22000201
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    n = X.shape[0]
    learning_rate = max(200, int(n / 12)) if learning_rate is None else learning_rate
    perplexity = max(30, int(n / 100)) if perplexity is None else perplexity

    info = f"Applying t-SNE (perplexity={perplexity}, learning_rate={learning_rate}) using "

    if use_tsnecuda and TSNE_CUDA_INSTALLED:
        if verbose:
            info += "tsnecuda... "
            logger.info(info)

        start_time = time()
        tsne_embedded = TSNE_CUDA(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_seed=random_state,
        ).fit_transform(X)
    else:
        if verbose:
            info += "opentsne... "
            logger.info(info)

        start_time = time()
        tsne_embedded = TSNE(
            n_components=n_components,
            n_jobs=32,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=random_state,
        ).fit_transform(X)

    if verbose:
        logger.info(f"Done in {time() - start_time:.2f} seconds.")

    return tsne_embedded


def apply_lda(
    X: np.ndarray | pd.DataFrame,
    indices: list[np.ndarray | pd.Index],
    n_components: int = 2,
    *,
    standardize_first: bool = True,
) -> np.ndarray:
    """Apply LDA to reduce the dimensionality of the given high-dimensional array.

    Args:
        X (np.ndarray | pd.DataFrame): The input data to be transformed.
        indices (list[np.ndarray | pd.Index]): A list of boolean arrays or index arrays indicating
            different groups/clusters.
        n_components (int, optional): Number of components to keep. Default is 2.
        standardize_first (bool, optional): Whether to standardize the data before applying LDA.
            Default is True.

    Returns:
        np.ndarray: The transformed data.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Create labels from indices
    labels = np.zeros(X.shape[0], dtype=int)
    for i, idx in enumerate(indices, 1):
        labels[idx] = i

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    pipeline = LinearDiscriminantAnalysis(n_components=n_components)
    if standardize_first:
        pipeline = make_pipeline(StandardScaler(), pipeline)

    return pipeline.fit_transform(X, y)


def apply_ica(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
    *,
    standardize_first: bool = True,
) -> np.ndarray:
    """Apply ICA to reduce the dimensionality of the given high-dimensional array."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if standardize_first:
        pipeline = make_pipeline(
            StandardScaler(),
            FastICA(n_components=n_components, random_state=random_state),
        )
    else:
        pipeline = FastICA(n_components=n_components, random_state=random_state)
    return pipeline.fit_transform(X)


if UMAP_INSTALLED:

    def apply_umap(
        X: np.ndarray | pd.DataFrame,
        n_components: int = 2,
        standardize_first: bool = True,  # noqa: FBT001, FBT002
        **kwargs: dict,
    ) -> tuple[np.ndarray, UMAP]:
        """Apply UMAP to reduce the dimensionality of the given high-dimensional array.

        Args:
            X : np.ndarray | pd.DataFrame
                The input data to be transformed.
            n_components : int, optional
                Number of components to keep. Default is 2.
            standardize_first : bool, optional
                Whether to standardize the data before applying UMAP. Default is True.
            **kwargs : dict
                Additional keyword arguments for UMAP.

        Returns:
            (tuple[np.ndarray, UMAP]): The transformed data and the UMAP model.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        umap = UMAP(
            n_components=n_components,
            **kwargs,
        )

        if standardize_first:
            pipeline = make_pipeline(StandardScaler(), umap)
        else:
            pipeline = umap

        return pipeline.fit_transform(X), pipeline


DIMENSIONALITY_REDUCTION_METHODS = {
    "pca": apply_pca,
    "tsne": apply_tsne,
    "lda": apply_lda,
    "ica": apply_ica,
}

if UMAP_INSTALLED:
    DIMENSIONALITY_REDUCTION_METHODS["umap"] = apply_umap


def visualize(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    method: Literal["pca", "tsne", "lda", "ica", "umap"] = "pca",
    n_components: int = 2,
    max_labels: int | None = None,
    dim_reduction_kwargs: dict | None = None,
    labels: list[str] | np.ndarray | None = None,
    style: str | None = None,
    cmap: str | None = None,
    special_points: list | None = None,
    special_points_labels: list | None = None,
    special_points_markers: list | None = None,
    special_points_sizes: list | None = None,
    special_points_colors: list | None = None,
    **kwargs,
) -> None:
    """Apply dimensionality reduction and plot the results.

    Colors the specified groups in indices.

    Args:
        X (np.ndarray | pd.DataFrame): The input data to be transformed and visualized.
        y (np.ndarray | pd.Series): The target labels for the data.
        method (str, optional): The method to use for dimensionality reduction.
            Can be 'pca', 'tsne', 'ica', 'umap', or 'lda' (default: 'pca')
        n_components (int, optional): Number of dimensions to reduce to (default: 2)
        max_labels (int, optional): Maximum number of unique labels to display individually.
            Less frequent labels will be grouped into an "Others" category. Default: None
            (show all).
        dim_reduction_kwargs (dict, optional): Additional keyword arguments for the
            dimensionality reduction method, by default None
        labels (list of str, optional): List of labels for each group (default: None)
        style (str, optional): The style to use for plotting. This is used to get different colors
            from the default matplotlib color cycle (default: None)
        cmap (str | None, optional): The colormap to use for the coloring of labels. If this is
            provided, then `style` is ignored for this coloring. By default None.
        special_points (list, optional): Special points to plot, by default None
        special_points_labels (list, optional): Labels for special points, by default None
        special_points_markers (list, optional): Markers for special points, by default None
        special_points_sizes (list, optional): Sizes for special points, by default None
        special_points_colors (list, optional): Colors for special points, by default None
        **kwargs: Keyword arguments passed to `scatter_plot()`.

    Raises:
        ValueError: If the method is not one of the supported methods.
    """
    method = method.lower()

    if X.shape[1] < n_components:
        logger.warning(
            f"X.shape[1] = {X.shape[1]} is less than n_components = {n_components}.",
        )
        return

    if X.shape[1] == n_components:
        logger.warning(f"X.shape[1] = {X.shape[1]} is equal to n_components = {n_components}.")

    if method not in DIMENSIONALITY_REDUCTION_METHODS:
        raise ValueError(
            f"Unknown method: {method}. Use one of: "
            f"{', '.join(DIMENSIONALITY_REDUCTION_METHODS.keys())}",
        )

    if not dim_reduction_kwargs:
        dim_reduction_kwargs = {}

    # Common parameters for all dimensionality reduction methods
    dim_reduction_kwargs = {
        "n_components": n_components,
        **dim_reduction_kwargs,
    }

    # Construct the boolean indices
    indices = [y == i for i in np.unique(y)]

    # Add it to the kwargs of the dimensionality reduction method if it is LDA, since the
    # algorithm requires it
    if method == "lda":
        dim_reduction_kwargs["indices"] = indices

    special_points_reduced = None

    # For PCA, we can fit on X and then transform special points
    if method == "pca":
        X_reduced, pipeline = DIMENSIONALITY_REDUCTION_METHODS[method](X, **dim_reduction_kwargs)

        # Transform special points if provided
        if special_points is not None:
            special_points = np.array(special_points)
            special_points_reduced = pipeline.transform(special_points)

    # UMAP has a transform method
    elif method == "umap":
        X_reduced, dim_reduction_model = DIMENSIONALITY_REDUCTION_METHODS[method](
            X, **dim_reduction_kwargs
        )

        if special_points is not None:
            special_points = np.array(special_points)
            special_points_reduced = dim_reduction_model.transform(special_points)

    # Stacking approach for other methods
    else:
        # If special points are provided, we need to apply the same transformations
        if special_points is not None:
            special_points = np.array(special_points)
            # Combine the data and the special points
            combined_data = np.vstack([X, special_points])

            # Apply dimensionality reduction to the combined data
            combined_reduced = DIMENSIONALITY_REDUCTION_METHODS[method](
                combined_data,
                **dim_reduction_kwargs,
            )

            # Split the results back
            X_reduced = combined_reduced[: X.shape[0]]
            special_points_reduced = combined_reduced[X.shape[0] :]
        else:
            X_reduced = DIMENSIONALITY_REDUCTION_METHODS[method](X, **dim_reduction_kwargs)

    # Prepare Labels and Indices for Plotting (with max_labels logic)
    unique_classes_orig, counts_orig = np.unique(y, return_counts=True)
    num_unique_classes = len(unique_classes_orig)

    plot_indices = []
    plot_labels = []

    if max_labels is not None and max_labels > 0 and num_unique_classes > max_labels:
        logger.info(
            f"Number of classes ({num_unique_classes}) exceeds max_labels ({max_labels}). "
            "Grouping least frequent classes.",
        )

        # Sort classes by frequency (descending)
        sorted_indices = np.argsort(counts_orig)[::-1]
        top_classes = unique_classes_orig[sorted_indices[:max_labels]]
        other_classes = unique_classes_orig[sorted_indices[max_labels:]]

        # Create a mapping from original class value to provided label name if available
        label_map = {}
        if labels is not None and len(labels) == num_unique_classes:
            # Assuming the order of `labels` corresponds to the sorted order of
            # `unique_classes_orig`
            label_map = dict(zip(unique_classes_orig, labels, strict=True))
        else:
            if labels is not None:
                logger.warning(
                    "Provided `labels` length mismatch. Using default class values as labels.",
                )
            # Use string representation of class values as default labels
            label_map = {cls_val: str(cls_val) for cls_val in unique_classes_orig}

        # Indices and Labels for top classes
        for cls_val in top_classes:
            plot_indices.append(np.where(y == cls_val)[0])
            plot_labels.append(label_map[cls_val])  # Get the corresponding label name

        # Indices and Label for "Others"
        other_indices = np.where(np.isin(y, other_classes))[0]
        if len(other_indices) > 0:
            plot_indices.append(other_indices)
            plot_labels.append("Others")  # Add the "Others" label

    else:  # No grouping needed or requested
        # Use all unique classes
        unique_classes_to_plot = unique_classes_orig

        # Create label map as above
        label_map = {}
        if labels is not None and len(labels) == num_unique_classes:
            label_map = dict(zip(unique_classes_orig, labels, strict=True))
        else:
            if (
                labels is not None and max_labels is None
            ):  # Only warn if labels were provided but mismatch
                logger.warning(
                    "Provided `labels` length mismatch. Using default class values as labels.",
                )
            label_map = {cls_val: str(cls_val) for cls_val in unique_classes_orig}

        for cls_val in unique_classes_to_plot:
            plot_indices.append(np.where(y == cls_val)[0])
            plot_labels.append(label_map[cls_val])

    if method == "pca":
        axis_label = "$PC_{}$"
    elif method == "lda":
        axis_label = "$LD_{}$"
    elif method == "ica":
        axis_label = "$IC_{}$"
    elif method == "umap":
        axis_label = "$UMAP_{}$"
    else:
        axis_label = "$Dim_{}$"

    scatter_plot(
        X_reduced,
        indices=plot_indices,
        labels=plot_labels,
        style=style,
        cmap=cmap,
        title=(
            f"{method.upper() if method != 'tsne' else 't-SNE'} {n_components}D Visualization"
            if kwargs.get("title") is None
            else kwargs.pop("title")
        ),
        xlabel=axis_label.format(1),
        ylabel=axis_label.format(2),
        zlabel=axis_label.format(3),
        special_points=special_points_reduced,
        special_points_labels=special_points_labels,
        special_points_markers=special_points_markers,
        special_points_sizes=special_points_sizes,
        special_points_colors=special_points_colors,
        **kwargs,
    )
