"""Utilities for generating hyperparameter grid search configurations."""

import itertools
from collections.abc import Generator
from copy import deepcopy
from typing import Any

from pretrain_braindecode_models.config import logger

# --- Define keys that are LISTS BY DEFAULT ---
# For these keys, a grid search is only triggered if their value is a LIST OF LISTS.
# The path is a tuple representing the nested keys.
DEFAULT_LIST_KEYS = {
    ("dataset_params", "save_params"),
    ("dataset_params", "eeg_channels"),
    ("dataset_params", "preprocessing_pipeline", "steps"),
    ("split_params", "specific_train_keys"),
    ("split_params", "specific_test_keys"),
    ("model", "model_kwargs", "mlp_hidden_dims"),
    ("model", "model_kwargs", "hidden_dims"),
    ("training_params", "optimizer_kwargs", "betas"),
    ("training_params", "plot_kwargs", "plt_style"),
    ("training_params", "plot_kwargs", "figsize"),
    ("training_params", "augmentation_params", "transforms"),
}


def _set_nested_value_by_dot_string(d: dict, dot_string: str, value: Any) -> None:  # noqa: ANN401
    """Set a value in a nested dictionary using a dot-separated string path."""
    keys = dot_string.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _set_nested_value(d: dict, path: tuple, value: Any) -> None:  # noqa: ANN401
    """Set a value in a nested dictionary using a path of keys."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _find_grid_keys(
    config: dict,
    path: tuple = (),
    *,
    verbose: bool = True,
) -> Generator[tuple[tuple, list], None, None]:
    """Recursively find keys in the config that are meant for grid search."""
    for key, value in config.items():
        current_path = (*path, key)
        if isinstance(value, dict):
            yield from _find_grid_keys(value, current_path, verbose=verbose)
        elif isinstance(value, list) and value:
            is_default_list = current_path in DEFAULT_LIST_KEYS

            # If it's a default list key, only treat it as a grid search
            # parameter if it's a list of lists.
            if is_default_list:
                if all(isinstance(i, list) for i in value):
                    if verbose:
                        logger.debug(
                            f"Found grid search key (list-of-lists): {'.'.join(current_path)}"
                        )
                    yield current_path, value
            # Otherwise, any list is a grid search parameter.
            else:
                if verbose:
                    logger.debug(f"Found grid search key: {'.'.join(current_path)}")
                yield current_path, value


def generate_grid_search_configs(
    original_config: dict[str, Any],
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Generate a list of specific experiment configs.

    From a single config that may contain lists for hyperparameter searching.

    Handles both standard Cartesian product grid search (from lists in the config)
    and synchronized parameter sweeps using a special `_linked_params` key.

    Args:
        original_config (dict[str, Any]): The configuration dictionary loaded from YAML.
        verbose (bool): Whether to log information about the grid search process.

    Returns:
        list[dict[str, Any]]: A list of configuration dictionaries, one for each experiment in
            the grid.
    """
    # --- Part 1: Handle synchronized parameters first if they exist ---
    linked_param_sets = original_config.get("_linked_params")
    if linked_param_sets:
        if not isinstance(linked_param_sets, list):
            logger.warning("'_linked_params' should be a list of dictionaries. Ignoring.")
            base_configs = [original_config]
        else:
            if verbose:
                logger.debug(
                    f"Found {len(linked_param_sets)} linked parameter sets. "
                    "Generating one config per set."
                )
            base_configs = []
            for linked_set in linked_param_sets:
                if not isinstance(linked_set, dict):
                    continue
                # Create a copy of the original config and apply the linked set overrides
                config_copy = deepcopy(original_config)
                for dot_path, value in linked_set.items():
                    _set_nested_value_by_dot_string(config_copy, dot_path, value)
                base_configs.append(config_copy)
    else:
        # If no linked params, the starting point is just the original config
        base_configs = [original_config]

    # Clean up the special key from all base configs
    for config in base_configs:
        config.pop("_linked_params", None)

    # --- Part 2: Apply standard grid search to EACH of the base configs ---
    final_configs = []
    for base_config in base_configs:
        grid_params = list(_find_grid_keys(base_config, verbose=verbose))

        if not grid_params:
            final_configs.append(base_config)
            continue

        param_paths = [p[0] for p in grid_params]
        param_values = [p[1] for p in grid_params]
        value_combinations = list(itertools.product(*param_values))

        if verbose:
            logger.debug(
                f"Expanding a base config with {len(grid_params)} standard grid search "
                f"parameters into {len(value_combinations)} variants."
            )

        for combination in value_combinations:
            config_copy = deepcopy(base_config)
            for path, value in zip(param_paths, combination, strict=True):
                _set_nested_value(config_copy, path, value)
            final_configs.append(config_copy)

    if verbose:
        logger.info(f"Generated a total of {len(final_configs)} experiment configurations.")

    return final_configs
