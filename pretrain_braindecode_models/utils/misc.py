# ruff: noqa: S101
"""Miscellaneous utility functions."""

from __future__ import annotations

import inspect
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import reduce
from gc import collect as garbage_collect
from typing import TYPE_CHECKING, Any, Literal, get_args, overload

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.cuda import empty_cache as cuda_empty_cache
from torch.cuda import mem_get_info

from pretrain_braindecode_models.config import logger
from pretrain_braindecode_models.utils.custom_types import (
    ArrayOrTensor,
    DimOrder,
    K,
    SupportsArithmetic,
    SupportsComparison,
    TrainingHistoryClassification,
)

if TYPE_CHECKING:
    from pathlib import Path


def calculate_consecutive_differences(
    values: list[SupportsArithmetic],
) -> list[SupportsArithmetic]:
    """Calculate the differences between consecutive values in a list.

    Args:
        values (list[SupportsArithmetic]): A list of numeric values

    Returns:
        list: A list of differences between consecutive values
    """
    if len(values) < 2:
        return []

    differences = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        differences.append(diff)

    return differences


def is_strictly_increasing(
    values: list[SupportsComparison] | np.ndarray,
    *,
    from_zero: bool = False,
    warn: bool = False,
    raise_error: bool = False,
) -> bool:
    """Check if a list of values is strictly increasing.

    A single value or empty list is considered strictly increasing.

    Args:
        values (list[SupportsComparison]): A list of values to check
        from_zero (bool): If True, the first value is considered as 0 for comparison
        warn (bool): Whether to print a warning if the list is not strictly increasing
        raise_error (bool): Whether to raise an error if the list is not strictly increasing

    Returns:
        bool: True if each value is greater than the previous one, False otherwise

    Raises:
        ValueError:
            - If `raise_error` is True and the list is not strictly increasing
            - If the first value is not 0 when `from_zero` is True
    """
    if len(values) < 2:
        return True  # A single value or empty list is considered strictly increasing

    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            if warn:
                logger.warning(
                    f"List is not strictly increasing at index {i}: "
                    f"{values[i - 1]} >= {values[i]}",
                )
            if raise_error:
                raise ValueError(
                    f"List is not strictly increasing at index {i}: "
                    f"{values[i - 1]} >= {values[i]}",
                )
            return False

    if from_zero and values[0] != 0:
        if warn:
            logger.warning(
                f"List is not strictly increasing from zero: first value {values[0]} is not 0.",
            )
        if raise_error:
            raise ValueError(
                f"List is not strictly increasing from zero: first value {values[0]} is not 0.",
            )
        return False

    return True


def get_nested_value(data: dict[K, Any], key_path: list[K], default: Any = None) -> Any:  # noqa: ANN401
    """Safely retrieve a value from a nested dictionary using a list of keys.

    Args:
        data (dict[K, Any]): The nested dictionary to search.
        key_path (list[K]): A list of keys representing the path to the desired value.
                            e.g., ['kwargs', 'norm_layer']
        default (Any): The value to return if any key in the path is not found.
                       Defaults to None.

    Returns:
        The value found at the specified path, or the default value if not found.
    """
    try:
        # reduce applies the get operator sequentially through the key_path
        # return reduce(operator.getitem, key_path, data)
        return reduce(lambda d, k: d[k], key_path, data)
    except (KeyError, TypeError):
        # Handles cases where a key doesn't exist or an intermediate value is not a dict
        return default


def get_nested_value_by_dot_string(
    data: dict[str, Any],
    dot_string: str,
    default: Any = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Safely retrieve a value from a nested dictionary using a dot-separated string.

    Args:
        data (dict[str, Any]): The nested dictionary to search.
        dot_string (str): A string representing the path, e.g., 'model.model_kwargs.name'.
        default (Any): The value to return if the path is not found.

    Returns:
        Any: The value found at the specified path, or the default value.
    """
    keys = dot_string.split(".")
    return get_nested_value(data, keys, default)


def metadata_is_filtered_out(
    metadata: dict, filters: dict[tuple[K, ...], set], *, metadata_name: str | None = ""
) -> bool:
    """Check if the given `metadata` dictionary matches the given `filters`.

    For example, if `filters` is:
    ```
    {
        ("model_class",): {"MLP", "LSTM"},
        ("dataset", "eeg_window_seconds"): {3.0},
        ("dataset", "flame_window_seconds"): {1.0},
        ("kwargs", "norm_layer"): {"batch"},
    }
    ```
    The function will return `True` if the given metadata does not match any of these conditions,
    that is, the given `metadata` must fulfill all of the conditions for this function to return
    `False`.

    Args:
        metadata (dict): The metadata dictionary to check.
        filters (dict[tuple[K, ...], set]): A dictionary where keys are tuples representing path
            to the desired value, and values are sets of allowed values.
        metadata_name (str | None): Optional name for the metadata, used for logging.

    Returns:
        bool: True if metadata matches all filters, False otherwise.
    """
    filter_out = False
    for key_path, allowed_values in filters.items():
        # Use the same helper to get the value for filtering
        value = get_nested_value(metadata, list(key_path))

        # Special handling to simplify model names for filtering
        if key_path == ("model_class",):
            if "MLP" in str(value):
                value = "MLP"
            if "LSTM" in str(value):
                value = "LSTM"

        if value is None or value not in allowed_values:
            prefix = f"Skipping {metadata_name}: " if metadata_name else ""
            logger.debug(
                f"{prefix}Value '{value}' for key path {key_path} not in allowed "
                f"set {allowed_values}."
            )
            filter_out = True
            break  # No need to check other filters for this file

    return filter_out


def pop_from_sets(data: dict[Any, set]) -> dict[Any, Any]:
    """Check that all sets as values in a dictionary `data` have only one value and pop it.

    Args:
        data (dict[Hashable, set]): Dictionary where values are sets

    Returns:
        (dict[Hashable, Any]): Dictionary with the same keys but with single values, instead of the
            original set
    """
    result = {}
    for k, v in data.items():
        if not isinstance(v, set):
            logger.error(f"Value for {k} is not a set: {v}")
            continue

        if len(v) != 1:
            logger.error(
                f"Key {k} does not have a single value in its set: {v}. "
                f"Getting the first  (next(iter(v))), i.e., {next(iter(v))}",
            )
        result[k] = v.pop()  # Get the single value
    return result


def rename_key(d: dict, old_key: K, new_key: K) -> None:
    """Rename a key in a dictionary in-place.

    Args:
        d (dict): The dictionary to modify.
        old_key (K): The key to rename.
        new_key (K): The new key name.

    Returns:
        dict: The updated dictionary.
    """
    if old_key in d:
        d[new_key] = d.pop(old_key)
    else:
        logger.warning(f"Key '{old_key}' not found in dictionary. No changes made.")


def rename_keys(d: dict, mapping: dict[K, K]) -> None:
    """Rename multiple keys in a dictionary in-place based on a mapping.

    Args:
        d (dict): The dictionary to modify.
        mapping (dict[K, K]): A dictionary where keys are old key names
            and values are new key names.

    Returns:
        dict: The updated dictionary.
    """
    for old_key, new_key in mapping.items():
        rename_key(d, old_key, new_key)


def recursive_update(base_dict: dict, override_dict: Mapping) -> dict:
    """Recursively update a dictionary with values from another dictionary.

    Unlike `dict.update()`, this function merges nested dictionaries instead ofoverwriting them
    entirely.

    Args:
        base_dict (dict): The dictionary to be updated.
        override_dict (Mapping): The dictionary providing the new values.

    Returns:
        dict: The updated base_dict.
    """
    for key, value in override_dict.items():
        if isinstance(value, Mapping) and key in base_dict and isinstance(base_dict[key], dict):
            # If both are dicts, recurse
            base_dict[key] = recursive_update(base_dict[key], value)
        else:
            # Otherwise, overwrite the value
            base_dict[key] = value
    return base_dict


def delete_nested_keys(d: dict, keys_to_delete: Sequence[str | tuple[str, ...]]) -> None:
    """Recursively deletes keys from a nested dictionary **in-place**.

    Keys can be specified as a string for top-level keys or a tuple
    for a path to a nested key.

    Example:
    >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
    >>> keys_to_delete = ['a', ('b', 'c')]
    >>> delete_nested_keys(d, keys_to_delete)
    # d is now {'b': {'d': 3}, 'e': 4}
    """
    for key_path in keys_to_delete:
        current_dict = d
        if isinstance(key_path, str):
            # Top-level key
            current_dict.pop(key_path, None)
        elif isinstance(key_path, tuple) and len(key_path) > 1:
            # Nested key: traverse to the parent dictionary
            try:
                for key in key_path[:-1]:
                    current_dict = current_dict[key]
                # Delete the final key from the parent
                if isinstance(current_dict, dict):
                    current_dict.pop(key_path[-1], None)
            except (KeyError, TypeError):
                # Path doesn't exist or not a dict, which is fine, just ignore.
                pass
        elif isinstance(key_path, tuple) and len(key_path) == 1:
            # A tuple with one element, treat as a top-level key
            d.pop(key_path[0], None)


def diff_dicts(dict1: Mapping[K, Any], dict2: Mapping[K, Any]) -> dict[str, Any]:
    """Recursively compare two dictionaries and returns a dictionary of their differences.

    The resulting dictionary will have three top-level keys:
    - 'added': Keys present in dict2 but not in dict1.
    - 'removed': Keys present in dict1 but not in dict2.
    - 'changed': Keys present in both dicts but with different values. For nested
                 dictionaries, this will contain a recursive diff.

    Args:
        dict1 (Mapping[str, Any]): The first dictionary (considered the 'old' version).
        dict2 (Mapping[str, Any]): The second dictionary (considered the 'new' version).

    Returns:
        (dict[str, Any]): A dictionary detailing the differences. Returns an empty dictionary if
        the input dictionaries are identical.
    """
    d1_keys = set(dict1.keys())
    d2_keys = set(dict2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)

    added = {key: dict2[key] for key in d2_keys - d1_keys}
    removed = {key: dict1[key] for key in d1_keys - d2_keys}
    changed = {}

    for key in intersect_keys:
        val1, val2 = dict1[key], dict2[key]
        if isinstance(val1, dict) and isinstance(val2, dict):
            nested_diff = diff_dicts(val1, val2)
            if nested_diff:
                changed[key] = nested_diff
        # Compare these lists as sets to ignore order
        elif key in {"train_keys", "test_keys"}:
            if set(val1) != set(val2):
                changed[key] = {"old": val1, "new": val2}
        elif isinstance(val1, str) and isinstance(val2, str):
            if val1.strip() != val2.strip():
                changed[key] = {"old": val1.strip(), "new": val2.strip()}
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if tuple(val1) != tuple(val2):
                changed[key] = {"old": val1, "new": val2}
        elif key in {"input_shape", "output_shape", "criterion", "scaler_class"}:
            if str(val1) != str(val2):
                changed[key] = {"old": val1, "new": val2}
        elif val1 != val2:
            changed[key] = {"old": val1, "new": val2}

    differences = {}
    if added:
        differences["added"] = added
    if removed:
        differences["removed"] = removed
    if changed:
        differences["changed"] = changed

    return differences


def pretty_print_diff(diff: dict[str, Any], indent_level: int = 0, indent_str: str = "  ") -> None:
    """Print the output of diff_dicts in a color-coded, human-readable format.

    - Green for added keys/values.
    - Red for removed keys/values.
    - Yellow for changed values.

    Args:
        diff: The difference dictionary produced by `diff_dicts`.
        indent_level: The current indentation level for recursive calls.
        indent_str: The string to use for each level of indentation.
    """
    # ANSI color codes
    C_GREEN = "\033[92m"
    C_RED = "\033[91m"
    C_YELLOW = "\033[93m"
    C_RESET = "\033[0m"

    prefix = indent_str * indent_level

    if "removed" in diff:
        for key, value in diff["removed"].items():
            print(f"{prefix}{C_RED}- {key}: {value}{C_RESET}")

    if "added" in diff:
        for key, value in diff["added"].items():
            print(f"{prefix}{C_GREEN}+ {key}: {value}{C_RESET}")

    if "changed" in diff:
        for key, value in diff["changed"].items():
            if isinstance(value, dict) and ("old" not in value and "new" not in value):
                # This is a nested diff
                print(f"{prefix}{C_YELLOW}~ {key}:{C_RESET}")
                pretty_print_diff(value, indent_level + 1)
            else:
                # This is a direct value change
                print(
                    f"{prefix}{C_YELLOW}~ {key}:{C_RESET}\n"
                    f"{prefix}{indent_str}{C_RED}- Old: {value['old']}{C_RESET}\n"
                    f"{prefix}{indent_str}{C_GREEN}+ New: {value['new']}{C_RESET}"
                )


def make_json_serializable(
    data: Any,  # noqa: ANN401
    *,
    ignore_non_serializable: bool = False,
    warn: bool = False,
) -> Any:  # noqa: ANN401
    """Make the given `data` JSON serializable by recursively converting all keys and values.

    Args:
        data (Any): The input data to make JSON serializable.
        ignore_non_serializable (bool, optional): If True, non-serializable values are ignored.
        warn (bool, optional): If True, warnings are issued for non-serializable values.

    Returns:
        Any: JSON serializable version of the input data.
    """

    def _process_dict(data: dict, *, ignore_non_serializable: bool, warn: bool) -> dict:
        result = {}
        for key, value in data.items():
            try:
                result[str(key)] = make_json_serializable(
                    value,
                    ignore_non_serializable=ignore_non_serializable,
                    warn=warn,
                )
            except TypeError as e:
                if ignore_non_serializable:
                    if warn:
                        logger.warning(
                            f"Non-serializable value found for key '{key}'. Ignoring it. "
                            f"Error: {e}",
                        )
                    continue
                raise
        return result

    def _process_list(data: list, *, ignore_non_serializable: bool, warn: bool) -> list:
        return [
            make_json_serializable(
                item,
                ignore_non_serializable=ignore_non_serializable,
                warn=warn,
            )
            for item in data
        ]

    def _process_value(value: Any, *, ignore_non_serializable: bool, warn: bool) -> Any:  # noqa: ANN401
        if isinstance(value, np.ndarray):
            return _process_list(
                value.tolist(),  # type: ignore[reportArgumentType]
                ignore_non_serializable=ignore_non_serializable,
                warn=warn,
            )

        if isinstance(value, (int, float, str, bool, type(None))):
            return value

        try:
            return str(value)  # Try to convert to string if it's not a basic type
        except TypeError as e:
            if ignore_non_serializable:
                if warn:
                    logger.warning(
                        f"Non-serializable value found: {value}. Ignoring it. Error: {e}",
                    )
                return None
            raise

    if isinstance(data, dict):
        return _process_dict(data, ignore_non_serializable=ignore_non_serializable, warn=warn)

    if isinstance(data, list):
        return _process_list(data, ignore_non_serializable=ignore_non_serializable, warn=warn)

    return _process_value(data, ignore_non_serializable=ignore_non_serializable, warn=warn)


def make_hashable(obj: Any) -> Any:  # noqa: ANN401
    """Recursively converts a dictionary to a hashable type (frozenset of items)."""
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, list):
        return tuple(make_hashable(v) for v in obj)
    return obj


def dict_to_X_y_labels(  # noqa: N802
    data: dict,
    data_key: str | None = None,
    max_labels: int | None = None,
    other_label: str = "Others",
    *,
    transpose_data: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Prepare data from a dictionary for visualization or ML input.

    Extracts data arrays associated with labels, optionally transposes them,
    concatenates them into a single feature matrix (X), creates a corresponding
    label vector (y), handles limiting the number of unique labels by grouping
    less frequent ones, and encodes the final labels numerically.

    Args:
        data (dict[str, dict[str, np.ndarray]] | np.ndarray]): Dictionary where keys are the
            initial string labels and values contain the data. Values can be NumPy arrays directly
            or dictionaries holding the data array under the key specified by `data_key`.
        data_key (str | None, optional): If the values in `data` are dictionaries, this key
            specifies where the actual NumPy data array is stored. If None, the values themselves
            are assumed to be the data arrays. Default is "data".
        transpose_data (bool, optional): If True, transpose the extracted data arrays before
            concatenation. This is useful if the original data is shaped (features, samples)
            and you need (samples, features). Default is True.
        max_labels (int or None, optional): Maximum number of unique labels to keep distinct. If
            the number of unique labels in `data` exceeds this value, only the `max_labels` most
            frequent labels are kept, and the rest are grouped under the label specified by
            `other_label`. If None, all unique labels are kept. Default is None.
        other_label (str, optional): The label name to use for the grouped less frequent classes
            when `max_labels` is active. Default is "Others".

    Returns:
        (tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]):
            - X : np.ndarray
                The concatenated feature matrix, typically shape (total_samples, n_features).
            - y_encoded : np.ndarray
                The numerically encoded label vector, shape (total_samples,).
            - class_names : np.ndarray
                An array containing the string names of the classes corresponding
                to the numerical encoding in `y_encoded` (e.g., index 0 maps to
                `class_names[0]`). Includes `other_label` if grouping occurred.
            - label_encoder : LabelEncoder (optional)
                The fitted scikit-learn LabelEncoder instance, only returned if
                `return_encoder` is True.

    Raises:
        ValueError
            If `data` is empty, if data arrays have inconsistent feature dimensions after potential
            transposition, or if data extraction fails.
        TypeError
            If extracted data is not a NumPy array or Pandas DataFrame.
    """
    if not data:
        raise ValueError("Input 'data' dictionary cannot be empty.")

    X_list = []
    y_list = []
    expected_features = None
    processed_labels_cache = {}  # Cache label processing if max_labels is used

    # --- First pass (optional but good for max_labels): Get label counts ---
    if max_labels is not None and max_labels > 0:
        temp_label_counts: Counter = Counter()
        for label, value in data.items():
            if data_key:
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Expected dict value for key '{label}' when data_key is specified, "
                        f"got {type(value)}",
                    )
                data_array = value[data_key]
            else:
                data_array = value

            if isinstance(data_array, pd.DataFrame):
                data_array = data_array.to_numpy()
            elif not isinstance(data_array, np.ndarray):
                raise TypeError(
                    f"Data for label '{label}' must be a NumPy array or Pandas DataFrame, got "
                    f"{type(data_array)}",
                )

            # Determine sample count based on transpose flag
            n_samples = data_array.shape[1] if transpose_data else data_array.shape[0]
            temp_label_counts[label] += n_samples

        if len(temp_label_counts) > max_labels:
            # Identify top labels and the set of 'other' labels
            most_common_labels = {
                label for label, count in temp_label_counts.most_common(max_labels)
            }
            all_labels = set(temp_label_counts.keys())
            other_labels_set = all_labels - most_common_labels
            logger.info(
                f"Grouping {len(other_labels_set)} labels into '{other_label}':{other_labels_set}",
            )
            # Pre-calculate which original label maps to what final label
            for label in all_labels:
                processed_labels_cache[label] = (
                    label if label in most_common_labels else other_label
                )
        else:
            # No grouping needed, reset max_labels logic effectively
            max_labels = None
            logger.debug("Number of labels does not exceed max_labels. No grouping needed.")

    # --- Second pass: Process data and create final lists ---
    for label, value in data.items():
        try:
            # Extract data array
            if data_key:
                # Type check done in first pass if max_labels was used, else check now
                if not isinstance(value, dict) and max_labels is None:
                    raise TypeError(
                        f"Expected dict value for key '{label}' when data_key is specified, "
                        f"got {type(value)}",
                    )
                data_array = value[data_key]
            else:
                data_array = value

            if isinstance(data_array, pd.DataFrame):
                data_array = data_array.to_numpy()
            elif not isinstance(data_array, np.ndarray) and max_labels is None:
                # Type check done in first pass if max_labels was used, else check now
                raise TypeError(
                    f"Data for label '{label}' must be a NumPy array or Pandas DataFrame, "
                    f"got {type(data_array)}",
                )

            # Transpose if needed
            if transpose_data:
                data_array = data_array.T  # Now shape is (samples, features)

            # Check feature consistency
            n_samples, n_features = data_array.shape
            if expected_features is None:
                expected_features = n_features
            elif n_features != expected_features:
                raise ValueError(
                    f"Inconsistent number of features for label '{label}'. "
                    f"Expected {expected_features}, found {n_features}."
                )

            # Determine final label (original or 'Others')
            current_label = (
                processed_labels_cache.get(label, label) if max_labels is not None else label
            )

            # Append data and labels
            X_list.append(data_array)
            y_list.extend([current_label] * n_samples)
        except Exception as e:
            logger.error(f"Unexpected error processing label '{label}': {e}")
            raise  # Re-raise other unexpected errors

    if not X_list:
        # This might happen if data was not empty but contained no valid data
        raise ValueError("No valid data found in 'data' to process.")

    # --- Concatenate and Encode ---
    X = np.concatenate(X_list, axis=0)
    y_str = np.array(y_list)  # Array of string labels (including 'Others')

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)
    class_names = label_encoder.classes_  # These are the final unique string labels

    logger.info(f"Prepared data: X shape {X.shape}, y shape {y_encoded.shape}")  # type: ignore[reportAttributeAccessIssue]
    logger.info(f"Encoded classes: {class_names.tolist()}")

    return X, y_encoded, class_names, label_encoder  # type: ignore[reportReturnType]


def describe_dict_contents(
    data: dict,
    indent: int = 0,
    max_keys: int = 3,
    *,
    is_last: bool = False,  # noqa: ARG001
    initial_indent: bool = True,
    return_string: bool = False,
) -> str | None:
    """Recursively print keys and informative descriptions of their values from a dictionary.

    Handles various data types, providing details like length for lists and shape for numpy arrays.
    If `max_keys` is `None`, all keys are printed. Otherwise, only up to `max_keys` keys are shown
    with an ellipsis indicating more keys exist.

    Args:
        data (dict): The dictionary to describe.
        indent (int, optional): The current indentation level for pretty printing. Defaults to 0.
        max_keys (int, optional): Maximum number of keys to print before truncating
            with an ellipsis. If `None`, all keys are printed. Defaults to 3.
        is_last (bool, optional): Whether this is the last item in the current level of recursion.
            If `True`, no comma is printed after the last item. Defaults to `False`.
        initial_indent (bool, optional): If `True`, prints the opening brace `{` without
            indentation. This is useful for the initial call to start the output cleanly.
            Defaults to `True`.
        return_string (bool, optional): If `True`, returns the formatted string instead of
            printing it. Defaults to `False`.

    Returns:
        str | None: The formatted string representation if `return_string` is `True`,
                   otherwise `None`.
    """
    indent_str = "    " * indent  # Use 4 spaces instead of tabs
    output_lines = []

    # Print opening brace
    if indent == 0 and initial_indent:
        output_lines.append("{")
    elif indent == 0:
        output_lines.append(indent_str + "{")
    else:
        output_lines.append("{")  # For nested dicts, opening brace is on the same line as the key

    total_keys = len(data)
    keys_processed = 0

    for i, (key, value) in enumerate(data.items(), start=1):
        # Handle different value types
        if isinstance(value, dict):
            line = f'{indent_str}    "{key}": {{'
            output_lines.append(line)

            nested_result = describe_dict_contents(
                value,
                indent + 1,
                max_keys,
                is_last=(i == total_keys),
                initial_indent=False,
                return_string=True,
            )

            if nested_result is None:
                logger.error(f"Failed to describe nested dictionary for key '{key}'.")
                continue

            # Skip the first line (opening brace) from nested result since we already added it
            nested_lines = nested_result.split("\n")[1:]
            output_lines.extend(nested_lines)

            if i < total_keys:
                output_lines.append(f"{indent_str}    " + "},")  # Closing brace with comma
            else:
                output_lines.append(f"{indent_str}    " + "}")  # No comma for the last item
        elif isinstance(value, list):
            item_types = {type(item).__name__ for item in value}
            line = (
                f'{indent_str}    "{key}": '
                f"List (Length: {len(value)}) containing {', '.join(item_types)} items"
                + ("," if i < total_keys else "")
            )
            output_lines.append(line)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):  # More robust numpy check
            line = (
                f'{indent_str}    "{key}": '
                f"Numpy array with shape {value.shape} and dtype {value.dtype}"
                + ("," if i < total_keys else "")
            )
            output_lines.append(line)
        elif isinstance(value, (str, int, float, bool)):
            formatted_value = f'"{value}"' if isinstance(value, str) else str(value)
            line = f'{indent_str}    "{key}": {formatted_value}' + ("," if i < total_keys else "")
            output_lines.append(line)
        else:
            line = f'{indent_str}    "{key}": {type(value).__name__}' + (
                "," if i < total_keys else ""
            )
            output_lines.append(line)

        keys_processed += 1
        if max_keys is not None and keys_processed >= max_keys:
            output_lines.append(indent_str + "    ...")  # Proper indentation for ellipsis
            break

    # Closing brace for the dictionary
    if indent == 0:
        output_lines.append("}")
    # For nested dicts, closing brace is handled in the parent's loop

    result = "\n".join(output_lines)

    if return_string:
        return result

    print(result)
    return None


def find_and_group_values_by_key(
    data: Mapping[K, Any], target_keys: list[K]
) -> Mapping[K, list[Any]]:
    """Traverse a nested structure to find all values for target keys and group them.

    This function is a general-purpose utility for extracting and organizing
    data from complex, nested objects.

    Args:
        data (Mapping[K, Any]): The dictionary or list to search.
        target_keys (list[K]): A list of key names to search for.

    Returns:
        (Mapping[K, list[Any]]): A dictionary where each key is one of the
        target_keys and its value is a list of all values found for that key.
    """
    target_keys_set = set(target_keys)
    # Use defaultdict for clean appending to lists
    found_values_by_key = defaultdict(list)

    def _recursive_finder(current_data: Mapping[K, Any] | list[Any]) -> None:
        """Inner recursive helper to build up the found_values_by_key dict."""
        if isinstance(current_data, dict):
            for key, value in current_data.items():
                if key in target_keys_set:
                    found_values_by_key[key].append(value)
                # Continue traversal into the value
                _recursive_finder(value)
        elif isinstance(current_data, list):
            for item in current_data:
                _recursive_finder(item)

    _recursive_finder(data)
    return dict(found_values_by_key)


@overload
def extract_unique_shapes(
    values: list[Any], *, count_occurrences: Literal[True]
) -> dict[tuple[int, ...], int]: ...


@overload
def extract_unique_shapes(
    values: list[Any], *, count_occurrences: Literal[False] = False
) -> set[tuple[int, ...]]: ...


def extract_unique_shapes(
    values: list[Any], *, count_occurrences: bool = False
) -> set[tuple[int, ...]] | dict[tuple[int, ...], int]:
    """Process a list of objects to extract their shapes.

    Can either return a set of unique shapes or a dictionary counting the
    occurrences of each shape.

    Args:
        values (list[Any]): A list of items, some of which may have a .shape attribute.
        count_occurrences (bool, optional): If True, returns a dict mapping each
            shape to its count. If False, returns a set of unique shapes.
            Defaults to False.

    Returns:
        (set[tuple[int, ...]] | dict[tuple[int, ...], int]): A set of unique shapes
        or a dictionary of shape counts.
    """
    # First, extract all valid shape tuples from the list of values
    shapes = [
        value.shape
        for value in values
        if hasattr(value, "shape") and isinstance(getattr(value, "shape", None), tuple)
    ]

    if count_occurrences:
        # Counter is a highly optimized dict subclass for counting hashable objects
        return dict(Counter(shapes))

    # Return a set of unique shapes
    return set(shapes)


@overload
def get_unique_shapes_for_keys(
    data: Mapping[K, Any],
    target_keys: list[K],
    *,
    group_by_key: Literal[True],
    count_occurrences: Literal[True],
) -> dict[K, dict[tuple[int, ...], int]]: ...


@overload
def get_unique_shapes_for_keys(
    data: Mapping[K, Any],
    target_keys: list[K],
    *,
    group_by_key: Literal[True],
    count_occurrences: Literal[False] = False,
) -> dict[K, set[tuple[int, ...]]]: ...


@overload
def get_unique_shapes_for_keys(
    data: Mapping[K, Any],
    target_keys: list[K],
    *,
    group_by_key: Literal[False] = False,
    count_occurrences: Literal[True],
) -> dict[tuple[int, ...], int]: ...


@overload
def get_unique_shapes_for_keys(
    data: Mapping[K, Any],
    target_keys: list[K],
    *,
    group_by_key: Literal[False] = False,
    count_occurrences: Literal[False] = False,
) -> set[tuple[int, ...]]: ...


def get_unique_shapes_for_keys(
    data: Mapping[K, Any],
    target_keys: list[K],
    *,
    group_by_key: bool = False,
    count_occurrences: bool = False,
) -> (
    set[tuple[int, ...]]
    | dict[tuple[int, ...], int]
    | dict[K, set[tuple[int, ...]]]
    | dict[K, dict[tuple[int, ...], int]]
):
    """Find unique shapes or shape counts for keys within a nested dictionary.

    Args:
        data (dict[K, Any]): The nested dictionary to search.
        target_keys (list[K]): Keys whose values' shapes you want.
        group_by_key (bool, optional): If True, groups results by target key.
            Defaults to False.
        count_occurrences (bool, optional): If True, counts occurrences of each
            shape instead of just listing unique shapes. Defaults to False.

    Returns:
        The result, which can be one of four types depending on the parameters:
        - set: Unique shapes, combined from all keys.
        - dict: Shape counts, combined from all keys.
        - dict of sets: Unique shapes, grouped by key.
        - dict of dicts: Shape counts, grouped by key.
    """
    grouped_values = find_and_group_values_by_key(data, target_keys)

    if group_by_key:
        # By separating the logic, Mypy knows that for this entire branch,
        # the function returns a dict of dicts.
        if count_occurrences:
            return {
                key: extract_unique_shapes(values, count_occurrences=True)
                for key, values in grouped_values.items()
            }
        # And for this branch, it returns a dict of sets
        return {
            key: extract_unique_shapes(values, count_occurrences=False)
            for key, values in grouped_values.items()
        }

    all_values = [value for sublist in grouped_values.values() for value in sublist]

    # We must use an if/else here so Mypy knows the literal value being passed
    if count_occurrences:
        # In this branch, Mypy knows we are calling with count_occurrences=True,
        # which matches the Literal[True] overload.
        return extract_unique_shapes(all_values, count_occurrences=True)

    # In this branch, Mypy knows we are calling with count_occurrences=False,
    # which matches the Literal[False] overload.
    return extract_unique_shapes(all_values, count_occurrences=False)


def pretty_print_dict(
    d: dict[str, Any],
    indent_level: int = 0,
    indent_str: str = "    ",  # 4 spaces for indentation
    bullet_str: str = "- ",
    *,
    nums_as_pct: bool = False,
) -> str:
    """Recursively pretty-prints a dictionary into a bulleted, indented string.

    Args:
        d: The dictionary to print.
        indent_level: The current level of indentation (used for recursion).
        indent_str: The string to use for one level of indentation.
        bullet_str: The string to use as a bullet point for each key.
        nums_as_pct: If True, formats numeric values as percentages.

    Returns:
        A formatted string representation of the dictionary.
    """
    pretty_string = ""
    # Get the current indentation prefix
    prefix = indent_str * indent_level

    for i, (key, value) in enumerate(d.items()):
        # Add a newline before each item except the very first one at the top level
        if indent_level > 0 or i > 0:
            pretty_string += "\n"

        # Format the key part of the line
        line_start = f"{prefix}{bullet_str}{key}:"

        if isinstance(value, dict):
            # If the value is another dictionary, recurse
            pretty_string += f"{line_start}"
            pretty_string += pretty_print_dict(
                value,
                indent_level + 1,  # Increase indentation for the nested dict
                indent_str,
                bullet_str,
                nums_as_pct=nums_as_pct,
            )
        else:
            # If the value is not a dictionary, format it
            formatted_value = value
            if nums_as_pct and isinstance(value, (int, float)):
                # Format number as percentage
                formatted_value = f"{float(value) * 100:.2f}%"

            pretty_string += f"{line_start} {formatted_value}"

    return pretty_string


def logger_per_line(
    log: str | list[Any],
    level: Literal["info", "debug", "warning", "error"] = "info",
    title: str | None = None,
) -> None:
    """Log each line of a string separately, useful for multi-line logs."""
    if title:
        logger.info(title)

    log = log.strip() if isinstance(log, str) else "\n".join(str(item) for item in log)

    # Print one log line per newline
    for line in log.split("\n"):
        if level == "info":
            logger.info(line)
        elif level == "debug":
            logger.debug(line)
        elif level == "warning":
            logger.warning(line)
        elif level == "error":
            logger.error(line)
        else:
            raise ValueError(f"Unsupported log level: {level}")


def compare_nested_dicts(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    key_map: dict[str, str],
    *,
    check_shape: bool = True,
    check_content: bool = True,
    on_missing_key: Literal["warn", "skip", "raise"] = "warn",
) -> bool:
    """Compare numpy arrays within two nested dictionaries based on a key mapping.

    This function iterates through a 2-level nested dictionary structure
    (e.g., dict[rec_id][expr_name]) and compares specified numpy arrays
    between the two dictionaries.

    Args:
        dict1 (dict[str, Any]): The first dictionary to compare.
        dict2 (dict[str, Any]): The second dictionary to compare.
        key_map (dict[str, str]): A mapping of keys from dict1 to dict2. A dictionary mapping keys
            from dict1 to keys in dict2, e.g.: `{'eeg_data': 'data', 'timestamps': 'timestamps'}`
        check_shape (bool): If True, asserts that the shapes of the arrays are identical.
        check_content (bool): If True, asserts that the content of the arrays are identical using
            `np.array_equal`.
        on_missing_key (Literal["warn", "skip", "raise"]): What to do if a key is not found in one
            of the dictionaries.
            - 'warn': Log a warning and continue.
            - 'skip': Silently continue to the next item.
            - 'raise': Raise a KeyError.

    Returns:
        True if all checks passed, False otherwise.
    """
    if on_missing_key not in ["warn", "skip", "raise"]:
        raise ValueError("on_missing_key must be 'warn', 'skip', or 'raise'.")

    all_checks_passed = True

    # Iterate through the structure of the first dictionary
    for level1_key, level1_value in dict1.items():
        if not isinstance(level1_value, dict):
            continue

        for level2_key, level2_value in level1_value.items():
            if not isinstance(level2_value, dict):
                continue

            # --- Key Existence Checks ---
            try:
                dict2_level2_value = dict2[level1_key][level2_key]
            except KeyError as e:
                msg = (
                    f"Key path '{level1_key} -> {level2_key}' not found in the second dictionary."
                )
                if on_missing_key == "warn":
                    logger.warning(msg)
                elif on_missing_key == "raise":
                    raise KeyError(msg) from e
                all_checks_passed = False
                continue  # Skip to the next expression/item

            # --- Array Comparison Loop ---
            for key1, key2 in key_map.items():
                array1 = level2_value.get(key1)
                array2 = dict2_level2_value.get(key2)

                # Handle None values
                if array1 is None or array2 is None:
                    if array1 is None and array2 is None:
                        # Both are None, this is fine
                        continue
                    msg = (
                        f"Mismatch: One array is None at '{level1_key}/{level2_key}'. "
                        f"Key1 ('{key1}') is None: {array1 is None}, "
                        f"Key2 ('{key2}') is None: {array2 is None}"
                    )
                    if on_missing_key == "warn":
                        logger.warning(msg)
                    elif on_missing_key == "raise":
                        raise ValueError(msg)
                    all_checks_passed = False
                    continue

                # --- Shape Check ---
                if check_shape and array1.shape != array2.shape:
                    msg = (
                        f"Shape mismatch for '{key1}'/'{key2}' at '{level1_key}/{level2_key}': "
                        f"{array1.shape} != {array2.shape}"
                    )
                    # For shape mismatches, we always assert or warn, as content check is invalid
                    try:
                        raise AssertionError(msg)
                    except AssertionError:
                        logger.error(msg)
                        all_checks_passed = False
                    # If shapes don't match, content check is meaningless, so we skip it
                    continue

                # --- Content Check ---
                if check_content:
                    try:
                        assert np.array_equal(array1, array2), (
                            f"Content mismatch for '{key1}'/'{key2}' "
                            f"at '{level1_key}/{level2_key}'. "
                            f"Shapes are {array1.shape} and {array2.shape}."
                        )
                    except AssertionError as e:
                        logger.error(e)
                        all_checks_passed = False

    if all_checks_passed:
        logger.success(f"Comparison finished. All checks passed: {all_checks_passed}")
    else:
        logger.error(f"Comparison finished. Some checks failed: {all_checks_passed}")

    return all_checks_passed


def reorder_dict_keys(original_dict: dict[K, Any], desired_order: list[K]) -> dict[K, Any]:
    """Reorders the keys of a dictionary according to a specified list.

    Keys from `desired_order` that are present in the dictionary will appear first,
    in that order. Any remaining keys from the original dictionary will be appended
    afterwards, preserving their original relative order.

    Args:
        original_dict (dict[K, Any]): The dictionary to reorder.
        desired_order (list[K]): A list of keys specifying the desired order.

    Returns:
        (dict[K, Any]): A new dictionary with the reordered keys.
    """
    reordered_dict = {}
    original_keys = set(original_dict.keys())

    # 1. Add keys from the desired order first
    for key in desired_order:
        if key in original_keys:
            reordered_dict[key] = original_dict[key]
        else:
            logger.warning(
                f"Key '{key}' specified in desired order but not found in the dictionary."
            )

    # 2. Add the remaining keys in their original relative order
    for key, value in original_dict.items():
        if key not in reordered_dict:
            reordered_dict[key] = value

    return reordered_dict


def get_init_args(obj_instance: Any) -> dict[str, Any]:  # noqa: ANN401
    """Introspects a model instance and retrieves the arguments used to initialize it.

    This function is useful for saving model hyperparameters for later re-instantiation.
    It assumes that the names of the `__init__` arguments are stored as attributes
    on the instance with the exact same names.

    Args:
        obj_instance (Any): An instance of a class (e.g., a PyTorch model).

    Returns:
        (dict[str, Any]): A dictionary where keys are the `__init__` argument names and values are
        the corresponding values from the instance.

    Raises:
        AttributeError: If an `__init__` argument is not found as an attribute on the class
            instance.
    """
    cls = obj_instance.__class__
    try:
        init_signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        logger.warning(
            f"Could not get __init__ signature for {cls.__name__}. Returning empty dict."
        )
        return {}

    init_args = {}
    kwargs_param_name = None

    # --- Identify regular parameters and the **kwargs parameter name ---
    arg_names = []
    for param in init_signature.parameters.values():
        if param.name in ("self", "args"):
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # Found the **kwargs parameter, e.g., 'kwargs'
            kwargs_param_name = param.name
        else:
            arg_names.append(param.name)

    # --- Retrieve values for regular arguments ---
    for name in arg_names:
        try:
            value = getattr(obj_instance, name)
            init_args[name] = value
        except AttributeError:
            logger.warning(
                f"Argument '{name}' from {cls.__name__}.__init__ was not found "
                f"as an attribute. It will be missing from the returned dict. "
                f"Ensure `self.{name} = {name}` is in the class `__init__`."
            )

    # --- Retrieve and unpack the kwargs dictionary ---
    if kwargs_param_name:
        try:
            # Get the dictionary of keyword arguments from the instance
            kwargs_dict = getattr(obj_instance, kwargs_param_name)
            if isinstance(kwargs_dict, dict):
                # Merge the contents of the kwargs dict into our main args dict
                init_args.update(kwargs_dict)
            else:
                logger.warning(
                    f"Attribute '{kwargs_param_name}' for **kwargs on {cls.__name__} "
                    f"is not a dictionary (found {type(kwargs_dict)}). Cannot unpack."
                )
        except AttributeError:
            logger.warning(
                f"The `**{kwargs_param_name}` parameter was in the signature for "
                f"{cls.__name__}, but no matching attribute `self.{kwargs_param_name}` "
                f"was found on the instance."
            )

    logger.info(f"Dynamically retrieved init args for {cls.__name__}: {list(init_args.keys())}")
    return init_args


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    # Using np_random_generator
    # to avoid: Replace legacy `np.random.seed` call with `np.random.Generator` Ruff NPY002
    # np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def np_random_generator(seed: int = 42) -> np.random.Generator:
    """Create a NumPy random generator with a fixed seed."""
    return np.random.default_rng(seed)


@overload
def reorder_dims(
    data: np.ndarray, current_order: DimOrder, target_order: DimOrder
) -> np.ndarray: ...


@overload
def reorder_dims(
    data: torch.Tensor, current_order: DimOrder, target_order: DimOrder
) -> torch.Tensor: ...


def reorder_dims(
    data: ArrayOrTensor, current_order: DimOrder, target_order: DimOrder
) -> np.ndarray | torch.Tensor:
    """Reorder the dimensions of a NumPy array or PyTorch Tensor.

    Handles common 2D and 3D layouts like NCT (Batch, Channels, Time) and
    NTC (Batch, Time, Channels).

    Args:
        data (ArrayOrTensor): The input array or tensor to reorder.
        current_order (DimOrder): A string representing the current dimension order.
        target_order (DimOrder): A string representing the desired dimension order.

    Returns:
        ArrayOrTensor: The reordered array or tensor.

    Raises:
        ValueError: If the number of dimensions in the data does not match the
                    length of the order strings, or if the order is unsupported.
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Input data must be a NumPy array or PyTorch Tensor, got {type(data)}")

    if current_order == target_order:
        # No change needed, return original data
        return data

    if len(current_order) != len(target_order):
        raise ValueError(
            f"Dimension order strings must have the same length. "
            f"Got '{current_order}' and '{target_order}'."
        )

    if data.ndim != len(current_order):
        raise ValueError(
            f"Input data has {data.ndim} dimensions, but current_order "
            f"'{current_order}' implies {len(current_order)} dimensions."
        )
    # Validate that the order strings are supported
    supported_orders = get_args(DimOrder)
    if current_order not in supported_orders or target_order not in supported_orders:
        raise ValueError(f"Unsupported order. Use one of {supported_orders}.")

    # Validate that target order doesn't contain a dimension not present in current order
    if not set(target_order).issubset(set(current_order)):
        raise ValueError(
            f"Target order contains dimensions not present in current order. "
            f"Current: '{current_order}', Target: '{target_order}'. "
        )

        # Validate that current order doesn't contain a dimension not present in target order
    if not set(current_order).issubset(set(target_order)):
        raise ValueError(
            f"Current order contains dimensions not present in target order. "
            f"Current: '{current_order}', Target: '{target_order}'. "
        )

    # Create a mapping from character to its index in the current layout
    # e.g., for "NCT", source_map will be {'N': 0, 'C': 1, 'T': 2}
    source_map = {dim_char: i for i, dim_char in enumerate(current_order)}

    # Determine the permutation tuple for the transpose operation
    # e.g., to go from "NCT" to "NTC", target_order is "NTC".
    # We look up 'N', 'T', 'C' in source_map to get their original positions.
    # permutation -> (source_map['N'], source_map['T'], source_map['C']) -> (0, 2, 1)
    try:
        permutation = tuple(source_map[dim_char] for dim_char in target_order)
    except KeyError as e:
        raise ValueError(
            f"Target order '{target_order}' contains a dimension not present in "
            f"current order '{current_order}': {e}"
        ) from e

    # Perform the transposition
    if isinstance(data, np.ndarray):
        return data.transpose(permutation)

    # It must be a tensor at this point
    return data.permute(permutation)


@overload
def as_tensors_on_device(
    item: ArrayOrTensor,
    /,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
) -> torch.Tensor: ...


@overload
def as_tensors_on_device(
    item1: ArrayOrTensor,
    item2: ArrayOrTensor,
    /,
    *items: ArrayOrTensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, ...]: ...


def as_tensors_on_device(
    *items: ArrayOrTensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    verbose: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Ensure all input items are PyTorch tensors of a specified dtype on the target device.

    Accepts numpy arrays or existing PyTorch tensors.

    Args:
        *items (np.ndarray | torch.Tensor): One or more numpy arrays, PyTorch tensors.
        device (str | torch.device): The target torch device (e.g., 'cuda', 'cpu',
            torch.device('cuda:0')).
        dtype (torch.dtype | None): The desired data type for the tensors. If None,
            the original dtype is preserved as much as possible (default: None).
        verbose (bool): If True, logs information about conversions and moves.

    Returns:
        (torch.Tensor | tuple[torch.Tensor, ...]): If a single item was passed, returns the
            processed tensor. If multiple items were passed, returns a tuple of processed
            tensors in the same order they were received.

    Raises:
        TypeError: If an item cannot be converted to a tensor (and is not None).
        ValueError: If no items are provided or if an item is None.
    """
    if not items:
        raise ValueError("No items provided")

    processed_items: list[torch.Tensor] = []
    target_device = torch.device(device)  # Ensure it's a torch.device object

    for i, item in enumerate(items):
        if item is None:
            raise ValueError(
                f"Item {i} is None. Cannot convert None to tensor. "
                "If you want to allow None, please handle it before calling this function.",
            )

        tensor = None
        # Convert to Tensor if necessary
        if isinstance(item, np.ndarray):
            if verbose:
                logger.debug(f"Item {i} (np.ndarray) converting to tensor.")
            try:
                tensor = torch.from_numpy(item)  # More efficient for numpy
            except Exception as e:
                # Fallback for safety, though from_numpy is preferred
                logger.warning(
                    f"torch.from_numpy failed for item {i}, falling back to torch.tensor. "
                    f"Error: {e}",
                )
                try:
                    tensor = torch.tensor(item)
                except Exception as inner_e:
                    raise TypeError(
                        f"Failed to convert item {i} (type {type(item)}) to tensor: {inner_e}",
                    ) from inner_e

        elif isinstance(item, torch.Tensor):
            tensor = item  # Already a tensor
        else:
            # Try converting other types if needed, or raise error
            try:
                if verbose:
                    logger.debug(
                        f"Item {i} (type {type(item)}) attempting conversion via torch.tensor.",
                    )
                tensor = torch.tensor(item)
            except Exception as e:
                raise TypeError(
                    f"Item {i} is not a np.ndarray or torch.Tensor, and conversion failed: {e}",
                ) from e

        # 2. Ensure correct dtype and device
        current_dtype = tensor.dtype
        current_device = tensor.device

        if dtype is not None and current_dtype != dtype:
            if verbose:
                logger.debug(f"Item {i}: Changing dtype from {current_dtype} to {dtype}.")
            tensor = tensor.to(dtype=dtype)

        if current_device != target_device:
            if verbose:
                logger.debug(f"Item {i}: Moving from {current_device} to {target_device}.")
            tensor = tensor.to(device=target_device)

        if tensor is None:
            raise ValueError(
                f"Item {i} was processed to None. Cannot return None as a tensor.",
            )

        processed_items.append(tensor)

    # Return single item or tuple based on input count
    if len(processed_items) == 1:
        if processed_items[0] is None:
            raise ValueError(
                "Only one item was provided, but it is None. Cannot return None as a tensor.",
            )

        return processed_items[0]

    return tuple(processed_items)


@overload
def as_numpy_arrays(
    item: ArrayOrTensor,
    /,
    *,
    dtype: np.dtype | None = None,
    verbose: bool = False,
) -> np.ndarray: ...


@overload
def as_numpy_arrays(
    item1: ArrayOrTensor,
    item2: ArrayOrTensor,
    /,
    *items: ArrayOrTensor,
    dtype: np.dtype | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, ...]: ...


def as_numpy_arrays(
    *items: ArrayOrTensor,
    dtype: np.dtype | None = None,
    verbose: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Ensure all input items are NumPy arrays of a specified dtype.

    Accepts NumPy arrays or existing PyTorch tensors. If a PyTorch tensor is on a CUDA device,
    it is moved to the CPU before conversion.

    Args:
        *items (ArrayOrTensor): One or more NumPy arrays or PyTorch tensors.
        dtype (np.dtype | None): The desired NumPy data type for the arrays. If None, the original
               dtype is preserved as much as possible. Defaults to None.
        verbose (bool): If True, logs information about conversions and type changes.

    Returns:
        (np.ndarray | tuple[np.ndarray, ...]): If a single item was passed, returns the processed
            NumPy array. If multiple items were passed, returns a tuple of processed NumPy arrays.

    Raises:
        ValueError: If no items are provided.
        TypeError: If an item is not a NumPy array or a PyTorch Tensor.
    """
    if not items:
        raise ValueError("No items provided")

    processed_items: list[np.ndarray] = []

    for i, item in enumerate(items):
        if item is None:
            raise ValueError(f"Item {i} is None. Cannot convert None to a NumPy array.")

        np_array = None

        # 1. Convert to NumPy array if necessary
        if isinstance(item, torch.Tensor):
            if verbose:
                logger.debug(f"Item {i} (torch.Tensor) converting to np.ndarray.")
            # A tensor must be on the CPU to be converted to NumPy
            np_array = item.cpu().numpy()
        elif isinstance(item, np.ndarray):
            np_array = item  # Already a NumPy array
        else:
            raise TypeError(
                f"Item {i} is not a np.ndarray or torch.Tensor, but type {type(item)}."
            )

        # 2. Ensure correct dtype if specified
        if dtype is not None and np_array.dtype != dtype:
            if verbose:
                logger.debug(f"Item {i}: Changing dtype from {np_array.dtype} to {dtype}.")
            try:
                np_array = np_array.astype(dtype)
            except Exception as e:
                raise TypeError(f"Failed to convert item {i} to dtype {dtype}: {e}") from e

        processed_items.append(np_array)

    # Return single item or tuple based on input count
    if len(processed_items) == 1:
        return processed_items[0]

    return tuple(processed_items)


def check_for_nan_or_inf(*items: tuple[ArrayOrTensor, str] | ArrayOrTensor) -> None:
    """Check input items for NaN or Inf values and log warnings if found."""
    if not items:
        raise ValueError("No items provided")

    for i, item_ in enumerate(items):
        if item_ is None:
            raise ValueError(f"Item {i} is None")

        if isinstance(item_, tuple) and len(item_) == 2 and isinstance(item_[1], str):
            item, name = item_
            prefix = f" in '{name}'"
        else:
            item = item_
            prefix = ""

        if isinstance(item, np.ndarray):
            # Check if the numpy array contains numeric data before checking for NaN/Inf
            if np.issubdtype(item.dtype, np.number):
                if np.any(np.isnan(item)):
                    logger.warning(f"NaN values found{prefix} input data.")
                if np.any(np.isinf(item)):
                    logger.warning(f"Inf values found{prefix} input data.")
            else:
                logger.debug(f"Skipping NaN/Inf check for non-numeric numpy array{prefix}.")
        elif isinstance(item, torch.Tensor):
            # Check if the torch tensor contains numeric data before checking for NaN/Inf
            # All PyTorch numeric dtypes can be checked against is_floating_point or is_complex
            if item.is_floating_point() or item.is_complex():
                if torch.any(torch.isnan(item)):
                    logger.warning(f"NaN values found{prefix} input data.")
                if torch.any(torch.isinf(item)):
                    logger.warning(f"Inf values found{prefix} input data.")
            else:
                logger.debug(f"Skipping NaN/Inf check for non-numeric torch tensor{prefix}.")
        else:
            # For types that are not np.ndarray or torch.Tensor,
            # log a message, and ignore.
            logger.debug(f"Skipping NaN/Inf check for unsupported type {type(item)}{prefix}.")


def clean_cache(*, verbose: bool = True) -> None:
    """Clean the GPU memory cache."""
    garbage_collect()
    cuda_empty_cache()
    mem_info = mem_get_info()
    if verbose:
        logger.info(
            "Freeing GPU Memory. Free: %d MB\tTotal: %d MB"
            % (mem_info[0] // 1024**2, mem_info[1] // 1024**2),
        )


def convert_skorch_history_to_plotting_format(
    history_df: pd.DataFrame,
    *,
    best_metric: str = "valid_bal_acc",
    metric_prefix_map: dict[str, str] | None = None,
    ignore_cols: set[str] | None = {"train_batch_count", "valid_batch_count"},
) -> tuple[dict[str, list[float]], list[str], int]:
    """Convert a skorch history DataFrame to the format expected by `plot_history`.

    See `pretrain_braindecode_models/modeling/plotting.py:plot_history`.

    Args:
        history_df (pd.DataFrame): DataFrame created from a skorch EEGClassifier's history.
        best_metric (str): The name of the validation metric to use for determining the best epoch
            (e.g., 'valid_bal_acc', 'valid_f1').
        metric_prefix_map (dict[str, str] | None): An optional dictionary to rename metrics for
            plotting. e.g., {"bal_acc": "Balanced Accuracy", "f1": "F1 Score"}
        ignore_cols (list[str] | None): Columns to ignore when converting. Defaults to
            ["batch_count"].

    Returns:
        (tuple[dict[str, list[float]], list[str], int]): A tuple containing:
        - The formatted history dictionary.
        - A list of the base metric names found (e.g., ['bal_acc', 'f1']).
        - The index of the best epoch based on the specified metric.
    """
    history_dict: dict[str, list[float]] = {
        "train_loss": history_df["train_loss"].tolist(),
        "val_loss": history_df["valid_loss"].tolist(),
    }

    metrics_found = []

    # Dynamically find all train/validation metric pairs
    for col in history_df.columns:
        if ignore_cols and col in ignore_cols:
            continue

        if (
            col.startswith("train_")
            and col not in ["train_loss", "train_loss_best"]
            and not col.endswith("_best")
        ):
            base_metric_name = col.replace("train_", "")
            valid_col_name = (
                f"val_{base_metric_name}"  # skorch uses 'valid_' not 'val_' for EpochScoring
            )

            # skorch uses 'valid_' for EpochScoring but might use 'val_' for others
            if valid_col_name not in history_df.columns:
                valid_col_name = f"valid_{base_metric_name}"

            if valid_col_name in history_df.columns:
                # Use the custom name from the map, or the base name
                plot_metric_name = (
                    metric_prefix_map.get(base_metric_name, base_metric_name)
                    if metric_prefix_map
                    else base_metric_name
                )

                # Add to history dict for plotting
                history_dict[f"train_{plot_metric_name}"] = history_df[col].tolist()
                history_dict[f"val_{plot_metric_name}"] = history_df[valid_col_name].tolist()
                metrics_found.append(plot_metric_name)

    # Find the best epoch based on the specified validation metric
    if best_metric not in history_df.columns:
        raise ValueError(f"Best metric '{best_metric}' not found in history DataFrame columns.")

    # For accuracy/F1, higher is better. For loss, lower is better.
    # We assume all custom metrics follow "higher is better".
    best_epoch_idx = history_df[best_metric].idxmax()

    return history_dict, metrics_found, int(best_epoch_idx)


def clean_lightning_metric(metric: str) -> str:
    """Clean a PyTorch Lightning metric name for plotting and/or good display.

    This makes:
    - `"train_loss_epoch"` -> `"train_loss"`
    - `"val_loss_epoch"` -> `"val_loss"`
    - ... and leaves other metrics unchanged.

    Args:
        metric (str): The metric name to clean.

    Returns:
        str: The cleaned metric name.
    """
    if metric in {"train_loss", "val_loss"}:
        clean_key = metric
    # Remove suffix so "train_loss_epoch" -> "train_loss"
    elif metric.endswith("_epoch"):
        clean_key = metric.replace("_epoch", "")
    # elif metric.startswith("val_"):
    #     clean_key = "val_" + metric.replace("val_", "").replace("_", " ").title()
    # elif metric.startswith("train_"):
    #     # This now correctly handles train_balanced_accuracy, etc.
    #     clean_key = "train_" + metric.replace("train_", "").replace("_", " ").title()
    else:
        clean_key = metric

    return clean_key


def process_lightning_logs_to_history(
    log_dir: Path,
) -> TrainingHistoryClassification:
    """Read a PyTorch Lightning CSVLogger's metrics.csv file.

    - It converts it into a TrainingHistoryClassification obj compatible with plotting functions.
    - It handles aggregating step-level metrics into epoch-level metrics by taking the last
    recorded step for each epoch.
    - It correctly handles cases where train and validation metrics are logged on separate rows
    for the same epoch/step.

    Args:
        log_dir (Path): The directory where the Lightning logger saved its files
            (e.g., '.../lightning_logs/version_0/').

    Returns:
        (TrainingHistoryClassification): A TrainingHistoryClassification object
            containing the processed training history.
    """
    metrics_path = log_dir / "metrics.csv"
    if not metrics_path.exists():
        logger.warning(f"Metrics file not found at {metrics_path}. Cannot process history.")
        return TrainingHistoryClassification()

    df = pd.read_csv(metrics_path)

    # Identify all columns that represent aggregated epoch-level metrics.
    # Exclude step-level loss as we only want the epoch summary.
    epoch_metric_cols = [
        col
        for col in df.columns
        if col.endswith("_epoch")
        or col.startswith("val_")
        or (col.startswith("train_") and col != "train_loss_step")
    ]

    # 1. Forward-fill the values within each epoch. This propagates the last known
    #    metric value down, so the final row for each epoch contains all metrics.
    df[epoch_metric_cols] = df.groupby("epoch")[epoch_metric_cols].ffill()

    # 2. Now, we can simply take the last row for each epoch, which will have the
    #    complete set of metrics.
    epoch_summary = df.groupby("epoch").tail(1)

    history = defaultdict(list)
    for _, row in epoch_summary.iterrows():
        for col_name, value in row.items():
            col_name = str(col_name)  # noqa: PLW2901
            if pd.notna(value) and col_name not in ["epoch", "step"]:
                # --- Key Cleaning ---
                clean_key = clean_lightning_metric(col_name)
                history[clean_key].append(value)

    return TrainingHistoryClassification(**history)  # type: ignore[reportArgumentType, arg-type]


def get_n_classes(*ys: ArrayOrTensor) -> int:
    """Get the number of unique classes across multiple label arrays.

    Args:
        *ys (ArrayOrTensor): One or more arrays or tensors containing class labels.

    Returns:
        int: The number of unique classes across all provided arrays.
    """
    # Make sure all inputs are numpy arrays
    ys_np = [y.numpy() if isinstance(y, torch.Tensor) else y for y in ys]

    # If the given ys are numerical
    if all(isinstance(y, np.ndarray) and y.dtype.kind in "iu" for y in ys_np):
        # Use np.concatenate to flatten all arrays and find unique classes
        return len(np.unique(np.concatenate(ys_np)))

    unique_classes = set()
    for y in ys_np:
        if isinstance(y, np.ndarray):
            unique_classes.update(set(np.unique(y).tolist()))
        else:
            raise TypeError(
                f"Expected numpy arrays, but got {type(y)}. "
                "Ensure all inputs are numpy arrays of class labels.",
            )
    return len(unique_classes)
