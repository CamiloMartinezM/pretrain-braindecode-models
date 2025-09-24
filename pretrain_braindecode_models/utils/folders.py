"""Processes paths and folders."""

import math
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, overload

from pretrain_braindecode_models.config import logger
from pretrain_braindecode_models.modeling.models import extract_model_name
from pretrain_braindecode_models.utils.custom_types import RunInfo, RunStatus
from pretrain_braindecode_models.utils.loading import load_json, load_yaml
from pretrain_braindecode_models.utils.misc import (
    delete_nested_keys,
    diff_dicts,
    get_nested_value,
    logger_per_line,
    pretty_print_diff,
)


def find_files(
    folder: str | Path,
    *,
    extensions: str | list[str] | None = None,
    name_patterns: str | list[str] | None = None,
    case_sensitive: bool = False,
    followlinks: bool = True,
) -> list[str]:
    """Recursively finds all files matching given extensions and/or name patterns.

    This function traverses the directory tree and returns a sorted list of absolute file paths
    that satisfy the search criteria. At least one of `extensions` or `name_patterns` must be
    provided.

    Args:
        folder (str | Path): Path to the folder to search.
        extensions (str | list[str] | None, optional): A single file extension (e.g., ".txt")
            or a list of extensions to search for (e.g., [".jpg", ".png"]).
        name_patterns (str | list[str] | None, optional): A single glob-style pattern
            (e.g., "report_*") or a list of patterns to match against the full filename.
            The `*` wildcard is supported.
        case_sensitive (bool, optional): If True, the extension match is
            case-sensitive. Defaults to False (e.g., ".jpg" will match ".JPG").
        followlinks (bool, optional): If True, the search will traverse into directories that are
            symbolic links. Defaults to True.

    Returns:
        list[str]: A sorted list of absolute file paths matching the criteria.
    """
    if extensions is None and name_patterns is None:
        raise ValueError("At least one of `extensions` or `name_patterns` must be provided.")

    folder_path = Path(folder).resolve()
    found_files = []

    # --- Normalize Extensions ---
    ext_set = set()
    if extensions:
        ext_list = [extensions] if isinstance(extensions, str) else extensions
        for ext in ext_list:
            normalized_ext = f".{ext.lstrip('.')}"
            ext_set.add(normalized_ext if case_sensitive else normalized_ext.lower())

    # --- Normalize Name Patterns ---
    pattern_list = []
    if name_patterns:
        pattern_list = [name_patterns] if isinstance(name_patterns, str) else name_patterns
        if not case_sensitive:
            pattern_list = [p.lower() for p in pattern_list]

    # --- Walk the Directory Tree using os.walk for symlink control ---
    for dirpath, _, filenames in os.walk(str(folder_path), followlinks=followlinks):
        for filename in filenames:
            # Create a full Path object for matching and storage
            f = Path(dirpath) / filename

            # Prepare file attributes for comparison
            file_name_to_check = f.name if case_sensitive else f.name.lower()
            file_suffix_to_check = f.suffix if case_sensitive else f.suffix.lower()

            # --- Apply Filters ---
            passes_ext_filter = not ext_set or file_suffix_to_check in ext_set

            passes_name_filter = not pattern_list or any(
                # Use pathlib's match for glob patterns
                Path(file_name_to_check).match(p)
                for p in pattern_list
            )

            if passes_ext_filter and passes_name_filter:
                found_files.append(str(f.resolve()))

    return sorted(found_files)


def get_dir_last_modified_time(dir_path: Path) -> float:
    """Recursively find the most recent modification time of anything within a given directory.

    Args:
        dir_path (Path): The path to the directory.

    Returns:
        float: The modification time (as a float timestamp) of the most recently changed item, or
        0.0 if the directory does not exist.
    """
    if not dir_path.is_dir():
        return 0.0

    max_mtime = dir_path.stat().st_mtime

    for root, dirs, files in os.walk(dir_path):
        for d in dirs:
            max_mtime = max(max_mtime, (Path(root) / d).stat().st_mtime)
        for f in files:
            max_mtime = max(max_mtime, (Path(root) / f).stat().st_mtime)

    return max_mtime


def remove_contents(path: str | Path) -> None:
    """Remove all files and subdirectories in the specified `path`."""
    path = Path(path)
    for file_path in path.iterdir():
        if file_path.is_file() or file_path.is_symlink():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path)


def remove_folder(path: str | Path) -> None:
    """Remove the given `path` with all of its files and subdirectories."""
    path = Path(path)
    remove_contents(path)
    if path.exists():
        shutil.rmtree(path)


def link_folders(
    source_dir: Path,
    target_dir: Path,
    *,
    overwrite: bool = False,
    link_files: bool = True,
    ignore_hidden: bool = True,
) -> None:
    """Fuse contents of `source_dir` into `target_dir` using symbolic links.

    Creates symbolic links in a target directory pointing to the contents of a source directory,
    i.e., it iterates through all files and subdirectories in `source_dir` and creates a
    corresponding symbolic link for each in `target_dir`.

    Args:
        source_dir (Path): The path to the directory containing the original items (files and
            folders) that will be linked to.
        target_dir (Path): The path to the directory where the symbolic links will be created.
            This directory will be created if it does not exist.
        overwrite (bool): If True, any existing file or link in the target directory with the same
            name as an item in the source directory will be removed and replaced.
            Defaults to False.
        link_files (bool): If True, creates links for both files and directories.
            If False, only creates links for directories. Defaults to True.
        ignore_hidden (bool): If True, ignores files and directories that start with a dot ('.').
            Defaults to True.

    Raises:
        FileNotFoundError: If the `source_dir` does not exist.
        ValueError: If `source_dir` is not a directory.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")

    # Ensure the target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Linking contents of '{source_dir.name}' into '{target_dir.name}'...")

    # Iterate through all items (files and directories) in the source directory
    for source_item in source_dir.iterdir():
        if ignore_hidden and source_item.name.startswith("."):
            logger.debug(f"Ignoring hidden item: {source_item.name}")
            continue

        if not source_item.is_dir() and not link_files:
            logger.debug(f"Skipping file because `link_files` is False: {source_item.name}")
            continue

        # Define the path for the new symbolic link in the target directory
        link_path = target_dir / source_item.name

        # Check if a file/link with the same name already exists
        if link_path.exists() or link_path.is_symlink():
            if overwrite:
                logger.warning(f"'{link_path.name}' already exists in target. Overwriting.")
                if link_path.is_dir() and not link_path.is_symlink():
                    remove_folder(link_path)
                else:
                    # os.unlink works for both files and symlinks
                    link_path.unlink()
            else:
                logger.warning(
                    f"'{link_path.name}' already exists in target and overwrite is False. "
                    "Skipping."
                )
                continue

        # Create the symbolic link
        try:
            # os.symlink is often more reliable across platforms
            os.symlink(src=source_item.resolve(), dst=link_path)
            logger.success(f"  -> Created link: {link_path.name} -> {source_item.name}")
        except Exception as e:
            logger.error(
                f"Failed to create link for '{source_item.name}' -> '{link_path.name}': {e}"
            )


def format_size_bytes(size_bytes: int) -> str:
    """Convert bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = math.floor(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def calculate_dir_size(directory: Path, *, ignore: set[str] | None = None) -> int:
    """Calculate the total size of files in a directory.

    Args:
        directory (Path): The directory to calculate the size for.
        ignore (set[str] | None): A set of filenames to ignore.

    Returns:
        int: The total size of files in the directory.
    """
    total_size = 0
    if not directory.is_dir():
        return 0

    for item in directory.iterdir():
        if ignore and item.name in ignore:
            continue  # Skip the files we want to keep
        if item.is_file():
            total_size += item.stat().st_size
        elif item.is_dir():
            # Sum up the size of all files within the subdirectory
            total_size += sum(f.stat().st_size for f in item.glob("**/*") if f.is_file())
    return total_size


def new_run(
    runs_folder: Path,
    prefix: str = "run_",
    *,
    ignore_files: set[str] | None = None,
    ignore_dirs: set[str] | None = None,
    ignore_dirs_with_errors: bool = True,
) -> Path:
    """Create or reuse a run directory inside `runs_folder`.

    A new directory is created with an incremental number (e.g., prefix_0, prefix_1).
    However, if a previously created run directory is found to be "empty"
    (containing only items specified in `ignore_files` and `ignore_dirs`),
    that directory will be reused instead of creating a new one.

    Args:
        runs_folder (Path): Path to the folder where all run directories are stored.
        prefix (str): Prefix for the run directory name (e.g., "run_").
        ignore_files (set[str] | None): A set of filenames to ignore when checking if a directory
            is empty.
        ignore_dirs (set[str] | None): A set of directory names to ignore when checking if a
            directory is empty.
        ignore_dirs_with_errors (bool): Whether to ignore directories that contain error files.

    Returns:
        Path: The path to the newly created or reused run directory.
    """
    runs_folder.mkdir(parents=True, exist_ok=True)
    ignore_files = ignore_files or set()
    ignore_dirs = ignore_dirs or set()

    # Find all existing run directories and parse their indices
    run_indices = []
    for p in runs_folder.glob(f"{prefix}*"):
        if not p.is_dir():
            continue
        try:
            # Extract the numeric index from the directory name
            index_str = p.name.replace(prefix, "")
            if index_str.isdigit():
                run_indices.append(int(index_str))
        except ValueError:
            continue

    # Check existing runs to see if any can be reused
    for index in sorted(run_indices):
        run_dir = runs_folder / f"{prefix}{index}"

        # --- Lock File Check ---
        lock_file = run_dir / ".lock"
        if lock_file.exists():
            logger.debug(f"Directory {run_dir.name} is locked. Skipping.")
            continue

        # --- Content Checking Logic ---
        is_empty_for_reuse = True
        for item in run_dir.iterdir():
            # Ignore directories with files that are errors
            # (e.g., IndexError.json, FileNotFoundError.json, etc.)
            if ignore_dirs_with_errors and item.is_file() and "Error" in item.name:
                break
            if item.is_file() and item.name not in ignore_files:
                # Found a file that is NOT in the ignore list
                is_empty_for_reuse = False
                break
            if item.is_dir() and item.name not in ignore_dirs:
                # Found a directory that is NOT in the ignore list
                is_empty_for_reuse = False
                break

        if is_empty_for_reuse:
            # This directory is considered empty, so we can reuse it.
            # We should also clean it of the ignored items for a fresh start.
            logger.info(f"Reusing empty or ignorable run directory: {run_dir}")
            for item in run_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)  # Use shutil to remove directory and its contents
            return run_dir

    # If no existing directory could be reused, create a new one
    new_index = max(run_indices, default=-1) + 1
    new_run_dir = runs_folder / f"{prefix}{new_index}"
    new_run_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"Creating new run directory: {new_run_dir}")
    return new_run_dir


def check_run_exists(
    runs_folder: Path,
    new_metadata: dict[str, Any],
    ignore_keys: list[str | tuple[str, ...]],
    *,
    debug: bool = False,
) -> Path | None:
    """Check if a run with functionally identical metadata already exists.

    It compares `new_metadata` against all existing `metadata.json` or `initial_metadata.json`
    files in the `runs_folder`, after stripping away irrelevant keys specified in `ignore_keys`.

    Args:
        runs_folder (Path): The base directory containing all experiment run folders.
        new_metadata (dict[str, Any]): The metadata dictionary for the new run about to be started.
        ignore_keys (list[str | tuple[str, ...]]): A list of keys to exclude from the comparison.
            Each key can be a string for top-level keys or a tuple for nested keys.
        debug (bool): If True, prints detailed debug information about the comparison process.

    Returns:
        (Path | None): The Path to the existing run directory if a match is found, otherwise None.
    """
    # Create a "comparable" version of the new metadata by removing ignored keys.
    # We work on a deep copy to avoid modifying the original dict.
    comparable_new_meta = deepcopy(new_metadata)
    delete_nested_keys(comparable_new_meta, ignore_keys)

    # Find all possible metadata files in the runs folder
    existing_meta_files = find_files(
        runs_folder,
        name_patterns=["metadata.json", "initial_metadata.json"],
        followlinks=True,
    )

    for meta_path_str in existing_meta_files:
        if debug:
            logger.debug(f"Checking {meta_path_str}")

        meta_path = Path(meta_path_str)
        try:
            # Load the existing metadata
            existing_meta = load_json(meta_path)

            # Create a comparable version of the existing metadata
            comparable_existing_meta = deepcopy(existing_meta)
            delete_nested_keys(comparable_existing_meta, ignore_keys)

            # Compare the sanitized dictionaries
            differences = diff_dicts(comparable_existing_meta, comparable_new_meta)
            if not differences:
                if debug:
                    logger.debug(f"Found existing run with identical metadata: {meta_path.parent}")
                return meta_path.parent  # Return the path to the run directory

            if debug and differences:
                logger.debug("-" * 80)
                logger.debug(f"Comparing new config against existing run: {meta_path.parent.name}")
                logger.debug("Found differences:")
                pretty_print_diff(differences)
                logger.debug("-" * 80)
        except Exception as e:
            logger.warning(f"Could not read or compare metadata for {meta_path}: {e}")

    return None  # No identical run was found


def check_run_exists_by_config(
    runs_folder: Path,
    new_config: dict[str, Any],
    ignore_keys: Sequence[str | tuple[str, ...]],
    *,
    debug: bool = False,
) -> Path | None:
    """Check if a run with an identical config.yml already exists and is completed.

    This is a fast "pre-flight" check that avoids loading large metadata files.

    Args:
        runs_folder (Path): The base directory containing experiment run folders.
        new_config (dict[str, Any]): The configuration dictionary for the new run.
        ignore_keys (Sequence[str | tuple[str, ...]]): A list of keys to exclude from the
            comparison. Can be dot-separated strings for nested keys
            (e.g., "training_params.patience").
        debug (bool): If True, prints detailed debug information.

    Returns:
        Path: The Path to the existing completed run directory if a match is found, otherwise None.
    """
    comparable_new_config = deepcopy(new_config)

    # Convert dot-notation keys to tuples for the delete helper
    keys_to_delete: list[tuple[str, ...] | str] = []
    for k in ignore_keys:
        if isinstance(k, str) and "." in k:
            keys_to_delete.append(tuple(k.split(".")))
        else:
            keys_to_delete.append(k)

    delete_nested_keys(comparable_new_config, keys_to_delete)

    if not runs_folder.is_dir():
        return None

    # Find all potential run directories
    existing_configs = find_files(runs_folder, name_patterns="config.yml", followlinks=True)

    for config_path in existing_configs:
        run_dir = Path(config_path).parent

        # Only check runs that aren't failed or empty
        if get_run_status(run_dir) not in (RunStatus.COMPLETED, RunStatus.RUNNING):
            continue

        try:
            existing_config = load_yaml(config_path)
            comparable_existing_config = deepcopy(existing_config)
            delete_nested_keys(comparable_existing_config, keys_to_delete)

            # Compare the sanitized dictionaries
            differences = diff_dicts(comparable_existing_config, comparable_new_config)
            if not differences:
                if debug:
                    logger.debug(f"Found existing run with identical metadata: {run_dir}")
                return run_dir

            if debug and differences:
                logger.debug("-" * 80)
                logger.debug(f"Comparing new config against existing run: {run_dir}")
                logger.debug("Found differences:")
                pretty_print_diff(differences)
                logger.debug("-" * 80)
        except Exception as e:
            if debug:
                logger.error(f"Could not read or compare config for {config_path}: {e}. Skipping")
            continue

    return None


def get_run_status(run_dir: Path) -> RunStatus:
    """Determine the status of a single experiment run directory.

    Args:
        run_dir (Path): The path to the experiment run directory.

    Returns:
        RunStatus: One of "running", "completed", "failed", or "empty".
    """
    if not run_dir.is_dir():
        return RunStatus.EMPTY  # Should not happen if called correctly

    items = list(run_dir.iterdir())
    item_names = [item.name for item in items]

    if ".lock" in item_names:
        return RunStatus.RUNNING

    if "metadata.json" in item_names:
        return RunStatus.COMPLETED

    # Check for failure JSON files (e.g., "TypeError.json", "FAILURE_ValueError.json")
    # This regex matches common error names or the FAILURE_ prefix.
    failure_pattern = re.compile(r"^(FAILURE_)?[A-Z][a-zA-Z]*Error\.json$")
    json_files = [f for f in item_names if f.endswith(".json")]

    is_failed_run = False
    if len(json_files) == 1 and failure_pattern.match(json_files[0]):
        # Check if there are other significant files besides the error json
        other_files = [item for item in item_names if not failure_pattern.match(item)]
        # Allow for a config.yml and run.log to exist alongside the error file
        if all(f in {"config.yml", "run.log"} for f in other_files):
            is_failed_run = True

    if is_failed_run:
        return RunStatus.FAILED

    return RunStatus.EMPTY


def summarize_runs(base_dir: Path, indent: str = "", *, return_string: bool = False) -> str | None:
    """Recursively scans a directory and prints a summary of experiment run statuses.

    Args:
        base_dir (Path): The directory to start scanning from.
        indent (str): The indentation string for formatting the output tree.
        return_string (bool): If True, returns the summary as a string. If False, prints it.
    """
    if not base_dir.is_dir():
        return None

    output_lines = []
    run_statuses = []
    sub_dirs_to_scan = []

    # First, scan items in the current directory
    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            # Heuristic: If a directory name looks like a run folder, analyze it.
            # Otherwise, assume it's a category folder (like "EEGNetv4") and recurse into it.
            if re.match(r"^\d{6}_", item.name):  # Matches "DDMMYY_" prefix
                status = get_run_status(item)
                if status != RunStatus.EMPTY:
                    run_statuses.append(status)
            else:
                sub_dirs_to_scan.append(item)

    if run_statuses:
        status_counts = Counter(run_statuses)
        completed = status_counts.get(RunStatus.COMPLETED, 0)
        running = status_counts.get(RunStatus.RUNNING, 0)
        failed = status_counts.get(RunStatus.FAILED, 0)
        total = len(run_statuses)
        summary_str = (
            f"Completed: {completed}, Running: {running}, "
            f"Failed/Skipped: {failed} (Total: {total})"
        )
        output_lines.append(f"{indent}📁 {base_dir.name}: {summary_str}")

    for sub_dir in sub_dirs_to_scan:
        # Recursively summarize subdirectories
        nested_summary = summarize_runs(sub_dir, indent=indent + "  ", return_string=True)
        if nested_summary:
            output_lines.append(nested_summary)

    final_output = "\n".join(filter(None, output_lines))

    if return_string:
        return final_output

    print(final_output)
    return None


def cleanup_failed_runs(base_dir: Path, *, dry_run: bool = True) -> None:
    """Find and optionally deletes failed/skipped experiment run directories.

    A "failed" run is defined as a directory containing only a single JSON file
    whose name is a Python exception (e.g., 'ValueError.json'), and optionally
    a 'config.yml' and 'run.log'.

    Args:
        base_dir (Path): The root directory to scan for failed runs.
        dry_run (bool): If True, only reports the directories that would be deleted.
            If False, performs the deletion.
    """
    logger.info(f"Scanning for failed runs in {base_dir}...")
    logger.info(f"Dry run mode: {'ON' if dry_run else 'OFF'}")

    # Group failed runs by their parent directory
    grouped_dirs_to_delete = defaultdict(list)
    for item in base_dir.rglob("*"):
        if (
            item.is_dir()
            and re.match(r"^\d{6}_", item.name)
            and get_run_status(item) == RunStatus.FAILED
        ):
            grouped_dirs_to_delete[item.parent].append(item)

    if not grouped_dirs_to_delete:
        logger.success("No failed run directories found to clean up.")
        return

    # --- Print Grouped Summary ---
    total_to_delete = 0
    summary_lines = ["Found failed run directories to be cleaned up:"]
    # Sort by path for consistent output
    for parent_dir, failed_runs in sorted(grouped_dirs_to_delete.items()):
        count = len(failed_runs)
        total_to_delete += count
        relative_path = parent_dir.relative_to(base_dir)
        dir_name = base_dir.name if str(relative_path) == "." else relative_path
        summary_lines.append(f"  - 📁 {dir_name}: {count} failed run(s)")

    summary_lines.append(f"Total to be cleaned up: {total_to_delete}")
    logger_per_line("\n".join(summary_lines), level="info")

    if not dry_run:
        confirm = input("Are you sure you want to permanently delete these directories? (y/N): ")
        if confirm.lower() == "y":
            logger.warning("Proceeding with deletion...")
            deleted_count = 0
            for parent_dir, failed_runs in grouped_dirs_to_delete.items():
                logger.info(f"Cleaning up '{parent_dir.relative_to(base_dir)}'...")
                for d in failed_runs:
                    try:
                        remove_folder(d)  # Uses shutil.rmtree
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"  - Failed to delete {d.name}: {e}")
            logger.success(f"Successfully deleted {deleted_count}/{total_to_delete} directories.")
        else:
            logger.info("Deletion cancelled by user.")
    else:
        logger.info("This was a dry run. No files were deleted.")


def find_running_run_dirs(base_dir: Path) -> dict[Path, list[Path]]:
    """Recursively find all run directories that contain a '.lock' file.

    Args:
        base_dir (Path): The directory to start scanning from.

    Returns:
        (dict[Path, list[Path]]): A dictionary where keys are the parent directories (experiment
            categories) and values are lists of the full paths to the running run directories.
    """
    running_dirs_by_parent = defaultdict(list)

    # Iterate through all items in the base directory recursively
    for item in base_dir.rglob("*"):
        # We are only interested in directories that look like run folders
        if (
            item.is_dir()
            and re.match(r"^\d{6}_", item.name)
            and get_run_status(item) == RunStatus.RUNNING
        ):
            parent_dir = item.parent
            running_dirs_by_parent[parent_dir].append(item)

    return dict(sorted(running_dirs_by_parent.items()))


def find_metadata_files(base_dir: Path, *, verbose: bool = False) -> list[Path]:
    """Recursively finds all 'metadata.json' files within a base directory.

    Args:
        base_dir (Path): The root directory to start searching from.
        verbose (bool): If True, enables detailed logging of the search process.

    Returns:
        list[Path]: A list of Paths to the found 'metadata.json' files.
    """
    logger.info(f"Searching for 'metadata.json' files in {base_dir}...")
    files = find_files(base_dir, name_patterns="metadata.json")
    files_p = [Path(f) for f in files]
    logger.info(f"Found {len(files_p)} metadata files" + (":" if verbose else "."))
    if verbose:
        for f in files_p:
            # Print only the last 3 parents and name
            logger.info(f" - {f.relative_to(base_dir.parent)}")
    return files_p


@overload
def find_best_model_in_experiments(
    experiments_base_dir: Path,
    *,
    best_metric: str = "val_loss",
    top_k: int | None = None,
    group_by_key_path: tuple[str, ...] | None = None,
    return_structured: Literal[False] = False,
    verbose: bool = False,
) -> list[Path]: ...


@overload
def find_best_model_in_experiments(
    experiments_base_dir: Path,
    *,
    best_metric: str = "val_loss",
    top_k: int | None = None,
    group_by_key_path: tuple[str, ...] | None = None,
    return_structured: Literal[True],
    verbose: bool = False,
) -> dict[Any, list[RunInfo]]: ...


def find_best_model_in_experiments(
    experiments_base_dir: Path,
    *,
    best_metric: str = "val_loss",
    top_k: int | None = None,
    group_by_key_path: tuple[str, ...] | None = None,
    return_structured: bool = False,
    verbose: bool = False,
) -> list[Path] | dict[Any, list[RunInfo]]:
    """Find the top k best performing models, with optional grouping and structured return.

    Scans all `metadata.json` files recursively, ranks them based on a specified
    metric, and returns the paths to the best ones.

    Args:
        experiments_base_dir (Path): The root directory containing experiment folders.
        best_metric (str): The metric to use for ranking
            (e.g., "val_loss", "val_Balanced Acuracy"). The "val_" prefix is recommended.
        top_k (int | None): The number of best models to return for each group. If None, returns
            all found models, sorted from best to worst.
        group_by_key_path (tuple[str, ...] | None): An optional tuple of strings representing the
            nested key path to group models by (e.g., `('model_name',)`).
        return_structured (bool): If True, returns a dictionary mapping group names to a sorted
            list of RunInfo objects. If False, returns a flat list of paths.
        verbose (bool): If True, enables detailed logging of the process.

    Returns:
        (list[Path] | dict[Any, list[RunInfo]]): If `return_structured` is True, a dictionary
        mapping group names to lists of `RunInfo` dicts. Otherwise, a flat list of `Path` objects
        to the best-performing models' metadata  files.
    """
    if verbose:
        logger.info(f"Analyzing runs under '{experiments_base_dir.name}' for best models...")

    all_metadata_files = find_files(experiments_base_dir, name_patterns="metadata.json")
    if not all_metadata_files:
        logger.warning(f"No 'metadata.json' files found in '{experiments_base_dir}'.")
        return {} if return_structured else []

    # --- 1. Collect data from all runs ---
    all_runs_data = []
    lower_is_better = "loss" in best_metric.lower()

    for metadata_path_str in all_metadata_files:
        metadata_path = Path(metadata_path_str)
        try:
            metadata = load_json(metadata_path)
            metric_history = metadata.get("history", {}).get(best_metric, [])
            if not metric_history:
                if verbose:
                    logger.warning(
                        f"Metric '{best_metric}' not found in 'history' of {metadata_path}"
                    )
                continue

            best_value = min(metric_history) if lower_is_better else max(metric_history)

            group_value = None
            if group_by_key_path:
                group_value = get_nested_value(metadata, list(group_by_key_path))
                if group_by_key_path == ("model_name",):
                    group_value = extract_model_name(metadata)
                if isinstance(group_value, list):
                    group_value = tuple(group_value)

            all_runs_data.append(
                {"path": metadata_path, "best_value": best_value, "group": group_value}
            )
        except Exception as e:
            logger.error(f"Could not process {metadata_path}: {e}")

    # --- 2. Group and Sort ---
    if group_by_key_path:
        # Group runs by the extracted group_value
        grouped_runs = defaultdict(list)
        for run in all_runs_data:
            if run["group"] is not None:
                grouped_runs[run["group"]].append(run)

        structured_results = {}
        for group, runs in grouped_runs.items():
            runs.sort(key=lambda r: r["best_value"], reverse=not lower_is_better)
            top_runs = runs[:top_k] if top_k is not None else runs

            structured_results[group] = [
                RunInfo(path=run["path"], best_value=run["best_value"]) for run in top_runs
            ]
            if verbose:
                logger.success(f"  -> Top {len(top_runs)} model(s) for group '{group}':")
                for i, run_info in enumerate(structured_results[group]):
                    logger.info(
                        f"    #{i + 1}: {run_info['path'].parent.name} "
                        f"({best_metric}: {run_info['best_value']:.4f})"
                    )

        if return_structured:
            return structured_results

        # Flatten for the old behavior
        flat_paths = []
        for runs_info in structured_results.values():
            flat_paths.extend([info["path"] for info in runs_info])
        return flat_paths

    # No grouping, just sort all runs together
    all_runs_data.sort(key=lambda r: r["best_value"], reverse=not lower_is_better)
    top_runs = all_runs_data[:top_k] if top_k is not None else all_runs_data
    if verbose:
        logger.success(f"  -> Top {len(top_runs)} model(s) overall:")
        for i, run in enumerate(top_runs):
            logger.info(
                f"    #{i + 1}: {run['path'].parent.name} ({best_metric}: {run['best_value']:.4f})"
            )

    if return_structured:
        return {
            "all": [RunInfo(path=run["path"], best_value=run["best_value"]) for run in top_runs]
        }

    return [run["path"] for run in top_runs]


def format_hyperparameter_for_filename(value: Any) -> str:  # noqa: ANN401
    """Format a hyperparameter value into a concise, filename-safe string.

    - Booleans -> 'T' or 'F'
    - Lists -> 'item1-item2-item3'
    - Floats -> Scientific notation for small numbers, otherwise standard string.
    - None -> 'None'

    Args:
        value (Any): The hyperparameter value to format.

    Returns:
        str: A clean string representation suitable for a filename.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        if not value:
            return "empty"
        return "-".join(map(str, value))
    if isinstance(value, float):
        # Use scientific notation for small floats to keep it short
        return f"{value:0.1e}" if abs(value) > 0 and abs(value) < 1e-3 else str(value)
    if value is None:
        return "None"

    # For all other types, convert to string and sanitize
    clean_str = str(value)
    # Remove or replace characters that are problematic in filenames
    return re.sub(r'[\\/*?:"<>|]', "_", clean_str)
