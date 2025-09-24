"""Run a complete training experiment based on a YAML configuration file."""

import inspect
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from random import shuffle
from typing import Any, TypedDict, cast

import yaml
from cyclopts import App
from loguru import logger
from matplotlib import cm
from prettytable import PrettyTable
from torch import nn, optim
from tqdm import tqdm

from pretrain_braindecode_models.config import DEVICE, MODELS_DIR, PROJ_ROOT
from pretrain_braindecode_models.features import prepare_tuh_dataset
from pretrain_braindecode_models.modeling import models, train
from pretrain_braindecode_models.utils.colors import color_text, create_custom_colormap
from pretrain_braindecode_models.utils.custom_types import RunStatus
from pretrain_braindecode_models.utils.folders import (
    calculate_dir_size,
    check_run_exists,
    check_run_exists_by_config,
    cleanup_failed_runs,
    find_files,
    find_running_run_dirs,
    format_hyperparameter_for_filename,
    format_size_bytes,
    get_dir_last_modified_time,
    get_run_status,
    new_run,
    remove_folder,
    summarize_runs,
)
from pretrain_braindecode_models.utils.grid_search import generate_grid_search_configs
from pretrain_braindecode_models.utils.loading import (
    load_json,
    load_yaml,
    save_json,
)
from pretrain_braindecode_models.utils.logging import setup_logging
from pretrain_braindecode_models.utils.misc import (
    get_nested_value_by_dot_string,
    logger_per_line,
    make_hashable,
    pretty_print_dict,
    recursive_update,
)
from pretrain_braindecode_models.utils.timeutils import today

# ANSI color codes for Status column
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_RESET = "\033[0m"

app = App()


@contextmanager
def run_lock(run_dir: Path) -> Generator[None, None, None]:
    """Handle handle lock file creation and cleanup via context manager."""
    lock_file = run_dir / ".lock"
    try:
        lock_file.touch()
        logger.info(f"Acquired lock for run: {run_dir.name}")
        yield
    finally:
        lock_file.unlink(missing_ok=True)
        logger.info(f"Released lock for run: {run_dir.name}")


def _build_config_for_comparison(config: dict[str, Any]) -> dict[str, Any]:
    """Build a stripped-down config dictionary used for checking for duplicates."""
    # Create a deep copy to avoid modifying the original config dict
    comparable_config = deepcopy(config)

    # Use a set of keys to ignore for efficient lookup
    keys_to_ignore = {
        "experiment_name",
        "models_subdir",
        "delete_when_error",
        "task",
        "patience",
        "plot_kwargs",
        "plot_every_epoch",
        "save_train_test_data",
    }

    # Create a new dictionary with only the relevant keys
    clean_config = {}
    for key, value in comparable_config.items():
        if key not in keys_to_ignore:
            clean_config[key] = value  # noqa: PERF403

    # Also remove specific nested keys if they exist
    if "model_kwargs" in clean_config.get("model", {}):
        clean_config["model"]["model_kwargs"].pop("name", None)

    return clean_config


def _handle_config_inheritance(
    root_config: dict[str, Any],
    *,
    remove_inheritance_key: bool = False,
) -> dict[str, Any]:
    """Handle configuration inheritance from a base config file.

    If `from_config` doesn't exist, it simply returns the original `root_config`. Otherwise,

    1. Start with a deep copy of the base config.
    2. Recursively update the base config with the keys from the override config
    (the one with the `from_config` key). This ensures that any key you define in your new file
    will overwrite the corresponding key from the base file.

    Args:
        root_config (dict[str, Any]): The root configuration dictionary.
        remove_inheritance_key (bool, optional): If True, removes the `from_config` key
            from the final configuration. Defaults to False.

    Returns:
        (dict[str, Any]): The final configuration dictionary after applying inheritance.
    """
    if "from_config" not in root_config:
        return deepcopy(root_config)  # Return a copy to avoid side effects

    base_config_path_str = root_config["from_config"]
    # Adjust path to be relative to MODELS_DIR
    base_config_path = MODELS_DIR / base_config_path_str

    if not base_config_path.exists():
        logger.error(
            f"Base config specified in 'from_config' not found at {base_config_path}. "
            "Returning original config without inheritance."
        )
        return deepcopy(root_config)

    try:
        base_config = load_yaml(base_config_path)
        # Start with the base config and recursively update it with the overrides
        merged_config = recursive_update(deepcopy(base_config), root_config)
        if remove_inheritance_key:
            # Remove the inheritance key from the final config as it has been processed
            merged_config.pop("from_config", None)
    except Exception as e:
        logger.error(f"Failed to load or merge base config: {e}")
        return deepcopy(root_config)
    else:
        return merged_config


def _format_experiment_name(config: dict[str, Any]) -> str:
    """Format the experiment name by substituting placeholders with hyperparameter values.

    Placeholders can be nested, e.g., {training_params.batch_size}.

    For example, if the config has:
    ```
        "experiment_name": "exp_{training_params.batch_size}_{model.model_class}"
    ```
    and the config has:
    ```
        "training_params": {"batch_size": 32},
        "model": {"model_class": "MyModel"},
    ```
    the formatted name will be: `"exp_32_MyModel"`

    Args:
        config (dict[str, Any]): The configuration dictionary containing hyperparameters.
    """
    name_template = config.get("experiment_name", "experiment")
    placeholders = re.findall(r"\{([^}]+)\}", name_template)

    if not placeholders:
        return name_template

    format_kwargs = {}
    for placeholder in placeholders:
        # The key for .format() should not contain dots
        clean_key = placeholder.replace(".", "_")
        value = get_nested_value_by_dot_string(config, placeholder, default="<NOT_FOUND>")
        if value == "<NOT_FOUND>":
            logger.warning(
                f"Placeholder '{{{placeholder}}}' in experiment_name "
                "not found in config. It will be replaced with '<NOT_FOUND>'."
            )

        format_kwargs[clean_key] = format_hyperparameter_for_filename(value)
        # We also need to replace the dotted placeholder with the cleaned key in the template
        name_template = name_template.replace(f"{{{placeholder}}}", f"{{{clean_key}}}")

    return name_template.format(**format_kwargs)


def _run_single_experiment(
    config: dict[str, Any],
    runs_folder: Path,
    *,
    debug: bool = False,
    force_new: bool = False,
) -> None:
    """Execute a single, fully-specified training experiment.

    Args:
        config (dict[str, Any]): A dictionary containing the complete configuration for one run.
        runs_folder (Path): The base directory where all experiment runs are stored.
        debug (bool, optional): Whether to enable debug mode for console-level logging.
            Defaults to False.
        force_new (bool, optional): If True, forces the execution of a new run even if an
            identical configuration has been run before. Defaults to False.
    """
    formatted_experiment_name = _format_experiment_name(config)

    # Update the config in-place so this name is used for logging and metadata.
    config["experiment_name"] = formatted_experiment_name

    # 1. Check if an identical run already exists
    # Create the metadata dict *before* creating the run dir.
    # Note: This is a simplified metadata dict for comparison purposes.
    # The full one inside `train_model` later.
    scaler = models.get_scaler(
        config["training_params"]["scaler"],
        config["training_params"].get("scaler_kwargs", {}),
    )

    # 2. Get a new, clean, and unlocked run directory
    RUN_DIR = new_run(
        runs_folder,
        prefix=f"{today()}_{config['experiment_name']}_",
        ignore_files={
            "initial_metadata.json",
            "run.log",
            "losses.png",
            "losses_per_expr_bar.png",
            "losses_per_expr_line.png",
            "losses_per_expr_box.png",
            "losses_per_comp_loss.png",
            "config.yml",
        },
        ignore_dirs={"figures", "checkpoint"},
    )

    if RUN_DIR is None:
        logger.error("Could not find or create an unlocked run directory. Aborting.")
        return

    # Setup Logging for this Specific Run
    setup_logging(RUN_DIR, console_level="DEBUG" if debug else "INFO", file_level="DEBUG")

    logger.success(f"--- Running experiment: {formatted_experiment_name} ---")

    try:
        remove_run_dir = False  # Flag to determine if the run directory should be deleted later

        # 3. Acquire lock and run the training
        with run_lock(RUN_DIR):
            # Make a copy of the yaml config in the run directory
            config_copy_path = RUN_DIR / "config.yml"
            logger.info(f"Saving specific run configuration to: {config_copy_path}")
            with config_copy_path.open("w") as f:
                yaml.dump(config, f, indent=2, sort_keys=False)

            # Print the entire current config.yml
            logger.info(f"Current config.yml:\n{pretty_print_dict(config)}")

            # a. Prepare Data
            dataset_name = config["dataset_params"].pop("name")

            if dataset_name == "tuh_abnormal":
                logger.info("--- Preparing TUH Abnormal EEG Dataset ---")
                train_dataset, test_dataset = prepare_tuh_dataset(
                    **{
                        k: v
                        for k, v in config["dataset_params"].items()
                        if k not in {"precision", "dim_order"}
                    }
                )

                # For classification, y_train and y_test are just the labels
                X_train = train_dataset.windows.numpy()
                y_train = train_dataset.labels.numpy()
                X_test = test_dataset.windows.numpy()
                y_test = test_dataset.labels.numpy()

                # Class labels, e.g., {0: "normal", 1: "abnormal"}
                class_labels = train_dataset.class_labels
            else:
                raise NotImplementedError("Dataset not supported")

            # b. Prepare training data
            (
                (X_train, X_test, scaler),
                (model, my_optimizer, my_criterion),
                config,
            ) = train.setup_train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                config=config,
                device=DEVICE,
            )

            # c. Check for existing runs
            existing_run_path = check_run_exists(
                runs_folder,
                new_metadata=config,
                # This list is crucial for defining what makes an experiment "unique"
                ignore_keys=[
                    "experiment_name",
                    "from_config",
                    "models_subdir",
                    "history",
                    "num_params",
                    "shapes",
                    "criterion",
                    "optimizer",
                    ("model", "model_name"),
                    ("model", "model_kwargs", "name"),
                    ("training_params", "plot_kwargs"),
                    ("training_params", "save_train_test_data"),
                    ("training_params", "patience"),
                    ("training_params", "plot_every_epoch"),
                    ("losses", "component_weighting", "weights"),
                    ("losses", "component_weighting", "epsilon"),
                ],
            )

            if not force_new and existing_run_path:
                logger.warning(
                    f"Skipping run. Identical completed run found at: {existing_run_path}"
                )
                remove_run_dir = True  # Set to True to remove the run directory after checking
            else:
                # f. Run Training
                logger.info("--- Starting Training ---")

                if not inspect.isclass(my_optimizer) or not issubclass(
                    my_optimizer, optim.Optimizer
                ):
                    raise TypeError(
                        "For classification tasks, the optimizer should be a class, not an "
                        "instance."
                    )
                if not inspect.isclass(my_criterion) or not issubclass(
                    my_criterion, nn.CrossEntropyLoss
                ):
                    raise TypeError(
                        "For classification tasks, the criterion should be a class, not an "
                        "instance."
                    )

                train.train_classifier_lightning(
                    model=model,
                    X_train=X_train,
                    y_expr_train=y_train,
                    X_test=X_test,
                    y_expr_test=y_test,
                    scaler=scaler,
                    class_labels=class_labels,
                    metadata=config,
                    run_dir=RUN_DIR,
                )

                logger.success(
                    f"Experiment '{config['experiment_name']}' finished. Results are in: {RUN_DIR}"
                )

        if remove_run_dir:
            # Delete the run directory that was just created if it was marked for deletion
            logger.warning(f"Deleting run directory: {RUN_DIR}")
            remove_folder(RUN_DIR)

    except Exception as e:
        logger.error(f"An {e.__class__.__name__} occurred during the experiment:")
        logger.exception(e)  # Logs the full traceback

        # Create a failure metadata file
        failure_metadata = {
            "error": e.__class__.__name__,
            "message": str(e),
            "run_config": config,
        }
        failure_metadata_path = RUN_DIR / f"{e.__class__.__name__}.json"
        save_json(failure_metadata, failure_metadata_path, serialize=True)

        if config.get("delete_when_error", False):
            logger.info(f"Deleting run directory due to error: {RUN_DIR}")
            shutil.rmtree(RUN_DIR, ignore_errors=True)
        else:
            logger.info(f"Saved failure metadata to {failure_metadata_path}")
            logger.warning(
                f"Run directory kept for debugging: {RUN_DIR}. "
                "Set 'delete_when_error: true' in the config to delete it automatically."
            )


@app.command
def inspect_running(
    config_name: str | None = None,
    configs_dir: Path = PROJ_ROOT / "pretrain_braindecode_models" / "configs",
    runs_base_dir: Path = MODELS_DIR,
    *,
    exclude: bool = False,
    delete: bool = False,
) -> None:
    """Inspect and list the paths of currently 'running' experiments.

    A run is considered 'running' if its directory contains a '.lock' file.
    This is useful for finding stale locks from crashed or killed jobs.

    Args:
        config_name (str | None): The name of the config file to filter by. If provided, only
            shows running jobs associated with this specific config file (e.g., "TCN", "EEGNetv4").
        configs_dir (Path): The directory containing the YAML configuration files.
        runs_base_dir (Path): The base directory where all experiment runs are stored.
            Defaults to `MODELS_DIR`.
        exclude: If set, inverts the `config_name` filter. It will inspect all running jobs EXCEPT
            those from the specified config. Only has an effect if `config_name` is also provided.
        delete (bool): If set, prompts for confirmation and then deletes the found 'running'
            directories. DANGEROUS - use with care.
    """
    logger.info(f"Inspecting for running jobs (directories with a .lock file) in {runs_base_dir}")

    # --- Determine the scope of the inspection ---
    dirs_to_inspect = {}

    if config_name:
        # Load the specified config to get its `models_subdir`
        config_path = configs_dir / f"{config_name}.yml"
        if not config_path.exists():
            config_path = configs_dir / f"{config_name}.yaml"
            if not config_path.exists():
                logger.error(f"Config file for '{config_name}' not found.")
                return

        try:
            with config_path.open("r") as f:
                config = yaml.safe_load(f)
            models_subdir = config.get("models_subdir")
            if not models_subdir:
                logger.error(f"'models_subdir' not found in config '{config_path.name}'.")
                return

            filter_dir = runs_base_dir / models_subdir

            if exclude:
                # --- EXCLUDE LOGIC ---
                dirs_to_inspect = find_running_run_dirs(runs_base_dir)

                logger.info(
                    f"Excluding runs from config '{config_name}' (directory: {filter_dir})"
                )
                # Flatten all paths into a single set
                all_paths_set = {path for paths in dirs_to_inspect.values() for path in paths}
                # Find paths to exclude
                excluded_paths = {
                    path for path in all_paths_set if path.is_relative_to(filter_dir)
                }
                # Get the final set of paths to inspect
                final_paths = all_paths_set - excluded_paths

                # Rebuild the grouped dictionary for printing
                dirs_to_inspect = defaultdict(list)
                for path in final_paths:
                    dirs_to_inspect[path.parent].append(path)
                dirs_to_inspect = dict(sorted(dirs_to_inspect.items()))

            else:
                # --- INCLUDE LOGIC ---
                logger.info(f"Filtering inspection to directory: {filter_dir.resolve()}")
                dirs_to_inspect = find_running_run_dirs(filter_dir.resolve())

        except Exception as e:
            logger.error(f"Failed to process config '{config_path.name}': {e}")
            return

    if not dirs_to_inspect:
        logger.success("No matching running jobs (stale .lock files) found.")
        return

    total_to_delete = 0
    all_paths_to_delete = []

    logger.info("Found the following 'running' experiment directories:")

    for parent_dir, run_paths in dirs_to_inspect.items():
        relative_parent = parent_dir.relative_to(runs_base_dir)
        print(f"📁 {C_YELLOW}{relative_parent}{C_RESET} ({len(run_paths)} running):")
        for path in sorted(run_paths):
            print(f"  - {path}")
            all_paths_to_delete.append(path)
        total_to_delete += len(run_paths)

    if delete:
        logger.warning(f"Deletion flag is active. {total_to_delete} directories are targeted.")
        # Ask for explicit confirmation
        confirm = input(f"Type 'yes' to permanently delete these {total_to_delete} directories: ")

        if confirm == "yes":
            logger.info("Proceeding with deletion...")
            deleted_count = 0
            for path in all_paths_to_delete:
                try:
                    remove_folder(path)
                    logger.success(f"  Deleted: {path}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"  Failed to delete {path}: {e}")
            logger.success(f"Successfully deleted {deleted_count}/{total_to_delete} directories.")
        else:
            logger.info("Deletion cancelled by user.")
    else:
        logger.warning("This was a dry run. To delete these directories, use the --delete flag.")
        logger.warning(
            "If jobs are not actually running, you can also manually delete the '.lock' file "
            "in each directory."
        )


@app.command
def diff_runs(
    config_name: str,
    configs_dir: Path = PROJ_ROOT / "pretrain_braindecode_models" / "configs",
    runs_base_dir: Path = MODELS_DIR,
) -> None:
    """Compare a config's expected runs against completed runs on disk.

    This command identifies discrepancies, such as:
    - Runs on disk that are not defined by the current config (e.g., from old versions).
    - Runs defined in the config that have not yet been completed.
    - Runs that have been completed multiple times.

    Args:
        config_name (str): The name of the config file to analyze (e.g., "TCN").
        configs_dir (Path): The directory containing the YAML configuration files.
        runs_base_dir (Path): The base directory where all experiment runs are stored.
    """
    logger.info(f"Comparing expected vs. actual runs for config: '{config_name}.yml'")

    # --- 1. Load the specified config file ---
    config_path = configs_dir / f"{config_name}.yml"
    if not config_path.exists():
        config_path = configs_dir / f"{config_name}.yaml"
        if not config_path.exists():
            logger.error(f"Config file for '{config_name}' not found.")
            return

    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        models_subdir = config.get("models_subdir")
        if not models_subdir:
            logger.error(f"'models_subdir' not found in config '{config_path.name}'.")
            return
    except Exception as e:
        logger.error(f"Failed to load or parse config '{config_path.name}': {e}")
        return

    # --- 2. Generate the set of expected experiment fingerprints ---
    expected_configs = generate_grid_search_configs(config)
    expected_fingerprints = {
        make_hashable(_build_config_for_comparison(c)) for c in expected_configs
    }
    logger.info(
        f"Config '{config_path.name}' defines {len(expected_fingerprints)} unique experiments."
    )

    # --- 3. Find and group actual completed runs by their fingerprint ---
    target_dir = runs_base_dir / models_subdir
    actual_runs_by_fingerprint = defaultdict(list)

    if not target_dir.is_dir():
        logger.warning(f"Target directory '{target_dir}' does not exist.")
    else:
        run_dirs = [d for d in target_dir.iterdir() if d.is_dir() and re.match(r"^\d{6}_", d.name)]
        for run_dir in run_dirs:
            if get_run_status(run_dir) == RunStatus.COMPLETED:
                try:
                    run_config_path = run_dir / "config.yml"
                    if not run_config_path.exists():
                        continue
                    with run_config_path.open("r") as f:
                        run_config = yaml.safe_load(f)

                    fingerprint = make_hashable(_build_config_for_comparison(run_config))
                    actual_runs_by_fingerprint[fingerprint].append(run_dir)
                except Exception as e:
                    logger.warning(
                        f"Skipping run directory '{run_dir}' due to malformed config: {e}"
                    )

    actual_fingerprints = set(actual_runs_by_fingerprint.keys())
    total_actual_runs = sum(len(paths) for paths in actual_runs_by_fingerprint.values())
    logger.info(
        f"Found {total_actual_runs} completed run directories for "
        f"{len(actual_fingerprints)} unique experiment configurations in '{target_dir}'."
    )

    # --- 4. Compare the sets of fingerprints and report differences ---
    extra_fingerprints = actual_fingerprints - expected_fingerprints
    missing_fingerprints = expected_fingerprints - actual_fingerprints

    # Duplicates are now runs that share the same fingerprint
    duplicate_runs = {
        fp: paths for fp, paths in actual_runs_by_fingerprint.items() if len(paths) > 1
    }

    if extra_fingerprints:
        logger.warning(
            f"Found {len(extra_fingerprints)} run configurations on disk NOT defined by the "
            "current config:"
        )
        for fp in extra_fingerprints:
            for path in actual_runs_by_fingerprint[fp]:
                print(f"  - [Extra] {path}")

    if duplicate_runs:
        logger.warning(
            f"Found {len(duplicate_runs)} configurations with MULTIPLE completed runs "
            "(duplicates):"
        )
        for fp, paths in duplicate_runs.items():
            if fp not in extra_fingerprints:  # Don't double-report
                print(f"  - Config has {len(paths)} completed runs:")
                for path in paths:
                    print(f"    - {path}")

    if missing_fingerprints:
        logger.info(
            f"Found {len(missing_fingerprints)} runs defined in config but NOT completed on disk."
        )


@app.command
def grid_summary(
    configs_dir: Path = PROJ_ROOT / "pretrain_braindecode_models" / "configs",
    runs_base_dir: Path = MODELS_DIR,
    ignore_configs: set[str] = {"best_model_rules.yml"},
    *,
    update: bool = False,
    use_cache_only: bool = False,
) -> None:
    """Compare experiment configs against on-disk results and shows a progress summary.

    This command scans the `configs_dir` for all .yml files, calculates the expected number of runs
    for each grid search, and then checks the corresponding output directory
    (specified by `models_subdir` in each config) to report the number of completed, running, and
    failed runs.

    Args:
        configs_dir (Path): The directory containing the YAML configuration files.
        runs_base_dir (Path): The base directory where all experiment runs are stored.
        ignore_configs (set[str]): A set of config file names to ignore. By default,
            it includes "best_model_rules.yml".
        update (bool): If True, forces a re-scan of all configs and run directories,
            updating the cache file. If False, only scans configs that have changed since the
            last cache update. Defaults to False.
        use_cache_only (bool): If True, skips all file system checks and prints the summary
            directly from the last saved cache file.
    """

    class GridSummaryRow(TypedDict):
        config: str | Path | None
        expected: int
        completed: int
        unique_completed: int
        running: int
        failed: int
        completion_pct: float

    # --- Define and create a custom colormap ---
    # We define the transition from red -> yellow -> bright green
    # The hex code for the standard bright green ANSI color is approximately #00C000
    custom_cmap_colors = ["#FF0000", "#FFFF00", "#00C000"]  # Red -> Yellow -> Bright Green
    try:
        # Use our new function to create the colormap
        custom_cmap = create_custom_colormap("CustomProgressMap", custom_cmap_colors)
        cmap_to_use = custom_cmap
    except Exception as e:
        logger.warning(f"Could not create custom colormap due to: {e}. Falling back to 'RdYlGn'.")
        cmap_to_use = cm.get_cmap("RdYlGn")

    # --- Caching Logic ---
    cache_path = runs_base_dir / ".grid_summary_cache.json"
    cached_summary = load_json(cache_path)
    last_cache_time = cached_summary.get("last_updated", 0.0)

    summary_data: list[GridSummaryRow] = []

    if use_cache_only:
        logger.info("--- Displaying summary from cache (no new data scanned) ---")
        cached_summary = load_json(cache_path)
        if not cached_summary or "last_updated" not in cached_summary:
            logger.error(
                f"Cache file not found or is invalid at {cache_path}. Cannot display summary."
            )
            return

        for key, value in cached_summary.items():
            if key == "last_updated":
                continue

            # The value is already a dict matching GridSummaryRow, construct the object
            summary_data.append(cast("GridSummaryRow", value))
    else:  # Default behavior: scan and update cache
        logger.info(f"Scanning for YAML configs in: {configs_dir}")
        config_files = find_files(configs_dir, extensions=["yml", "yaml"])

        if not config_files:
            logger.warning("No config files found.")
            return

        needs_update = False

        for config_path_str in config_files:
            config_path = Path(config_path_str)

            if config_path.name in ignore_configs:
                continue

            try:
                with config_path.open("r") as f:
                    config = yaml.safe_load(f)

                # Use relative path as the unique key for display and caching
                relative_config_path = str(config_path.relative_to(configs_dir))

                models_subdir = config.get("models_subdir")
                if not models_subdir:
                    continue

                target_dir = runs_base_dir / models_subdir

                # --- Staleness Check ---
                config_mtime = config_path.stat().st_mtime
                target_dir_mtime = get_dir_last_modified_time(target_dir)
                is_stale = (
                    relative_config_path not in cached_summary
                    or config_mtime > last_cache_time
                    or target_dir_mtime > last_cache_time
                )

                if is_stale or update:
                    logger.debug(f"Re-calculating summary for '{relative_config_path}' (stale)...")
                    needs_update = True

                    # 1. Calculate EXPECTED runs based on the config file
                    expected_configs = generate_grid_search_configs(config, verbose=False)

                    # Create a set of hashable fingerprints for what we expect
                    expected_fingerprints = {
                        make_hashable(_build_config_for_comparison(c)) for c in expected_configs
                    }
                    expected_runs = len(expected_fingerprints)

                    # 2. Analyze the target directory to find ACTUAL runs
                    actual_run_fingerprints = set()
                    completed, running, failed = 0, 0, 0

                    if target_dir.is_dir():
                        run_dirs = [
                            d
                            for d in target_dir.iterdir()
                            if d.is_dir() and re.match(r"^\d{6}_", d.name)
                        ]
                        for run_dir in run_dirs:
                            status = get_run_status(run_dir)
                            if status == RunStatus.COMPLETED:
                                completed += 1
                                try:
                                    # Load the run's own config to generate its fingerprint
                                    run_config_path = run_dir / "config.yml"
                                    if not run_config_path.exists():
                                        continue
                                    with run_config_path.open("r") as f:
                                        run_config = yaml.safe_load(f)

                                    fingerprint = make_hashable(
                                        _build_config_for_comparison(run_config)
                                    )
                                    actual_run_fingerprints.add(fingerprint)
                                except Exception:
                                    failed += 1  # Treat un-parsable configs as failed
                            elif status == RunStatus.RUNNING:
                                logger.debug(f"\tFound running: {run_dir}")
                                running += 1
                            elif status == RunStatus.FAILED:
                                failed += 1

                    # The number of "unique" completed runs is the number of unique fingerprints
                    unique_completed_count = len(actual_run_fingerprints)
                    completion_pct = (
                        (unique_completed_count / expected_runs) if expected_runs > 0 else 0.0
                    )

                    current_data = GridSummaryRow(
                        config=relative_config_path,
                        expected=expected_runs,
                        completed=completed,
                        unique_completed=unique_completed_count,
                        running=running,
                        failed=failed,
                        completion_pct=completion_pct,
                    )
                    summary_data.append(current_data)
                    cached_summary[relative_config_path] = current_data
                else:
                    # Use cached data
                    logger.debug(f"Using cached summary for '{relative_config_path}'.")

                    # current_data = GridSummaryRow(**cached_summary[relative_config_path])
                    current_data = cast("GridSummaryRow", cached_summary[relative_config_path])
                    summary_data.append(current_data)

            except Exception as e:
                logger.error(f"Failed to process config '{config_path.name}': {e}")

        # Save updated cache if necessary
        if needs_update:
            logger.info(f"Updating summary cache at {cache_path}")
            cached_summary["last_updated"] = time.time()
            save_json(cached_summary, cache_path)

    if not summary_data:
        logger.info("No valid configs with 'models_subdir' found to summarize.")
        return

    # --- Sort data by completion percentage (descending) AND config file category (ascending) ---
    def sort_key(item: GridSummaryRow) -> tuple:
        if item["config"] is None:
            return (None, -item["completion_pct"])

        return (str(Path(item["config"]).parent), item["completion_pct"])

    summary_data = sorted(summary_data, key=sort_key, reverse=True)

    # --- Create and populate the PrettyTable ---
    table = PrettyTable()
    table.field_names = [
        "Config File",
        "Expected",
        "Completed (Unique)",
        "Running",
        "Failed",
        "Status",
    ]
    table.align["Expected"] = "r"
    table.align["Completed (Unique)"] = "r"
    table.align["Running"] = "r"
    table.align["Failed"] = "r"
    table.align["Config File"] = "l"
    previous_category = None
    for data in summary_data:
        # Format the "Completed" cell using the UNIQUE count for percentage
        pct = data["completion_pct"]
        # Show both total and unique counts: e.g., "1328 (1088 unique)"
        completed_str = f"{data['completed']} ({data['unique_completed']})"

        colored_completed_str = color_text(
            f"{completed_str} ({pct:.0%})", value=pct, cmap_name=cmap_to_use
        )

        # Determine status string
        actual_total = data["completed"] + data["running"] + data["failed"]
        status_str = ""
        if data["running"] > 0:
            status_str = f"{C_BLUE}In Progress...{C_RESET}"
        elif data["expected"] == 0:
            status_str = f"{C_YELLOW}No Runs Defined{C_RESET}"
        elif (
            data["completed"] >= data["expected"] and data["failed"] == 0 and data["running"] == 0
        ):
            status_str = f"{C_GREEN}Complete{C_RESET}"
        elif actual_total < data["expected"]:
            pending = data["expected"] - actual_total
            status_str = f"{C_YELLOW}Incomplete ({pending} pending){C_RESET}"
        else:
            status_str = f"{C_RED}Mismatch{C_RESET}"

        if data["config"]:
            # e.g., "classification", "regression"
            current_category = str(Path(data["config"]).parent)

            if previous_category != current_category:
                table.add_divider()
                previous_category = current_category

        table.add_row(
            [
                str(data["config"]),
                int(data["expected"]),
                colored_completed_str,
                int(data["running"]),
                int(data["failed"]),
                status_str,
            ]
        )

    table.sortby = None
    print(table)


@app.command
def summary(
    runs_folder: Path = MODELS_DIR / "experiments",
) -> None:
    """Print a summary of the status of all experiment runs.

    Scans the specified directory recursively for run folders and categorizes them
    as running, completed, or failed.

    Args:
        runs_folder (Path): The base directory to scan for experiments.
    """
    title = f"🔬 Experiment Status Summary for: {runs_folder}"
    header = "-" * len(title)

    # Get the summary as a string from the improved function
    summary_str = summarize_runs(runs_folder, return_string=True)

    if not summary_str:
        summary_str = "No experiment runs found."

    # Use logger_per_line for clean, multi-line output
    full_log_message = f"{header}\n{title}\n{header}\n{summary_str}\n{header}"
    logger_per_line(full_log_message, level="info")


@app.command
def cleanup_failed(
    runs_folder: Path = MODELS_DIR / "experiments",
    *,
    execute: bool = False,
) -> None:
    """Find and optionally delete failed/skipped experiment directories.

    A run is considered 'failed' if its directory contains only a single error JSON
    (e.g., ValueError.json) and optionally a config.yml and run.log.

    Args:
        runs_folder (Path): The base directory to scan for experiments.
            Defaults to MODELS_DIR / "experiments".
        execute (bool): If set, the directories will be deleted. By default, this is a dry run
            that only lists the directories.
    """
    cleanup_failed_runs(runs_folder, dry_run=not execute)


@app.command
def cleanup_runs(
    top_k_to_keep: int = 50,
    rules_config_path: Path = MODELS_DIR / "best_model_rules.yml",
    base_dir: Path = MODELS_DIR,
    ignore_dir_names: set[str] = {"best"},
    ignore_suffix: str = ".old",
    files_to_keep: set[str] = {"metadata.json", "config.yml"},
    *,
    execute: bool = False,
) -> None:
    """Clean up worst-performing runs to save disk space, keeping metadata and config.

    For each model group (e.g., 'CTNet', 'EEGNetv4'), this command finds all completed runs, ranks
    them by the specified metric, and identifies all runs except for the top K best. For these
    identified runs, it deletes their contents except for the files specified in `files_to_keep`.

    Args:
        top_k_to_keep (int): The number of best-performing runs to keep untouched
            for each model group.
        rules_config_path (Path): Path to the YAML file defining which metric to use for ranking.
        base_dir (Path): The root directory to scan for experiments (e.g., 'models/').
        ignore_dir_names (set[str]): Name of directories to ignore from the cleaning operation. For
            instance, the 'best' directory, where the best runs are symlinked by `link_best`.
        ignore_suffix (str): Suffix for top-level directories to ignore (e.g., '.old').
        files_to_keep (set[str]): A set of filenames to preserve inside the runs
            that will be archived.
        execute (bool): If False (default), performs a dry run, printing what would be
            deleted. If True, performs the actual file deletion after confirmation.
    """

    class CleanupTask(TypedDict):
        """Hold the information for a cleanup task for a single model group."""

        group_name: str
        total_dirs: int
        completed_runs: int
        runs_to_archive: list[Path]
        space_to_free: int

    if not execute:
        logger.warning("--- DRY RUN MODE---")
        logger.warning("No files will be deleted. Run with --execute to perform deletion.")

    logger.info(
        f"Scanning '{base_dir}' to archive runs, keeping the top {top_k_to_keep} per group."
    )

    try:
        rules_config = load_yaml(rules_config_path)
        rules = {rule["folder_name"]: rule["metric"] for rule in rules_config.get("rules", [])}
        default_metric = rules_config.get("defaults", {}).get("metric", "val_loss")
    except FileNotFoundError:
        logger.error(f"Rules config file not found at: {rules_config_path}")
        return

    exp_folders = [
        d
        for d in base_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and not d.name.endswith(ignore_suffix)
        and d.name not in ignore_dir_names
    ]

    all_cleanup_tasks = []
    total_space_to_free = 0

    for exp_folder in exp_folders:
        metric_to_use = rules.get(exp_folder.name, default_metric)
        logger.info(f"--- Analyzing '{exp_folder.name}' (ranking by: '{metric_to_use}') ---")

        for model_dir in exp_folder.iterdir():
            if not model_dir.is_dir():
                continue

            all_run_dirs = [
                d for d in model_dir.iterdir() if d.is_dir() and re.match(r"^\d{6}_", d.name)
            ]
            total_dirs_count = len(all_run_dirs)

            completed_metadata_paths = [
                run_dir / "metadata.json"
                for run_dir in all_run_dirs
                if get_run_status(run_dir) == RunStatus.COMPLETED
                and (run_dir / "metadata.json").exists()
            ]

            if not completed_metadata_paths:
                continue

            # find_best_model_in_experiments works by scanning the directory, which is inefficient
            # here. We already have the metadata paths, so we'll rank them manually.
            lower_is_better = "loss" in metric_to_use.lower()
            ranked_runs = []
            for meta_path in completed_metadata_paths:
                try:
                    metadata = load_json(meta_path)
                    metric_history = metadata.get("history", {}).get(metric_to_use, [])
                    if metric_history:
                        best_value = (
                            min(metric_history) if lower_is_better else max(metric_history)
                        )
                        ranked_runs.append({"path": meta_path, "best_value": best_value})
                except Exception as e:
                    logger.warning(
                        f"Could not process metadata for ranking {meta_path.parent.name}: {e}"
                    )

            ranked_runs.sort(key=lambda x: x["best_value"], reverse=not lower_is_better)
            all_runs_sorted_paths = [run["path"] for run in ranked_runs]

            if len(all_runs_sorted_paths) <= top_k_to_keep:
                continue

            runs_to_archive_metadata = all_runs_sorted_paths[top_k_to_keep:]

            group_cleanup_info = CleanupTask(
                group_name=f"📁 {exp_folder.name}/{model_dir.name}",
                total_dirs=total_dirs_count,
                completed_runs=len(all_runs_sorted_paths),
                runs_to_archive=[],
                space_to_free=0,
            )

            for metadata_path in runs_to_archive_metadata:
                run_dir = metadata_path.parent
                cleanup_size = calculate_dir_size(run_dir, ignore=files_to_keep)
                total_space_to_free += cleanup_size
                group_cleanup_info["space_to_free"] += cleanup_size
                group_cleanup_info["runs_to_archive"].append(run_dir)

            if group_cleanup_info["runs_to_archive"]:
                all_cleanup_tasks.append(group_cleanup_info)

    if not all_cleanup_tasks:
        logger.success("✅ No runs found that meet the criteria for cleanup.")
        return

    table = PrettyTable()
    table.field_names = [
        "Model Group",
        "Total Dirs",
        "Completed",
        "To Keep",
        "To Archive",
        "Total After",
        "Space to Free",
    ]
    for field in table.field_names:
        table.align[field] = "r"
    table.align["Model Group"] = "l"

    for task in all_cleanup_tasks:
        table.add_row(
            [
                task["group_name"],
                task["total_dirs"],
                task["completed_runs"],
                top_k_to_keep,
                len(task["runs_to_archive"]),
                task["total_dirs"],  # Total After is the same as Total Dirs
                format_size_bytes(task["space_to_free"]),
            ]
        )

    print(table)
    logger.info(f"Total estimated space to be freed: {format_size_bytes(total_space_to_free)}")

    if execute:
        confirm = input(
            f"This will archive {sum(len(t['runs_to_archive']) for t in all_cleanup_tasks)} "
            "run directories. Are you sure you want to delete their contents? (y/N): "
        )
        if confirm.lower() == "y":
            logger.warning("Proceeding with file deletion...")

            total_archived_count = 0
            for task in tqdm(all_cleanup_tasks, desc="Archiving model groups"):
                for run_dir in tqdm(
                    task["runs_to_archive"],
                    desc=f"  - Archiving {task['group_name']}",
                    leave=False,
                ):
                    try:
                        for item in run_dir.iterdir():
                            if item.name not in files_to_keep:
                                if item.is_file() or item.is_symlink():
                                    item.unlink()
                                elif item.is_dir():
                                    shutil.rmtree(item)
                        total_archived_count += 1
                    except Exception as e:
                        logger.error(f"Failed to clean directory {run_dir}: {e}")

            logger.success(
                f"✅ Archiving complete. Processed {total_archived_count} run directories."
            )
        else:
            logger.info("Archiving cancelled by user.")
    else:
        logger.info("This was a dry run. No files were deleted.")


@app.default
def run(
    config_path: Path,
    *,
    pick: int = -1,
    debug: bool = False,
    force_new: bool = False,
) -> None:
    """Run a complete training experiment based on a YAML configuration file.

    If the config file contains lists for hyperparameters, a grid search
    will be performed, running an experiment for each combination.

    Args:
        config_path (Path): The path to the experiment's .yaml configuration file.
        pick (int, optional): If > 0, only runs the first `pick` configurations
            generated from the grid search. Defaults to -1 (run all).
        debug (bool, optional): Whether to enable debug mode for console-level logging.
            Defaults to False.
        force_new (bool, optional): If True, forces the execution of a new run even if an
            identical configuration has been run before. Defaults to False.
    """
    # 1. Load root configuration
    logger.info(f"Loading experiment configuration from: {config_path}")
    with config_path.open("r") as f:
        root_config = yaml.safe_load(f)

    # --- Handle Config Inheritance ---
    # In the case that `from_config` key exists in root_config, we need to merge it
    root_config = _handle_config_inheritance(root_config)

    # 2. Generate all experiment configurations from the root config
    configs_to_run = generate_grid_search_configs(root_config)

    # If pick is set, select only that amount of configurations
    if pick > 0:
        configs_to_run = configs_to_run[:pick]
        logger.info(
            f"Pick is set to {pick}. Running only the first {len(configs_to_run)} configs."
        )

    num_experiments = len(configs_to_run)
    logger.info(f"Generated {num_experiments} experiment(s) from the configuration file.")

    # Get the models_subdir key to save the run in a subdirectory, e.g, "experiments/EEGNetv4"
    models_subdir = root_config.get("models_subdir", "experiments")

    runs_folder = MODELS_DIR / models_subdir
    runs_folder.mkdir(parents=True, exist_ok=True)

    # Randomly shuffle the configurations to ensure varied execution order
    shuffle(configs_to_run)

    # 3. Loop through and execute each experiment
    for i, single_config in enumerate(configs_to_run, 1):
        logger.success("-" * 80)
        logger.success(" " * 10 + f"Running Experiment {i}/{num_experiments}" + " " * 10)
        logger.success("-" * 80)

        # Check that the current config.yml hasn't been ran before
        formatted_name = _format_experiment_name(single_config)
        logger.info(f"Checking config: {formatted_name}")

        if not force_new:
            existing_run = check_run_exists_by_config(
                runs_folder,
                new_config=single_config,
                ignore_keys=[
                    "experiment_name",
                    "models_subdir",
                    "delete_when_error",
                    "from_config",
                    "training_params.plot_kwargs",
                    "training_params.save_train_test_data",
                    "training_params.patience",
                    "training_params.plot_every_epoch",
                ],
            )

            if existing_run:
                logger.warning(f"Skipping run. Identical run found at: {existing_run}")
                continue  # Skip to the next config

        try:
            _run_single_experiment(single_config, runs_folder, debug=debug, force_new=force_new)
        except Exception as e:
            # This top-level catch is a safeguard. The inner one should handle logging.
            logger.critical(
                f"A critical unhandled error occurred in experiment {i}/{num_experiments}."
            )
            logger.exception(e)

    logger.info("--- All experiments concluded ---")


if __name__ == "__main__":
    app()
