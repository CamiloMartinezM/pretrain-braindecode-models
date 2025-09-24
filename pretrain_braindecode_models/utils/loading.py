"""Provides utility functions for loading and manipulating different sources."""

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import yaml

from pretrain_braindecode_models.config import logger
from pretrain_braindecode_models.utils.misc import make_json_serializable


def load_json(
    file_path: Path | str,
    exclude_keys: set[str] | None = None,
    *,
    warn: bool = False,
    raise_error: bool = False,
) -> dict:
    """Load annotations from a JSON file.

    Args:
        file_path (str): Path to the JSON file
        exclude_keys (set[str] | None): Keys to exclude from the loaded data
        warn (bool): Whether to print a warning if the file is not found
        raise_error (bool): Whether to raise an error if the file is not found

    Returns:
        dict: Loaded data from the JSON file or an empty dictionary if not found

    Raises:
        FileNotFoundError: If the file is not found and raise_error is True
    """
    result = {}
    if Path(file_path).exists():
        with Path(file_path).open("r") as f:
            result = json.load(f)
    elif raise_error:
        raise FileNotFoundError(f"File not found: {file_path}")
    elif warn:
        logger.warning(f"File not found: {file_path}")

    if exclude_keys:
        for key in exclude_keys:
            if key in result:
                del result[key]

    return result


def save_json(
    data: dict,
    file_path: Path | str,
    dump_args: dict | None = None,
    *,
    serialize: bool = True,
) -> None:
    """Save annotations to a JSON file.

    Args:
        data (dict): The data to save
        file_path (Path | str): Path to save the JSON file
        dump_args (dict | None): Additional arguments for json.dump, if needed. For example,
            `{"sort_keys": True}` to sort the keys in the JSON file. Defaults to None.
        serialize (bool): Whether to make the data JSON serializable
    """
    # Ensure the directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Create backup before saving
    create_backup(file_path)

    if serialize:
        data = make_json_serializable(data, ignore_non_serializable=False, warn=True)

    with Path(file_path).open("w") as f:
        # Pretty printing with consistent indentation
        json.dump(data, f, indent=4, **(dump_args or {}))


def load_yaml(
    file_path: Path | str,
    *,
    warn: bool = False,
    raise_error: bool = False,
) -> dict:
    """Load annotations from a YAML file.

    Args:
        file_path (str): Path to the YAML file
        warn (bool): Whether to print a warning if the file is not found
        raise_error (bool): Whether to raise an error if the file is not found

    Returns:
        dict: Loaded data from the YAML file or an empty dictionary if not found

    Raises:
        FileNotFoundError: If the file is not found and raise_error is True
    """
    result = {}
    if Path(file_path).exists():
        with Path(file_path).open("r") as f:
            result = yaml.safe_load(f)
    elif raise_error:
        raise FileNotFoundError(f"File not found: {file_path}")
    elif warn:
        logger.warning(f"File not found: {file_path}")

    return result


def create_backup(file_path: Path | str) -> None:
    """Create a backup of a file with .bak extension.

    Args:
        file_path (str): Path to the file to backup
    """
    if Path(file_path).exists():
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)


def restore_from_backup(file_path: Path | str) -> bool:
    """Restore a file from its backup (.bak) and make the original file the new backup.

    Args:
        file_path (Path | str): Path to the file to restore

    Returns:
        bool: True if restoration was successful, False otherwise
    """
    backup_path = f"{file_path}.bak"

    # Check if backup exists
    if not Path(backup_path).exists():
        logger.info(f"No backup file found at {backup_path}")
        return False

    # Create a new backup of the current file
    try:
        if Path(file_path).exists():
            temp_path = f"{file_path}.tmp"
            shutil.copy2(file_path, temp_path)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Restore the backup to the original filename
        shutil.copy2(backup_path, file_path)

        # Replace the old backup with the temp file
        if Path(temp_path).exists():
            shutil.move(temp_path, backup_path)
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        return False
    else:
        return True


def load_from_pickle(
    path: str | Path,
    *,
    trigger_warning: bool = False,
    trigger_exception: bool = False,
    verbose: bool = True,
) -> Any:  # noqa: ANN401
    """Load an object from a pickle file given the `filename`, inside the folder `folder`.

    If the file does not exist,it will return `None`, and a warning or an exception can be
    triggered, if specified.

    Args:
        path (str | Path): The path to the folder containing the pickle file.
        trigger_warning (bool): If True, a warning will be logged if the file does not exist.
            Default is False.
        trigger_exception (bool): If True, an exception will be raised if the file does not exist.
            Default is False.
        verbose (bool): If True, print loading status. Default is True.

    Returns:
        Any: The loaded object from the pickle file, or None if the file does not exist.
    """
    path = Path(path)
    if path.exists():
        if verbose:
            logger.info(f"Loading {path} ... ", end="")
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    if trigger_warning:
        logger.warning(f"{path} does not exist, returning None instead")
    elif trigger_exception:
        raise FileNotFoundError(f"{path} does not exist")
    return None


def save_as_pickle(
    path: str | Path,
    obj: Any,  # noqa: ANN401
    *,
    overwrite: bool = True,
) -> None:
    """Save an object as a pickle file.

    Args:
        path (str | Path): The path where the pickle file will be saved (including .pkl extension).
        obj (Any): The object to be saved as a pickle file.
        overwrite (bool, optional): If True, the pickle file will be overwritten if it already
            exists. Otherwise, it won't, by default True
    """
    path = Path(path)

    # Ensure the destination directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists
    if path.exists() and not overwrite:
        logger.warning(f"The file {path} already exists and it won't be overwritten.")
        return

    # Save the object as a pickle file
    with path.open("wb") as f:
        pickle.dump(obj, f)


def get_video_file(
    folder_path: Path,
    filename: str | None = None,
    video_extensions: set[str] = {".mp4", ".MP4", ".avi", ".mov", ".MOV"},
    *,
    warn: bool = False,
    raise_error: bool = False,
) -> Path | None:
    """Find the first video file in a folder.

    Args:
        folder_path (Path): Path to the folder
        filename (str | None): Specific filename to look for (without the extension). If None, it
            will look for any video file. Default is None.
        video_extensions (set[str]): Set of video file extensions to look for
            Default is {".mp4", ".MP4", ".avi", ".mov", ".MOV"}
        warn (bool): Whether to print a warning if no video file is found
        raise_error (bool): Whether to raise an error if no video file is found

    Returns:
        Path: Path to the video file or None if not found
    """
    for f in folder_path.iterdir():
        # Check if the filename matches (without extension)
        if (
            f.is_file()
            and f.suffix in video_extensions
            and ((filename is not None and f.stem == filename) or filename is None)
        ):
            return f

    if raise_error:
        raise FileNotFoundError(
            f"No video file found in {folder_path} with any of the extensions {video_extensions}",
        )
    if warn:
        logger.warning(
            f"No video file found in {folder_path} with any of the extensions {video_extensions}",
        )

    return None
