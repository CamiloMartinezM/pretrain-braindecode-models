"""Custom PyTorch Dataset for the TUH Abnormal EEG Corpus."""

import random
from pathlib import Path

import mne
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from pretrain_braindecode_models.utils import eeg
from pretrain_braindecode_models.utils.folders import find_files

# Mapping from old TUH names to the standard 10-10 names MNE uses
TUH_TO_STANDARD_CHANNEL_MAP = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


class TUHAbnormal(Dataset):
    """PyTorch Dataset for the Temple University Hospital (TUH) Abnormal EEG Corpus.

    This class handles scanning the directory, loading EDF files, creating sliding
    windows, and providing (X, y) pairs for training.
    """

    def __init__(
        self,
        root_path: str | Path,
        *,
        eeg_channels: list[str] | None = None,
        montage_name: str = "standard_1020",
        split: str = "train",
        window_seconds: float = 1.0,
        stride_seconds: float = 0.5,
        sfreq: float = 250.0,
        n_files_per_class: int | None = None,
        pad_end: bool = True,
        pad_mode: str = "reflect",
        preload: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_path (str | Path): Path to the root of the TUH Abnormal dataset
                (the directory containing the 'edf' folder).
            split (str): Which data split to use ('train' or 'eval').
            window_seconds (float): Duration of each EEG window in seconds.
            stride_seconds (float): The step size between consecutive windows in seconds.
            sfreq (float): The target sampling frequency to resample all signals to.
            n_files_per_class (int | None): If specified, uses only the first N files from each
                class (abnormal/normal) for quick testing. If None, uses all files.
            pad_end (bool): If True, pads the end of the EEG signal to extract additional windows.
            pad_mode (str): The padding mode to use (e.g., 'reflect', 'constant').
            preload (bool): If True, loads all data into memory upon initialization.
        """
        self.root_path = Path(root_path)
        self.montage_name = montage_name
        self.eeg_channels = (
            eeg_channels or mne.channels.make_standard_montage(montage_name).ch_names
        )
        self.split = split
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.target_sfreq = sfreq
        self.pad_end = pad_end
        self.pad_mode = pad_mode
        self.preload = preload
        self.n_files_per_class = n_files_per_class

        self.edf_files = self._find_files(n_files_per_class)

        self.windows, self.labels = self._create_windows()

    def _find_files(self, n_files_per_class: int | None) -> list[Path]:
        """Find all .edf files for the specified split."""
        split_path = self.root_path / self.split
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {split_path}")

        logger.info(f"Scanning for .edf files in {split_path}...")

        abnormal_path = split_path / "abnormal"
        normal_path = split_path / "normal"

        if not abnormal_path.exists() or not normal_path.exists():
            raise FileNotFoundError(
                f"Expected 'abnormal' and 'normal' subdirectories in {split_path}."
            )

        abnormal_files = find_files(abnormal_path, extensions=".edf")
        normal_files = find_files(normal_path, extensions=".edf")

        # Shuffle for random subset selection
        random.shuffle(abnormal_files)
        random.shuffle(normal_files)

        selected_abnormal = abnormal_files
        selected_normal = normal_files

        if n_files_per_class is not None:
            logger.info(
                f"Selecting up to {n_files_per_class} files per class "
                f"for the '{self.split}' split."
            )
            selected_abnormal = abnormal_files[:n_files_per_class]
            selected_normal = normal_files[:n_files_per_class]

            if len(selected_abnormal) < n_files_per_class:
                logger.warning(
                    f"Found only {len(selected_abnormal)} abnormal files "
                    f"(requested {n_files_per_class})."
                )
            if len(selected_normal) < n_files_per_class:
                logger.warning(
                    f"Found only {len(selected_normal)} normal files "
                    f"(requested {n_files_per_class})."
                )

        # Combine and shuffle the final list
        final_files = selected_abnormal + selected_normal
        random.shuffle(final_files)

        logger.success(
            f"Found {len(final_files)} total files for the '{self.split}' split "
            f"({len(selected_abnormal)} abnormal, {len(selected_normal)} normal)."
        )

        return [Path(f) for f in final_files]

    def _create_windows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load data, preprocess, and create sliding windows."""
        all_windows = []
        all_labels = []

        window_size = int(self.window_seconds * self.target_sfreq)
        stride_size = int(self.stride_seconds * self.target_sfreq)

        # Create the standard montage object we will enforce on all files
        standard_montage = eeg.create_dig_montage(
            self.eeg_channels,
            montage_name=self.montage_name,
            uppercase=True,
        )

        # Create a standard info object from this montage
        standard_info = mne.create_info(
            ch_names=standard_montage.ch_names, sfreq=self.target_sfreq, ch_types="eeg"
        )
        standard_info.set_montage(standard_montage)

        num_channels = None  # To be determined from the first file

        for edf_path in tqdm(self.edf_files, desc=f"Processing {self.split} files"):
            try:
                # 1. Load and Preprocess EDF
                logger.debug(f"Loading {edf_path.name}")
                raw = mne.io.read_raw_edf(edf_path, preload=self.preload, verbose=False)

                logger.debug(f"\tOriginal channels ({len(raw.ch_names)}): {raw.ch_names}")

                # --- Channel Unification Logic ---
                # Rename channels to a standard convention (e.g., 'EEG FP1-REF' -> 'FP1')
                rename_map = {}
                for ch_name in raw.ch_names:
                    # Try to parse the standard "EEG XX-REF" format
                    parts = ch_name.split(" ")
                    if len(parts) > 1 and parts[0] == "EEG":
                        core_name = parts[1].split("-")[0]

                        # Map old names (T3, T4, etc.) to new standard names (T7, T8, etc.)
                        if core_name.upper() in TUH_TO_STANDARD_CHANNEL_MAP:
                            rename_map[ch_name] = TUH_TO_STANDARD_CHANNEL_MAP[core_name.upper()]
                        else:
                            # For non-standard names like 'EMG-REF', 'PHOTIC-REF', etc.,
                            # we keep them as-is for now. They will be dropped in the next step
                            # if they are not in our standard montage.
                            rename_map[ch_name] = core_name.upper()
                    else:
                        # If it doesn't match the expected pattern, keep the original name
                        # which will be dropped later if not in the montage
                        rename_map[ch_name] = ch_name.upper()

                raw.rename_channels(rename_map, allow_duplicates=True, verbose=False)

                # After renaming, some channels might be duplicates (e.g., EKG1 and EKG).
                # We can handle this by dropping them, as they are not in our standard montage
                ch_to_drop = [ch for ch in raw.ch_names if ch not in standard_montage.ch_names]
                if ch_to_drop:
                    logger.debug(f"\tDropping channels ({len(ch_to_drop)}): {ch_to_drop}")
                    raw.drop_channels(ch_to_drop, on_missing="ignore")

                # Add missing channels (as flat 0s) to match the standard montage
                # This ensures all data has the same number of channels in the same order
                missing_channels = [
                    mne.io.RawArray(
                        np.zeros((1, raw.n_times)),
                        mne.create_info([ch], raw.info["sfreq"], "eeg"),
                    )
                    for ch in standard_montage.ch_names
                    if ch not in raw.ch_names
                ]
                raw.add_channels(missing_channels, force_update_info=False)

                if missing_channels:
                    logger.debug(
                        f"\tAdded missing channels ({len(missing_channels)}): "
                        f"{[ch.ch_names[0] for ch in missing_channels]} as flat 0s"
                    )

                # Reorder channels to match the standard info object
                raw.reorder_channels(standard_info.ch_names)
                raw.set_montage(standard_montage)

                # Resample to target frequency if needed
                if raw.info["sfreq"] != self.target_sfreq:
                    logger.debug(
                        f"\tResampling from {raw.info['sfreq']} Hz to {self.target_sfreq} Hz"
                    )
                    raw.resample(self.target_sfreq, verbose=False)

                # Basic filtering
                raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)

                # Get data and label
                data = raw.get_data(return_times=False)  # (channels, time)
                label_str = edf_path.parent.parent.name  # 'abnormal' or 'normal'

                if label_str not in ["abnormal", "normal"]:
                    raise ValueError(f"Unexpected label '{label_str}' from path {edf_path}")

                label = 1 if label_str == "abnormal" else 0

                if not isinstance(data, np.ndarray):
                    raise TypeError(f"Data from {edf_path} is not a numpy array.")

                # Check the number of channels
                if num_channels is None:
                    num_channels = data.shape[0]

                if data.shape[0] != num_channels:
                    logger.warning(
                        f"Skipping file {edf_path.name} due to inconsistent channel count "
                        f"({data.shape[0]} vs expected {num_channels})."
                    )
                    continue

                # Padding logic
                num_samples_original = data.shape[1]

                if self.pad_end:
                    # Pad the end of the time dimension (axis 1)
                    # The amount of padding should be enough for the last window to be full
                    pad_width = ((0, 0), (0, window_size - 1))
                    data_padded = np.pad(data, pad_width=pad_width, mode=self.pad_mode)  # type: ignore[reportArgumentType, reportCallIssue, call-overload]
                else:
                    data_padded = data

                num_samples_padded = data_padded.shape[1]

                # 2. Create Sliding Windows
                # The loop now iterates over the potentially padded signal length
                for start in range(0, num_samples_original, stride_size):
                    end = start + window_size

                    # This check is now against the padded length, ensuring full windows
                    if end > num_samples_padded:
                        break

                    window = data_padded[:, start:end]

                    # Final safety check for window size
                    if window.shape[1] != window_size:
                        logger.warning(
                            f"Skipping window from {start} to {end} in file {edf_path.name} "
                            f"due to incorrect size {window.shape[1]} (expected {window_size})."
                        )
                        continue  # Should not happen with correct padding

                    all_windows.append(torch.from_numpy(window).float())
                    all_labels.append(label)

            except Exception as e:
                logger.warning(f"Skipping file {edf_path.name} due to error: {e}")
                raise  # NOTE: Remove

        all_windows_t = torch.stack(all_windows)
        all_labels_t = torch.tensor(all_labels, dtype=torch.long)

        # Get the counts of each class
        unique, counts = torch.unique(all_labels_t, return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist(), strict=False))

        self.classes = sorted(unique.tolist())
        self.class_labels = {0: "normal", 1: "abnormal"}

        logger.success(
            f"Created {tuple(all_windows_t.shape)} windows for the '{self.split}' split "
            f"with {tuple(all_labels_t.shape)} labels (classes: {self.classes})."
        )
        logger.debug(f"Class distribution in '{self.split}' split: {class_counts}")

        return all_windows_t, all_labels_t

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single `(X, y)` pair."""
        return self.windows[idx], self.labels[idx]
