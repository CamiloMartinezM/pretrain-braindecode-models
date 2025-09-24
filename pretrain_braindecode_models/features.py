"""Contains functions to compile, segment and extract features from EEG."""

from pathlib import Path

from cyclopts import App

from pretrain_braindecode_models.config import TUH_DATA_DIR
from pretrain_braindecode_models.dataset import TUHAbnormal

app = App()


def prepare_tuh_dataset(
    tuh_data_dir: Path = TUH_DATA_DIR,
    *,
    eeg_channels: list[str] | None = None,
    eeg_window_seconds: float = 2.0,
    stride_seconds: float = 0.5,
    sfreq: float = 250.0,
    n_train_files: int | None = None,
    n_eval_files: int | None = None,
    pad_end: bool = True,
    pad_mode: str = "reflect",
) -> tuple[
    TUHAbnormal,  # train_dataset
    TUHAbnormal,  # test_dataset
]:
    """Prepare the TUH Abnormal EEG dataset for training.

    Args:
        tuh_data_dir (Path): Root directory of the TUH Abnormal dataset.
        eeg_channels (list[str] | None): List of EEG channel names to use. If None, uses all
            available channels. Default is None.
        eeg_window_seconds (float): Duration of EEG windows.
        stride_seconds (float): Stride between consecutive windows.
        sfreq (float): Target sampling frequency.
        n_train_files (int | None): Number of training files to use (for subsetting) for each
            class (abnormal/normal). Default is None, which uses all files.
        n_eval_files (int | None): Number of evaluation files to use (for subsetting) for each
            class (abnormal/normal). Default is None, which uses all files.
        pad_end (bool): If True, pads the end of the EEG signal to extract additional windows.
        pad_mode (str): The padding mode to use (e.g., 'reflect', 'constant').

    Returns:
        A tuple containing the training dataset and testing dataset.
    """
    train_dataset = TUHAbnormal(
        root_path=tuh_data_dir,
        eeg_channels=eeg_channels,
        montage_name="standard_1020",
        split="train",
        window_seconds=eeg_window_seconds,
        stride_seconds=stride_seconds,
        sfreq=sfreq,
        n_files_per_class=n_train_files,
        pad_end=pad_end,
        pad_mode=pad_mode,
    )

    test_dataset = TUHAbnormal(
        root_path=tuh_data_dir,
        eeg_channels=eeg_channels,
        montage_name="standard_1020",
        split="eval",
        window_seconds=eeg_window_seconds,
        stride_seconds=stride_seconds,
        sfreq=sfreq,
        n_files_per_class=n_eval_files,
        pad_end=pad_end,
        pad_mode=pad_mode,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    app()
