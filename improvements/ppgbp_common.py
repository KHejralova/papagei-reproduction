"""
Shared helpers for the improvement experiments on PPG-BP.

Everything here mirrors the reproduction notebook (01_ppgbp_reprod.ipynb)
so that improvement results are directly comparable with Kaja's numbers:

- same metadata file and column names,
- same supplied 205-participant split (123 train / 41 val / 41 test),
- same per-recording preprocessing (drop last sample, z-score,
  pyPPG Chebyshev bandpass 0.5-12 Hz, resample 1000 -> 125 Hz),
- same model configurations and output index (outputs[0]),
- same participant-level aggregation (mean of the 3 recording embeddings).

The only thing the experiments change is what is done *after* preprocessing
(e.g. how a 2.1 s recording is turned into the model input).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linearprobing.utils import (
    load_model_without_module_prefix,
    resample_batch_signal,
)
from models.resnet import ResNet1D, ResNet1DMoE
from preprocessing.ppg import preprocess_one_ppg_signal

FS_ORIGINAL = 1000
FS_TARGET = 125
MODEL_INPUT_LENGTH = 1250  # 10 s at 125 Hz, the length used in pre-training
RECORDINGS_PER_PARTICIPANT = 3

# Supplied split, copied verbatim from the reproduction notebook (cell 17).
TRAIN_IDS = [
    2, 6, 8, 10, 12, 15, 16, 17, 18, 19, 22, 23, 26, 31, 32, 34, 35, 38, 40,
    45, 48, 50, 53, 55, 56, 58, 60, 61, 63, 65, 66, 83, 85, 87, 89, 92, 93,
    97, 98, 99, 100, 104, 105, 106, 107, 112, 113, 114, 116, 120, 122, 126,
    128, 131, 134, 135, 137, 138, 139, 140, 141, 146, 148, 149, 152, 153, 154,
    158, 160, 162, 164, 165, 167, 169, 170, 175, 176, 179, 183, 184, 186, 188,
    189, 190, 191, 193, 196, 197, 199, 205, 206, 207, 209, 210, 212, 216, 217,
    218, 223, 226, 227, 230, 231, 233, 234, 240, 242, 243, 244, 246, 247, 248,
    256, 257, 404, 407, 409, 412, 414, 415, 416, 417, 419,
]
VAL_IDS = [
    3, 11, 24, 27, 29, 30, 41, 43, 47, 64, 88, 91, 95, 115, 125, 127, 136,
    145, 155, 156, 161, 163, 166, 178, 198, 203, 208, 213, 215, 222, 229, 232,
    235, 237, 241, 245, 252, 254, 259, 411, 418,
]
TEST_IDS = [
    14, 21, 25, 51, 52, 62, 67, 86, 90, 96, 103, 108, 110, 119, 123, 124, 130,
    142, 144, 157, 172, 173, 174, 180, 182, 185, 192, 195, 200, 201, 211, 214,
    219, 221, 228, 239, 250, 403, 405, 406, 410,
]
SPLIT_IDS = {"train": TRAIN_IDS, "val": VAL_IDS, "test": TEST_IDS}

COLUMN_RENAMES = {
    "Sex(M/F)": "sex",
    "Age(year)": "age",
    "Height(cm)": "height",
    "Weight(kg)": "weight",
    "Systolic Blood Pressure(mmHg)": "sysbp",
    "Diastolic Blood Pressure(mmHg)": "diasbp",
    "Heart Rate(b/m)": "hr",
    "BMI(kg/m^2)": "bmi",
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def data_file_dir(data_dir):
    """Folder that contains 'PPG-BP dataset.xlsx' and '0_subject/'."""
    path = Path(data_dir) / "Data File"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'. Point --data-dir at the folder that "
            "contains the unzipped 'Data File' folder from figshare "
            "(the same folder as download_dir in the reproduction notebook)."
        )
    return path


def load_metadata(data_dir):
    """PPG-BP spreadsheet with the same renames as the notebook."""
    df = pd.read_excel(
        data_file_dir(data_dir) / "PPG-BP dataset.xlsx", header=1
    )
    df = df.rename(columns=COLUMN_RENAMES)
    return df


def split_metadata(df):
    """Return {'train': df, 'val': df, 'test': df} using the supplied split.

    Rows are sorted by subject_ID so features and targets always line up.
    """
    splits = {}
    for name, ids in SPLIT_IDS.items():
        part = df[df["subject_ID"].isin(ids)].sort_values("subject_ID")
        part = part.drop_duplicates(subset="subject_ID").reset_index(drop=True)
        if len(part) != len(ids):
            missing = sorted(set(ids) - set(part["subject_ID"]))
            raise ValueError(f"{name}: missing participants {missing}")
        splits[name] = part
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) \
        == (123, 41, 41)
    return splits


def load_raw_recording(data_dir, subject_id, recording):
    """One raw 2.1 s recording, with the last sample dropped as in the notebook."""
    path = data_file_dir(data_dir) / "0_subject" / f"{subject_id}_{recording}.txt"
    signal = pd.read_csv(path, sep="\t", header=None).values.squeeze()
    # Exactly as the notebook: drop the last entry (the files end with a
    # trailing tab, which pandas reads as an empty value).
    signal = np.asarray(signal[:-1], dtype=float)
    if np.isnan(signal).any():
        raise ValueError(f"NaN values left in {path} after dropping last entry")
    return signal


def zscore(signal):
    """Equivalent to torch_ecg Normalize(method='z-score') with defaults."""
    return (signal - signal.mean()) / signal.std()


def preprocess_recording(raw_signal):
    """Notebook order: z-score -> pyPPG bandpass -> resample to 125 Hz."""
    signal = zscore(raw_signal)
    signal, _, _, _ = preprocess_one_ppg_signal(
        waveform=signal, frequency=FS_ORIGINAL
    )
    signal = resample_batch_signal(
        signal, fs_original=FS_ORIGINAL, fs_target=FS_TARGET, axis=0
    )
    return np.asarray(signal, dtype=np.float32)


def load_preprocessed_recordings(data_dir, subject_ids):
    """{subject_id: [rec1, rec2, rec3]} of preprocessed 125 Hz signals."""
    out = {}
    for sid in subject_ids:
        out[int(sid)] = [
            preprocess_recording(load_raw_recording(data_dir, sid, r))
            for r in range(1, RECORDINGS_PER_PARTICIPANT + 1)
        ]
    check_preprocessing(data_dir, subject_ids)
    return out


_FILTER_CHECKED = False


def check_preprocessing(data_dir, subject_ids, n_participants=10):
    """Stop if pyPPG's band-pass filter is numerically unstable here.

    pyPPG designs its 4th-order Chebyshev II band-pass (0.5-12 Hz at 1000 Hz)
    in transfer-function (b, a) form and applies it with filtfilt. With such
    a low cut-off relative to the sampling rate, that form is numerically
    fragile: depending on the SciPy build, the output can blow up and the
    pulse shape is lost, silently corrupting every feature and embedding.
    (Found when installing aeon downgraded SciPy 1.18 -> 1.17 on Linux.)

    The check filters a few recordings both with pyPPG and with the same
    filter in the numerically stable second-order-sections (SOS) form, and
    requires the two to agree.
    """
    global _FILTER_CHECKED
    if _FILTER_CHECKED:
        return
    import scipy
    from scipy import signal
    sos = signal.cheby2(4, 20, [0.5, 12], "bandpass", fs=FS_ORIGINAL,
                        output="sos")
    win = round(FS_ORIGINAL * 50 / 1000)  # pyPPG's 50 ms smoothing window
    corrs = []
    for sid in list(subject_ids)[:n_participants]:
        for r in range(1, RECORDINGS_PER_PARTICIPANT + 1):
            z = zscore(load_raw_recording(data_dir, sid, r))
            ppg, _, _, _ = preprocess_one_ppg_signal(waveform=z,
                                                     frequency=FS_ORIGINAL)
            ref = signal.filtfilt(np.ones(win) / win, 1,
                                  signal.sosfiltfilt(sos, z))
            corrs.append(np.corrcoef(ppg, ref)[0, 1])
    if np.median(corrs) < 0.99:
        raise RuntimeError(
            "pyPPG's band-pass filter is numerically unstable with this "
            f"SciPy build ({scipy.__version__}): its output matches the "
            f"stable version with median correlation {np.median(corrs):.2f} "
            "(should be about 1.00). Every feature computed now would be "
            "wrong. Check your SciPy version (the run that produced the "
            "earlier results is fine), and delete "
            "data_PPG/features_improvements/ if it was filled in this "
            "environment.")
    _FILTER_CHECKED = True


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

MODEL_FILES = {
    "papagei_s": "papagei_s.pt",
    "papagei_s_svri": "papagei_s_svri.pt",
    "papagei_p": "papagei_p.pt",
}


def build_model(model_name, weights_dir="weights", device="cpu",
                load_weights=True):
    """Same configurations as the reproduction notebook (cells 29, 33, 36)."""
    common = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
                  "groups": 1, "n_block": 18, "n_classes": 512}
    if model_name == "papagei_s":
        model = ResNet1DMoE(n_experts=3, **common)
    elif model_name == "papagei_s_svri":
        model = ResNet1D(use_mt_regression=False, use_projection=False,
                         **common)
    elif model_name == "papagei_p":
        model = ResNet1D(**common)
    else:
        raise ValueError(f"Unknown model {model_name}")

    if load_weights:
        path = Path(weights_dir) / MODEL_FILES[model_name]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing weights '{path}'. Download {MODEL_FILES[model_name]} "
                "from https://zenodo.org/records/13983110 into weights/."
            )
        model = load_model_without_module_prefix(model, str(path))
    model.to(device)
    model.eval()
    return model


def embed_signals(model, signals, device="cpu", batch_size=256):
    """Embed a list of 1D signals; returns (n_signals, 512) array.

    Signals of different lengths are embedded in separate batches, which is
    fine because the encoder averages over time before its last layer.
    Uses outputs[0], the same output the notebook uses.
    """
    embeddings = [None] * len(signals)
    by_length = {}
    for i, s in enumerate(signals):
        by_length.setdefault(len(s), []).append(i)

    with torch.inference_mode():
        for idx in by_length.values():
            for start in range(0, len(idx), batch_size):
                chunk = idx[start:start + batch_size]
                x = torch.tensor(np.stack([signals[i] for i in chunk]),
                                 dtype=torch.float32)
                x = x.unsqueeze(1).to(device)
                out = model(x)[0].cpu().numpy()
                for j, i in enumerate(chunk):
                    embeddings[i] = out[j]
    return np.stack(embeddings)
