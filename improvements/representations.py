"""
Ways of turning a 2.1 s recording (263 samples at 125 Hz) into the input of
a model pre-trained on 10 s segments (1250 samples). Experiment 1 compares
them; everything else uses zero_pad, the paper's choice.

  zero_pad      centred zero padding to 1250 (paper / notebook)
  native        no padding (the encoder averages over time, so any length
                works)
  tile          repeat the recording end to end until it is 1250 long
  tile_cycles   like tile, but cut at systolic peaks first so every join
                falls at the same point of the heartbeat
  reflect       centred mirror padding
  concat3_tile  join the participant's 3 recordings, then tile to 1250 ->
                one input per participant instead of three
"""

import numpy as np
from data import FS_TARGET
from data import MODEL_INPUT_LENGTH as L
from features import find_systolic_peaks


def zero_pad(x):
    need = L - len(x)
    left = need // 2
    return np.pad(x, (left, need - left))  # identical to the notebook


def native(x):
    return x


def tile(x):
    return np.resize(x, L)  # np.resize repeats the array cyclically


def tile_cycles(x):
    """Tile whole cardiac cycles: cut from the first to the last systolic
    peak so the joins fall at the same point of the beat (no jump)."""
    peaks = find_systolic_peaks(x, FS_TARGET)
    if len(peaks) < 2:
        return tile(x)
    return np.resize(x[peaks[0]:peaks[-1]], L)


def reflect(x):
    need = L - len(x)
    left = need // 2
    return np.pad(x, (left, need - left), mode="symmetric")


PER_RECORDING = {
    "zero_pad": zero_pad,
    "native": native,
    "tile": tile,
    "tile_cycles": tile_cycles,
    "reflect": reflect,
}
PER_PARTICIPANT = {
    "concat3_tile": lambda recs: tile(np.concatenate(recs)),
}
ALL_VARIANTS = list(PER_RECORDING) + list(PER_PARTICIPANT)


def build_inputs(variant, recordings, subject_ids):
    """Model inputs for a variant: (list of inputs, list of owner IDs).

    Per-recording variants give three inputs per participant, the
    per-participant variant gives one."""
    inputs, owners = [], []
    for sid in subject_ids:
        recs = recordings[int(sid)]
        if variant in PER_RECORDING:
            for r in recs:
                x = PER_RECORDING[variant](r)
                assert variant == "native" or len(x) == L
                inputs.append(x.astype(np.float32))
                owners.append(int(sid))
        else:
            inputs.append(PER_PARTICIPANT[variant](recs).astype(np.float32))
            owners.append(int(sid))
    return inputs, owners
