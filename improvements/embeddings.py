"""
PaPaGei: loading the pre-trained models and turning recordings into 512-number embeddings, 
one row per participant (the mean over the participant's recordings, 
as in the reproduction notebook).

Embeddings are cached in <data-dir>/features_improvements/ per model 
and input representation, so each recording is only embedded once.
"""

from pathlib import Path

import numpy as np
import torch
from data import REPO_ROOT  # noqa: F401  (puts the repo root on sys.path)
from representations import ALL_VARIANTS, build_inputs

# from the PaPaGei code in the repo
from linearprobing.utils import load_model_without_module_prefix
from models.resnet import ResNet1D, ResNet1DMoE

# weight file for each model (downloaded from Zenodo into weights/)
MODEL_FILES = {
    "papagei_s": "papagei_s.pt",
    "papagei_s_svri": "papagei_s_svri.pt",
    "papagei_p": "papagei_p.pt",
}


def parse_embedding(name):
    """'papagei_s' -> ('papagei_s', 'zero_pad');
    'papagei_p@tile_cycles' -> ('papagei_p', 'tile_cycles')."""
    # split the name at "@" - no "@" means the default zero padding
    model_name, _, variant = name.partition("@")
    variant = variant or "zero_pad"
    # stop early if the name is wrong
    if model_name not in MODEL_FILES:
        raise SystemExit(f"Unknown model in --embeddings: {name}")
    if variant not in ALL_VARIANTS:
        raise SystemExit(f"Unknown input representation in --embeddings: "
                         f"{name} (choose from {ALL_VARIANTS})")
    return model_name, variant


def build_model(model_name, weights_dir="weights", device="cpu",
                load_weights=True):
    """Same configurations as the reproduction notebook (cells 29, 33, 36)."""
    # settings shared by all three models (output = 512 numbers)
    common = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
                  "groups": 1, "n_block": 18, "n_classes": 512}
    # PaPaGei-S uses a mixture of 3 experts, the others a plain ResNet
    if model_name == "papagei_s":
        model = ResNet1DMoE(n_experts=3, **common)
    elif model_name == "papagei_s_svri":
        model = ResNet1D(use_mt_regression=False, use_projection=False,
                         **common)
    elif model_name == "papagei_p":
        model = ResNet1D(**common)
    else:
        raise ValueError(f"Unknown model {model_name}")

    # load the pre-trained weights (skipped only for quick pipeline tests)
    if load_weights:
        path = Path(weights_dir) / MODEL_FILES[model_name]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing weights '{path}'. Download {MODEL_FILES[model_name]} "
                "from https://zenodo.org/records/13983110 into weights/.")
        model = load_model_without_module_prefix(model, str(path))
    model.to(device)
    model.eval() # evaluation mode: we only use the model, never train it
    return model


def embed_signals(model, signals, device="cpu", batch_size=256):
    """Embed a list of 1D signals, returns an (n_signals, 512) array.
    Signals of different lengths go in separate batches, which is fine
    because the encoder averages over time before its last layer. 
    Uses outputs[0] -> the same output the notebook uses.
    """
    embeddings = [None] * len(signals)
    # group the signals by length, since a batch needs equal lengths
    by_length = {}
    for i, s in enumerate(signals):
        by_length.setdefault(len(s), []).append(i)

    # no gradients needed, which makes it faster
    with torch.inference_mode():
        for idx in by_length.values():
            # go through each group in batches of batch_size
            for start in range(0, len(idx), batch_size):
                chunk = idx[start:start + batch_size]
                # shape (batch, 1 channel, length) as the model expects
                x = torch.tensor(np.stack([signals[i] for i in chunk]),
                                 dtype=torch.float32).unsqueeze(1).to(device)
                out = model(x)[0].cpu().numpy()
                # put each result back in its original position
                for j, i in enumerate(chunk):
                    embeddings[i] = out[j]
    return np.stack(embeddings)


def participant_embeddings(model, variant, recordings, subject_ids,
                           device="cpu"):
    """One 512-number row per participant, in the order of subject_ids:
    the mean of the embeddings of that participant's inputs."""
    # make the model inputs for this representation 
    # owners says which participant each input belongs to
    inputs, owners = build_inputs(variant, recordings, subject_ids)
    emb = embed_signals(model, inputs, device=device)
    owners = np.asarray(owners)
    # average each participant's embeddings into one row
    return np.stack([emb[owners == int(s)].mean(axis=0) for s in subject_ids])


def cached_embeddings(model_name, variant, ids_by_split, get_recordings,
                      cache_dir, weights_dir="weights", device="cpu",
                      load_weights=True, overwrite=False):
    """{split: (n, 512) array} for the participants of each split.
    Rows are reused from the cache file by participant ID, 
    only missing splits are embedded (calling get_recordings() to preprocess). 
    Random weights (load_weights=False, pipeline tests) use a separate cache file.
    """
    # one cache file per model, weights and representation
    tag = "pretrained" if load_weights else "random"
    cache = Path(cache_dir) / f"{model_name}_{tag}_{variant}.npz"
    stored = dict(np.load(cache)) if cache.exists() and not overwrite else {}
    out, model = {}, None
    for split, want in ids_by_split.items():
        # if every participant we want is already cached, just use that
        if f"X_{split}" in stored:
            lookup = dict(zip(stored[f"ids_{split}"].tolist(),
                              stored[f"X_{split}"]))
            if all(i in lookup for i in want):
                out[split] = np.stack([lookup[i] for i in want])
                continue
        # otherwise load the model and embed this split
        if model is None:
            model = build_model(model_name, weights_dir, device, load_weights)
        print(f"Embedding {model_name} / {variant} / {split} ...")
        X = participant_embeddings(model, variant, get_recordings(), want,
                                   device)
        # keep the participant IDs next to the embeddings, so rows can be matched by ID next time
        stored[f"X_{split}"], stored[f"ids_{split}"] = X, np.array(want)
        out[split] = X
    np.savez(cache, **stored)
    return out
