# Improvements

Experiments that build on the reproduction in `01_ppgbp_reprod.ipynb`.
They reuse the same data, supplied 205-participant split, preprocessing,
model configurations and `evaluation_utils.evaluate_features()`, so the
numbers are directly comparable with the reproduction.

Results are appended to `results/improvement_results.csv`; per-participant
predictions and summary tables go to `results/improvements/`.

## Setup

1. Download the PPG-BP data from
   [figshare](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299)
   and unzip it so that the repo contains `data_PPG/Data File/PPG-BP dataset.xlsx`
   and `data_PPG/Data File/0_subject/`.
2. Download `papagei_s.pt`, `papagei_p.pt` (and optionally
   `papagei_s_svri.pt`) from [Zenodo](https://zenodo.org/records/13983110)
   into `weights/`.

`data_PPG/` and `weights/` are git-ignored.

## 1. Input representation for short recordings

PPG-BP recordings are 2.1 s long (~263 samples at 125 Hz). The paper and the
released notebook zero-pad them to the 10 s (1250 samples) used in
pre-training, so ~79% of each input is zeros. This experiment compares
zero padding with no padding, tiling, mirror padding, and joining the three
recordings before tiling.

```bash
# choose on validation
python improvements/01_input_representation.py --data-dir data_PPG --split val

# only after the choice is frozen
python improvements/01_input_representation.py --data-dir data_PPG --split test
```

Options: `--models papagei_s papagei_p papagei_s_svri`,
`--variants zero_pad native tile reflect concat3_tile`,
`--targets sysbp diasbp hr`. Embeddings are cached in
`data_PPG/features_improvements/`, so re-runs are fast.

`--no-weights` runs the pipeline with random weights; it is only a smoke test
and its rows are labelled `PIPELINE_TEST_...` in the CSV.
