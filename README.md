# MFA vs Manual Phoneme Annotation: Impact on iEEG Decoding

**Paper:** *Comparing Automated (MFA) and Manual Phoneme Annotation for Intraoperative iEEG Speech Decoding*

## Overview

This repository contains analysis code and processed data for a study comparing **Manual (Kumar)** vs **Automated (MFA — Montreal Forced Aligner)** phoneme boundary annotations and their effect on intraoperative iEEG signals:

- **Timing accuracy** of phoneme boundaries (onset, offset, duration differences)
- **Neural representation** of phonemes in latent space (t-SNE + silhouette scores)
- **Phoneme decoding accuracy** (LDA classifier on high-gamma power)

**Participants:** 4 patients (S1–S4) undergoing intraoperative iEEG during a phoneme repetition task.
**Phonemes:** /a/, /ae/, /i/, /u/, /b/, /p/, /v/, /g/, /k/ at 3 word positions (P1, P2, P3).

---

## Repository Structure

```
├── src/
│   └── utils.py                         # Shared utilities: data loading, PCA, t-SNE, plotting
│
├── notebooks/
│   ├── 01_figure1_illustration.ipynb    # Spectrograms, waveforms, synthetic HG traces
│   ├── 02_figure2_timing_diffs.ipynb    # Manual vs Automated timing difference analysis
│   ├── 03_figure3_tsne.ipynb            # t-SNE + silhouette score analysis
│   ├── 04_figure3_hg_traces.ipynb       # High-gamma trace overlays per trial
│   └── 05_figure4_decoding.ipynb        # Phoneme decoding accuracy analysis
│
├── data/
│   ├── processed/                       # Preprocessed neural data (see note below)
│   └── results/                         # Decoding accuracy CSVs
│
└── figures/                             # Generated figures (figure1–figure4)
```

---

## Data

### Processed data (`data/processed/`)

| File | Description |
|------|-------------|
| `patient_data.pkl` | **Not tracked** (1.4 GB). Neural data for all patients × positions × methods: HG traces, maps, phoneme labels. Contact authors for access. |
| `mfa_kumar_diffs_meltedwithtrialnum.pkl` | Phoneme timing differences (onset, offset, duration) per trial, long format |
| `mfa_kumar_diffswithtrialnum.pkl` | Wide-format version of timing differences |
| `tsne_kumar.pkl` / `tsne_mfa.pkl` | Cached t-SNE embeddings |
| `phonemetimes_mfakumar.pkl` | Raw phoneme timing data for both aligners |

### Results (`data/results/`)

| File | Description |
|------|-------------|
| `Accuracy_df_Kumar_MFA.csv` | Phoneme decoding accuracy across patients, positions, and aligners |
| `Accuracy_DWs.csv` | Decoding accuracy across time windows (1s, 0.8s, ..., 0.1s) |
| `Accuracy_DWs2.csv` | Extended time window results (0.1s, 0.05s) |
| `Accuracy_DWs3.csv` | Fine-grained time window results (0.025s, 0.015s, 0.01s) |

---

## Setup

```bash
conda env create -f environment.yml
conda activate mfa-paper
```

Or with pip:

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn plotly \
            umap-learn librosa statsmodels tqdm joblib colorama \
            scikit-learn-intelex
```

---

## Running the Notebooks

Run notebooks in order from the `notebooks/` directory. Each loads data from `../data/` and saves figures to `../figures/`.

> **Note on raw data:** `patient_data.pkl` must be obtained separately (private intraoperative iEEG recordings). If the pickle is unavailable, update the path in `src/utils.py → fetch_patient_data()` to point to your local `.mat` files.

---

## Citation

*Coming soon*
