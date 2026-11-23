# Reproducing PerturbAI

## A. Google Colab + Google Drive (recommended)
1. Fork/clone the repo on GitHub. Edit `REPO_URL` once in the first cell of each notebook.
2. Open `notebooks/01_data_acquisition.ipynb` in Colab (File → Open → GitHub).
3. Run all cells; allow Drive access. Data land in `MyDrive/PerturbAI_data/` (change with env var `PERTURBAI_DRIVE_FOLDER`).
4. Continue in order 02 → 13. Every notebook reads/writes only through `perturbai.paths`.
5. Colab free tier: keep the default subset in `configs/data_config.yaml`; scale up gradually.

## B. Local machine
```bash
git clone https://github.com/<you>/PerturbAI.git && cd PerturbAI
conda env create -f environment.yml && conda activate perturbai   # or: python -m venv .venv && pip install -r requirements.txt && pip install -e .
export PERTURBAI_DATA=$HOME/PerturbAI_data      # any folder with enough space
jupyter lab notebooks/
```

## C. Smoke test (1 minute, no download)
Set `SYNTHETIC = True` in the header cell of notebooks 03 → 13, or run `pytest`.

## D. Pushing results back to GitHub from Colab
Use a fine-grained personal access token stored in Colab **Secrets** (key icon). Never paste tokens in notebooks. Commit only code, configs, small figures. Notebook outputs are large: use `nbstripout`.
