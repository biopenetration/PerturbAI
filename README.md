# PerturbAI

**Can a model trained on observed CRISPR perturbation responses predict the transcriptional response to genes it has never seen perturbed, and do such predictions help prioritise candidate targets?**

Fair benchmark first (control / mean / Ridge / kNN / RF / MLP / graph-smoothed / optional GEARS), leakage-free perturbation-level splits, multi-seed uncertainty, a noise ceiling, and a translational layer that outputs *hypotheses*, not validated drug targets.

## Quick start
| Where | Steps |
|---|---|
| Colab + Drive | Open `notebooks/01_data_acquisition.ipynb` in Colab, edit `REPO_URL` in cell 2, run. Data persist in `MyDrive/PerturbAI_data`. |
| Local | `pip install -e .` (see `docs/REPRODUCE.md`), `export PERTURBAI_DATA=/path/to/data`, open notebooks. |
| Smoke test | `pip install -e .[dev] && pytest` (no downloads) or set `SYNTHETIC = True` in notebooks. |

## Storage design
GitHub: code, configs, notebooks, tests. Google Drive / local data root: raw data, processed h5ad, models, results (`perturbai.paths.get_paths()` resolves it). Colab scratch disk for heavy I/O.

## Notebooks
| # | Notebook | Output |
|---|---|---|
| 01 | Data acquisition & audit | dataset audit JSON, raw file on Drive |
| 02 | Experimental design | design checks |
| 03 | QC & knockdown filtering | QC'd subset h5ad |
| 04 | Exploratory analysis | UMAP/PCA/clusters |
| 05 | Effect estimation | delta matrix, noise ceiling |
| 06 | Differential response | pseudobulk DE vs naive test |
| 07 | Pathways & perturbation network | enrichment, GO-sharing validation |
| 08 | Feature engineering | gene feature table |
| 09 | Baseline models | benchmark (5 seeds) |
| 10 | PyTorch MLP | benchmark |
| 11 | Graph learning (+ optional GEARS) | benchmark |
| 12 | Final benchmark & error analysis | figures, best model |
| 13 | Target prioritisation | ranked candidate list |

App: `streamlit run app/streamlit_app.py`; API: `uvicorn app.api:app`; Docker: see `Dockerfile`.

## Data and citations
See `docs/DATA_SOURCES.md`. Primary: Replogle et al., *Cell* 2022 (doi:10.1016/j.cell.2022.05.013); harmonised via scPerturb (Peidli et al., *Nat Methods* 2024).

## Honest limitations
Predictions of CRISPRi knockdown in one cell line; no claim of efficacy or safety; essentiality ≠ therapeutic window; many perturbations have weak, noisy effects (see noise ceiling); simple baselines can match complex models.

License: MIT.
