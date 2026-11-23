# Data sources, versions and licences
Record the access date and version for every external file you use (the notebooks store MD5 + size in `manifests/`).

| Resource | Used for | Where |
|---|---|---|
| Replogle et al., Cell 2022 (K562 essential, RPE1) | Main data / cross-cell-line | scPerturb Zenodo record 7041849 (file names in `configs/data_config.yaml`; verify), or pertpy loaders |
| Norman et al., Science 2019 (K562 CRISPRa, combos) | Scenario 3 | same Zenodo record; GEO GSE133344 |
| Adamson et al., Cell 2016 | Independent comparison | same Zenodo record |
| scPerturb (Peidli et al., Nat Methods 2024) | Harmonised h5ad collection | https://scperturb.org |
| Gene Ontology via Enrichr/gseapy | Pathway features, enrichment | downloaded at runtime |
| STRING (optional) | Gene graph | download links file, map to symbols, save `raw/string/string_edges_symbols.csv` (protein1, protein2, combined_score) |
| DepMap CRISPR gene effect (optional) | Essentiality evidence | https://depmap.org ; save `raw/depmap/CRISPRGeneEffect.csv` (+ `Model.csv`) |
| Cancer genes / druggability / disease association (optional) | Evidence | save as `raw/evidence/{cancer_genes,druggability,disease_assoc}.csv` with columns `gene,value` (e.g. COSMIC CGC, Open Targets, DGIdb: check each licence) |

Raw data are never committed to GitHub.
