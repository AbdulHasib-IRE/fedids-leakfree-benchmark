# Evaluation protocol matters more than aggregation rule in federated IoT intrusion detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23483716.svg)](https://doi.org/10.5281/zenodo.23483716)

Code, split definitions, per-run results and analysis scripts for the manuscript
"Evaluation protocol matters more than aggregation rule in federated IoT intrusion detection: a leak-free benchmark on CIC IoT-DIAD 2024"
(A. Hasib, S. C. Das, Md. Ashiqussalehin).

## What is here
| Folder | Contents |
|---|---|
| `paper_reproduction/` | Exported result tables (`data/`) and the scripts that compute every reported statistic and figure. `run_all.sh` regenerates all statistics (`outputs/analysis/`) and all 22 figures at 1000 dpi (`outputs/figures/`). |
| `results/` | Raw outputs of every run: 510 federated runs (round-by-round histories and round-100 test predictions), learning-rate control, five-local-epoch check, 50-client partial-participation runs, hyperparameter pilots, centralized references and baselines, windowing audit. |
| `splits/` | Row indices and labels of the duplicate-safe random split and the per-class time-block split (indices into the 282,617-flow sample), and the features kept by the train-only preprocessing. No features or IP addresses are included. |
| `results/v2/grid/random_iid_fednova_lr0.3_s42_check.json` | Not one of the 510 runs: the FedNova check with equal local steps, which reproduced FedAvg exactly (reported under verification in the paper). |
| `code/fl_core.py` | Federated simulation harness (partitions, local training, nine aggregation rules). |
| `notebook/` | The Colab notebook that ran every experiment, with outputs cleared. |

## Reproduce the statistics and figures (no GPU, about 1 min)
pip install -r requirements.txt
bash paper_reproduction/run_all.sh
Expected output includes `ALL PANELS PASS` (the figure-panel consistency check).
The scripts call `python3`; set `PYTHON=python` to use another interpreter.

## Re-run the experiments
Run the notebook cells in order on Google Colab (T4 GPU). The dataset is read from the public Kaggle mirror
`aymeneinformatique/cic-diad2024` with a Kaggle API token stored in Colab secrets.
A 100-round federated run takes about 4 min on a T4 (median 3.6 to 4.2 min, depending on the rule). Every run is seeded and deterministic GPU operations are enabled.


## Data
CIC IoT-DIAD 2024 is not redistributed. It is available from the Canadian Institute for Cybersecurity
(https://www.unb.ca/cic/datasets/iot-diad-2024.html).

## Environment
See `environment.json` (versions used for the runs) and `requirements.txt`.

## Citation
See `CITATION.cff`. Archived at Zenodo: https://doi.org/10.5281/zenodo.23483716 (all versions). License: MIT (code). Result files: CC BY 4.0.

## Contact
Saurav Chandra Das (corresponding author), saurav0001@uftb.ac.bd
