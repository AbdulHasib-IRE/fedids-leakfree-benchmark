# Evaluation protocol matters more than aggregation rule in federated IoT intrusion detection

Code, split definitions, per-run results and paper-reproduction scripts for the manuscript
"Evaluation protocol matters more than aggregation rule in federated IoT intrusion detection: a leak-free benchmark on CIC IoT-DIAD 2024"
(A. Hasib, S. C. Das, Md. Ashiqussalehin; submitted to *Computer Networks*).

## What is here
| Folder | Contents |
|---|---|
| `paper_reproduction/` | Exported result tables (`data/`), the scripts that compute every number, table row and figure, and the manuscript source (`paper/`). `run_all.sh` regenerates all 22 figures at 1000 dpi and checks that every number in `paper/main.tex` traces to a result file. |
| `results/` | Raw outputs of every run: 510 federated runs (round-by-round histories and round-100 test predictions), learning-rate control, five-local-epoch check, 50-client partial-participation runs, hyperparameter pilots, centralised references and baselines, windowing audit. |
| `splits/` | Row indices and labels of the duplicate-safe random split and the per-class time-block split (indices into the 282,617-flow sample), and the features kept by the train-only preprocessing. No features or IP addresses are included. |
| `results/v2/grid/random_iid_fednova_lr0.3_s42_check.json` | Not one of the 510 runs: the FedNova check with equal local steps, which reproduced FedAvg exactly (reported under verification in the paper). |
| `code/fl_core.py` | Federated simulation harness (partitions, local training, nine aggregation rules). |
| `notebook/` | The Colab notebook that ran every experiment, with outputs cleared. |

## Reproduce the paper's numbers and figures (no GPU, about 1 min)
pip install -r requirements.txt
bash paper_reproduction/run_all.sh
Expected last lines: `ALL PANELS PASS` and `691 numbers; 0 unmatched`.
The scripts call `python3`; set `PYTHON=python` to use another interpreter.

## Re-run the experiments
Run the notebook cells in order on Google Colab (T4 GPU). The dataset is read from the public Kaggle mirror
`aymeneinformatique/cic-diad2024` with a Kaggle API token stored in Colab secrets; never paste a token into a cell.
A 100-round federated run takes about 4 min on a T4 (median 3.6 to 4.2 min, depending on the rule). Every run is seeded and deterministic GPU operations are enabled.
The cells look for the project folder `FedAdapt_Thesis` in Google Drive (`/content/drive/MyDrive/FedAdapt_Thesis`); create it, or edit the path, before running.

## Data
CIC IoT-DIAD 2024 is not redistributed. It is available from the Canadian Institute for Cybersecurity
(https://www.unb.ca/cic/datasets/iot-diad-2024.html).

## Environment
See `environment.json` (versions used for the runs) and `requirements.txt`.

## Citation
See `CITATION.cff`. License: MIT (code). Result files: CC BY 4.0.

## Contact
Saurav Chandra Das (corresponding author), saurav0001@uftb.ac.bd
