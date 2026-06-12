# Insider Threat Detection with Federated Learning — Preliminary Analysis

## Overview

This repository contains a **preliminary, exploratory analysis** carried out as part of the RECITALS project's privacy-preserving federated learning component. The goal of this stage is to validate a federated insider-threat-detection pipeline on a well-established public benchmark **before** moving to real healthcare log data. Currently, in the repository, we have focused on the **re-implementation and combination of existing, published approaches**, used purely for feasibility testing and as a baseline reference point for the project's healthcare-domain experiments.

The work is conceptually motivated by the personalized federated learning approach to insider threat detection described in [1].

## What this repository contains

This repository currently includes **four standalone training scripts**:

| Script | Setting | Model |
|---|---|---|
| `centralized_lstm.py` | Centralized (single dataset, no federation) | LSTM-based classifier |
| `centralized_transformer.py` | Centralized (single dataset, no federation) | Transformer-encoder-based classifier |
| `federated_lstm.py` | Federated (FedAvg, multiple simulated clients) | LSTM-based classifier |
| `federated_transformer.py` | Federated (FedAvg, multiple simulated clients) | Transformer-encoder-based classifier |

The centralized scripts establish a baseline performance level for binary insider-threat classification (malicious vs. non-malicious) on a single, pooled dataset. The federated scripts simulate a multi-client (multi-organization / multi-department) setting using **FedAvg**, under both IID and non-IID (Dirichlet-distributed) client data partitions, to assess how performance changes when the data is distributed and aggregated via federated averaging rather than pooled centrally.

This is intended as a **proof-of-concept / pipeline validation step**, establishing that the federated training loop, data partitioning, model architectures, and evaluation protocol work end-to-end, prior to porting the same pipeline to a real, privacy-sensitive healthcare dataset within RECITALS.

## Dataset

These scripts are designed to run on the **CERT Insider Threat Test Dataset (r4.2)**, a synthetic but widely used benchmark for insider threat research [3], originally generated following the methodology described in [4].

- Dataset source: [CERT Insider Threat Test Dataset (Carnegie Mellon University / Kilthub)](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247/1)

The dataset is **not included** in this repository due to its size and licensing. To reproduce the experiments, download it from the link above.

## Data preprocessing / feature extraction

Feature extraction from the raw CERT logs (logon, device, file, HTTP, email, LDAP, psychometric data) into the week-level tabular format expected by these scripts (`weekr4.2.csv`) follows the methodology and code from [2]:

- Paper: Le, D. C., Zincir-Heywood, N., & Heywood, M. I. (2020). Analyzing data granularity levels for insider threat detection using machine learning [2].
- Code: [github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets](https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets)

**To reproduce the preprocessing step:**

1. Download and extract the CERT r4.2 dataset from the Kilthub link above.
2. Clone the feature extraction repository linked above and follow its instructions to run `feature_extraction.py` on the extracted r4.2 data.
3. This produces week-level aggregated feature files (e.g. `weekr4.2.csv`), with one row per (user, week) and an `insider` label column (0 = benign, >0 = malicious scenario ID for that week).

No modifications were made to the original feature extraction logic for this preliminary stage; it is used as-is to produce the input CSV consumed by the scripts in this repository.

## Expected directory structure

The scripts expect the extracted CERT data at a relative path of the form:

```
../dataset/r4.2/ExtractedData/weekr4.2.csv        # for centralized_*.py
../../dataset/r4.2/ExtractedData/weekr4.2.csv     # for federated_*.py
```

Adjust the `file_path` variable at the top of each script to match your local directory layout if needed.

## Label definition

Following the convention of the CERT dataset and the feature extraction repository above, the original `insider` column is multi-class (0 = benign, 1–4 = different malicious scenario types, depending on the dataset version). For this preliminary binary classification setup:

```python
data['insider_binary'] = (data['insider'] != 0).astype(int)
```

i.e. any non-zero scenario label is collapsed into a single "malicious" class (1), and 0 remains "non-malicious" (0).

## Models

Both model families are simple, standard architectures adapted to operate on a single feature vector per (user, week) instance (i.e., a sequence length of 1):

- **LSTM classifier**: linear input projection → multi-layer LSTM → fully connected classification head → sigmoid output for binary classification.
- **Transformer classifier**: linear input projection → learned positional encoding → Transformer encoder layers (multi-head self-attention) → mean pooling → fully connected classification head → sigmoid output for binary classification.

These architectures are intentionally kept simple and are not tuned extensively — the focus of this stage is pipeline correctness and relative comparison across centralized vs. federated and IID vs. non-IID settings, not state-of-the-art performance.

## Federated learning setup

The federated scripts (`federated_lstm.py`, `federated_transformer.py`) implement standard **FedAvg**:

- Training data is split across `NUM_CLIENTS` simulated clients.
- Two partitioning modes are supported:
  - **Homogeneous (IID)**: training data shuffled and split into roughly equal, class-balanced chunks per client.
  - **Heterogeneous (non-IID)**: training data split per client using a **Dirichlet distribution** over class labels, with a configurable concentration parameter `alpha` (lower `alpha` → more skewed/heterogeneous client distributions).
- For each federated round, every client trains locally for `LOCAL_EPOCHS` epochs on its local partition, after which client model weights are aggregated via weighted FedAvg (weighted by each client's local dataset size) to update the global model.
- The global model is evaluated on a held-out, centrally pooled test set after each round.

Several heterogeneity settings (`alpha = 0.1, 0.5, 1.0`, plus a homogeneous/IID baseline) are run automatically within each federated script for comparison.

## Evaluation

All scripts evaluate the trained model(s) on a held-out test set (80/20 train/test split, stratified by label) using:

- Accuracy
- Precision, recall, F1-score (binary, macro, and weighted variants)
- ROC-AUC
- False positive rate (FPR) and true positive rate (TPR)

The centralized scripts additionally sweep over a range of **decision thresholds** (0.1–0.9) applied to the sigmoid output, to characterize the precision/recall trade-off across thresholds, and over a range of **training epoch counts** (10–100, in steps of 10), training a fresh model for each.

The federated scripts report per-round accuracy/loss curves, final metrics per heterogeneity setting, and visualize the resulting per-client class distributions for each partitioning mode.

## Outputs

Each script writes its results (summary CSVs and diagnostic plots — accuracy/loss curves, metric comparison bar charts, client distribution heatmaps) to a `results/` subdirectory, organized by model type, number of clients, number of rounds, local epoch count, and decision threshold.

## How to run

```bash
pip install pandas numpy torch scikit-learn matplotlib

python centralized_lstm.py
python centralized_transformer.py
python federated_lstm.py
python federated_transformer.py
```

Each script is self-contained; configuration parameters (number of clients, rounds, local epochs, learning rate, etc.) are set as constants near the top of the federated scripts and can be adjusted directly.

## Relationship to the RECITALS project

This is a **preliminary, literature-grounded feasibility study**. The CERT dataset serves as a stand-in benchmark to validate the federated training pipeline, model architectures, and evaluation methodology in a controlled setting where ground-truth insider threat labels are available. The next stage of this work will adapt this same pipeline to **real healthcare behavioral/log data** within the RECITALS project's privacy-preserving federated learning component, where a personalized federated learning approach (as motivated in the referenced literature) will be more directly relevant due to the heterogeneity across healthcare institutions.

## References

[1] Ye, X., Luo, F., Cui, H., Wang, J., Xiong, X., Zhang, W., ... & Zhao, W. (2025). Research on insider threat detection based on personalized federated learning and behavior log analysis. *Scientific Reports*, 15(1), 19214.

[2] Le, D. C., Zincir-Heywood, N., & Heywood, M. I. (2020). Analyzing data granularity levels for insider threat detection using machine learning. *IEEE Transactions on Network and Service Management*, 17(1), 30-44. Code: [github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets](https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets)

[3] Lindauer, Brian (2020): Insider Threat Test Dataset. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/12841247.v1

[4] Glasser, J., & Lindauer, B. (2013). Bridging the Gap: A Pragmatic Approach to Generating Insider Threat Data. *2013 IEEE Security and Privacy Workshops*, San Francisco, CA, pp. 98-104. doi: 10.1109/SPW.2013.37

---

All model implementations (LSTM/Transformer classifiers) and the FedAvg training loop in this repository are standard, well-established techniques implemented for this preliminary analysis; no novel algorithmic contribution is claimed at this stage.
