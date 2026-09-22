# Human Activity Recognition (HAR) using Wearable Sensors

This project implements an end-to-end Machine Learning pipeline for **Human Activity Recognition (HAR)** using wearable inertial sensors.

The implementation is based on the **FORTH-TRACE** benchmark, which collects accelerometer, gyroscope, and magnetometer measurements across 5 body locations (wrists, chest, legs) from 15 participants.

---

## Project Overview & Pipeline

1. **Preprocessing & Outlier Analysis:** Computation of inertial vector magnitudes and outlier identification via univariate techniques (IQR, Z-Score across multiple $k$ thresholds) and multivariate clustering (K-Means and DBSCAN)
2. **Feature Engineering:** Extraction of temporal and spectral features using 5-second sliding windows with 50% overlap.
3. **Dimensionality Reduction & Feature Selection:** Z-score normalization, variance explanation using PCA, and feature ranking via Fisher Score and ReliefF.
4. **Data Augmentation:** Class balance assessment and synthetic sample generation using SMOTE[cite: 2, 4].
5. **Deep Embeddings:** Feature extraction at 30 Hz using the pretrained *HARNET5* model (*SSL-Wearables*) to compare against hand-crafted features.
6. **Classification & Validation:** Custom k-NN implementation evaluated under two data splitting strategies (*Within-Subject* and *Between-Subjects*), hyperparameter tuning, performance evaluation (F1-score, confusion matrices), and statistical hypothesis testing.
7. **Deployment:** Production-ready inference function that takes real-time streaming segments with shape `(256, 9)` and predicts the human activity.

---

## Repository Structure

```text
├── data/                      # Preprocessed feature tables and trained artifacts
│   └── best_model_artifacts.pkl
├── docs/                      # Course assignments and technical reports
│   ├── Relatório_Meta1_EA.pdf
│   ├── Relatório_Meta2_EA.pdf
│   ├── TP1.pdf
│   └── TP1_B_v2.pdf
├── src/                       # Reusable Python source modules
│   ├── clustering.py          # K-Means and DBSCAN algorithms
│   ├── data_splitting.py      # Within-subject and between-subject splits
│   ├── data_utils.py          # Data ingestion and vector magnitude computation
│   ├── embeddings_extractor.py# Self-supervised feature extraction (SSL-Wearables)
│   ├── feature_extraction.py  # Feature extraction, PCA, Fisher, ReliefF, and SMOTE
│   ├── outliers.py            # Univariate outlier detection (IQR, Z-Score)
│   ├── statistics_analysis.py # Statistical hypothesis testing
│   └── transformations.py     # Data transformation and pipeline utilities
├── .gitignore                 # Files ignored by Git
├── Main.ipynb                 # End-to-end Jupyter Notebook (Module A & B)
├── README.md                  # Project overview and instructions
├── requirements.txt           # Python environment dependencies
└── test_deployment.py         # Deployment validation script for (256, 9) segments
```

---

## Dataset Setup

This project uses the raw files from the **FORTH-TRACE benchmark dataset**.

> **Notice:** Due to GitHub's file size limits, the raw data folder `data/FORTH_TRACE_DATASET-master/` is excluded from this repository.

1. Download the raw dataset from the official repository:  
   [FORTH_TRACE_DATASET on GitHub]([https://github.com/FORTH-ICS-ISL/FORTH_TRACE_DATASET](https://github.com/spl-icsforth/FORTH_TRACE_DATASET.git)
2. Extract its contents into `data/FORTH_TRACE_DATASET-master/`.
3. Pre-extracted feature files (`features_dataset.csv`, `embeddings_dataset.csv`, etc.) exceed GitHub's 100 MB file limit and are generated upon executing the data extraction cells in `Main.ipynb`.
