# -EA-Classification-of-Human-Activities
Human Activity Recognition using wearable sensor data. Data analysis and machine learning pipeline covering data preprocessing, outlier detection, feature extraction, feature selection, dimensionality reduction and classification using Python, NumPy, SciPy and scikit-learn.

# Human Activity Recognition

This project was developed as part of **EA/ECAC 2025–2026** and focuses on **Human Activity Recognition (HAR)** using data collected from wearable sensors.

The project is based on the **FORTH-TRACE benchmark**, which contains sensor data collected from 15 participants performing different physical activities. Five wearable devices were used, positioned on different parts of the body:

| Device ID | Location        |
| --------- | --------------- |
| 1         | Left wrist      |
| 2         | Right wrist     |
| 3         | Chest           |
| 4         | Right upper leg |
| 5         | Left lower leg  |

Each sensor device provides measurements from an accelerometer, gyroscope and magnetometer, together with a timestamp and an activity label.

The original dataset contains **16 different activities**, including standing, sitting, walking, stair climbing, transitions between activities, and variations involving conversation.

---

# Module A — Data Preparation and Feature Engineering

The first part of the project focuses on the analysis and preparation of the sensor data, as well as the extraction and selection of informative features for Human Activity Recognition.

The work includes the implementation of Python, NumPy and SciPy functions for loading and processing the data.

### Outlier Analysis

Different methods are used to identify and analyse outliers in the sensor measurements.

The magnitude of the acceleration, gyroscope and magnetometer vectors is calculated and used for the analysis.

The following approaches are considered:

* Boxplot and IQR-based outlier detection
* Z-Score outlier detection using different thresholds
* K-Means clustering for multivariate outlier detection
* DBSCAN as an optional extension

The detected outliers are analysed across activities and sensor devices, and the results obtained with the different methods are compared.

### Feature Extraction

The project extracts temporal and spectral features from the sensor data using **5-second sliding windows with 50% overlap**.

Segments containing more than one activity are discarded.

The extracted features are based on the feature set described in:

> Zhang, M. & Sawchuk, A. A feature selection-based framework for human activity recognition using wearable multimodal sensors. BodyNets, 2011.

### Statistical Analysis

The extracted variables are statistically analysed to determine whether their mean values differ between the different activities.

Appropriate statistical tests are considered according to the characteristics of the data, including tests for assessing distribution normality.

### Dimensionality Reduction

**Principal Component Analysis (PCA)** is implemented and applied to the extracted feature set.

The features are standardized using Z-Score normalization, and the principal components are analysed according to the amount of variance they explain.

The objective is to determine the number of dimensions required to explain **75% of the variance** and to analyse the advantages and limitations of PCA for this problem.

### Feature Selection

Two feature selection methods are implemented:

* **Fisher Score**
* **ReliefF**

The **10 highest-ranked features** obtained with each method are identified and compared.

The advantages and limitations of each feature selection approach are also analysed.

---

# Module B — Machine Learning Model and Evaluation

The second part of the project focuses on applying and evaluating machine learning methods for Human Activity Recognition.

For this module, only the first **seven activities** from the original dataset are considered:

| Label | Activity                       |
| ----: | ------------------------------ |
|     1 | Stand                          |
|     2 | Sit                            |
|     3 | Sit and Talk                   |
|     4 | Walk                           |
|     5 | Walk and Talk                  |
|     6 | Climb Stair (up/down)          |
|     7 | Climb Stair (up/down) and Talk |

Activities with labels higher than 7 are discarded.

## Data Augmentation

The balance of the selected activities is analysed to determine whether the dataset is balanced.

The **SMOTE (Synthetic Minority Over-sampling Technique)** method is implemented to generate synthetic examples for a selected activity.

The method is specifically evaluated by generating three synthetic samples for **activity 4 (Walk) of participant 3**, using only data from that participant.

The original and synthetic samples are visualized using a 2D scatter plot based on the first two features.

## Embedding Features

In addition to the explicitly extracted features from Module A, the project explores feature representations learned by a pretrained deep learning model.

The provided `embeddings_extractor.py` code uses the **HARNET5** model from the [SSL-Wearables](https://github.com/OxWearables/ssl-wearables) project.

Only the X, Y and Z values of the accelerometer are considered. Each 5-second segment is resampled to **30 Hz**, after which an embedding vector is extracted from the pretrained model.

Two datasets are therefore considered:

* **FEATURES DATASET** — features extracted in Module A
* **EMBEDDINGS DATASET** — representations extracted from the pretrained model

## Data Splitting

Both datasets are evaluated using two different splitting strategies.

### Within-Subject Split

A **60/20/20% train-validation-test split** is performed within each participant, meaning that data from the same participant may occur in all three sets.

### Between-Subject Split

The split is performed at the participant level:

* 9 participants for training
* 3 participants for validation
* 3 participants for testing

The two strategies are compared, particularly in terms of their ability to estimate performance when classifying data from a new participant.

## Feature Representations

For both the FEATURES and EMBEDDINGS datasets, three representations are evaluated:

1. **All features/embeddings**
2. **PCA-reduced representation**, retaining the components that explain 90% of the variance
3. **ReliefF-selected representation**, retaining the 15 highest-ranked features

The PCA, ReliefF and normalization procedures must be determined using the training data only, without using information from the test set.

## Model Learning

A **k-Nearest Neighbors (kNN)** classifier is used to classify the activity associated with each segment.

A custom implementation of kNN is developed, with the scikit-learn implementation also available when required for improved performance.

Classification performance is evaluated using metrics including:

* Confusion matrix
* Accuracy
* F1-score
* Precision
* Recall

## Model Evaluation

The different combinations of datasets, feature representations and splitting strategies are evaluated and compared.

The value of **k** is selected using the training and validation data. The final model is then retrained using the training and validation sets and evaluated on the test set.

The results are analysed to determine:

* Which activities are more difficult to classify
* How the FEATURES and EMBEDDINGS datasets compare
* The effect of PCA on classification performance
* The effect of ReliefF feature selection
* The differences between within-subject and between-subject evaluation

Hypothesis testing is also performed to determine whether differences between models are statistically significant. The train-validation-test process is repeated several times to obtain performance distributions for the different models.

## Deployment

The selected model is used to create a function capable of receiving a NumPy array containing **256 samples and 9 sensor measurements**:

* Accelerometer X, Y, Z
* Gyroscope X, Y, Z
* Magnetometer X, Y, Z

The function processes the input and returns the predicted human activity.

## Further Work

The project also considers possible approaches for improving the classification system. An optional bonus consists of implementing and evaluating one or more of these improvements.
