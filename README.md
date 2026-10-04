# 🩷 PinkAI 2026 — Breast Mass Classification

Machine Learning project developed as part of the **DataCamp Hackathon – PinkAI 2026**.

The project focuses on classifying breast mass FNA records as **Benign (B)** or **Malignant (M)** using machine learning, with particular attention to data quality, patient-level validation, data leakage prevention, and threshold optimization.

> **Disclaimer:** This project was developed for educational and hackathon purposes. It is **not a clinically validated diagnostic system** and should not be used for medical diagnosis or clinical decision-making.

---

## 📌 Project Overview

Breast cancer classification is a binary classification problem where the model predicts whether a breast mass is:

* **B — Benign**
* **M — Malignant**

The project uses numerical measurements derived from fine needle aspiration (FNA) samples, along with selected categorical variables.

The main competition objective was to maximize:

> **Recall (Sensitivity) at Precision ≥ 90%**

Because the competition test labels are hidden, validation and out-of-fold predictions were used to evaluate the model and select the classification threshold.

---

## 🎯 Objectives

The main objectives of this project were to:

* Perform systematic data quality checks.
* Clean and standardize inconsistent data.
* Explore relationships between medical measurements and diagnosis.
* Compare multiple classification models.
* Prevent patient-level data leakage.
* Evaluate model stability using grouped cross-validation.
* Select a decision threshold based on the competition metric.
* Generate probability-based predictions for the test set.

---

## 📊 Dataset

The dataset contains FNA records collected across multiple clinical centers.

Each measurement family contains three representations:

* `_mean` — average measurement across nuclei.
* `_se` — standard error.
* `_worst` — average of the three largest measurements.

### Main Medical Features

The dataset includes measurements related to:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

Each measurement is available as:

```text
mean
se
worst
```

### Additional Variables

| Variable               | Description                  |
| ---------------------- | ---------------------------- |
| `patient_id`           | Patient registry identifier  |
| `exam_date`            | Date of examination          |
| `center`               | Clinical center              |
| `biopsy_followup_code` | Administrative registry code |
| `diagnosis`            | Target variable              |

Target encoding:

```text
0 = Benign
1 = Malignant
```

---

## 🧹 Data Cleaning

Several data-quality issues were identified and addressed before modeling.

### Cleaning steps

* Normalized `patient_id`.
* Converted numeric-looking object columns to numeric.
* Parsed `exam_date` as datetime.
* Standardized categorical values.
* Normalized inconsistent diagnosis labels.
* Identified malformed text encodings.
* Removed exact duplicate rows.
* Handled missing values through the preprocessing pipeline.

### Duplicate removal

After cleaning:

* **173 exact duplicate training rows** were removed.
* **58 exact duplicate test rows** were removed.

Importantly, repeated `patient_id` values were not automatically removed because a patient may have multiple records or examinations.

---

## 🔎 Exploratory Data Analysis

The EDA focused on:

* Diagnosis distribution.
* Feature distributions.
* Medical feature relationships with diagnosis.
* Correlation between numerical features.
* Potential outliers.
* Missing-value patterns.

Outliers were not automatically removed because extreme medical measurements may represent genuine observations rather than data errors.

---

## 🧩 Feature Decisions

The following decisions were made before modeling:

| Feature                | Decision | Reason                                  |
| ---------------------- | -------- | --------------------------------------- |
| `patient_id`           | Excluded | Identifier                              |
| `exam_date`            | Excluded | Raw date not used as predictive feature |
| `notes`                | Dropped  | Noisy free-text field                   |
| `sample_ref`           | Dropped  | Internal laboratory tracking identifier |
| `center`               | Used     | Categorical feature                     |
| `biopsy_followup_code` | Used     | Categorical administrative feature      |
| Medical measurements   | Used     | Main predictive information             |

No manual medical feature engineering was applied.

---

## ⚙️ Preprocessing Pipeline

Preprocessing was implemented using a scikit-learn `Pipeline` and `ColumnTransformer`.

### Numerical features

1. Median imputation
2. Standard scaling

### Categorical features

1. Most-frequent imputation
2. One-hot encoding

This ensured that preprocessing was fitted only on training data within each validation fold, reducing the risk of data leakage.

---

## 🤖 Models

Two baseline models were compared:

### Logistic Regression

Validation performance:

* ROC-AUC: **0.9795**
* Brier Score: **0.0299**

### Random Forest

Validation performance:

* ROC-AUC: **0.9891**
* Brier Score: **0.0240**

Random Forest performed better on both metrics and was selected as the final model.

---

## 🔐 Validation Strategy

Because multiple records may belong to the same patient, ordinary random splitting can introduce patient-level leakage.

We therefore used:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### Why?

**Group**

Ensures records belonging to the same patient are not split between training and validation.

**Stratified**

Maintains a similar Benign/Malignant class distribution across folds.

### Validation Check

The train/validation split was explicitly checked for shared patients:

```text
Shared patients = 0
```

---

## 📈 Cross-Validation Results

The Random Forest model was evaluated using 5-fold patient-level cross-validation.

| Metric      |       Mean |        Std |
| ----------- | ---------: | ---------: |
| ROC-AUC     | **0.9892** | **0.0023** |
| Brier Score | **0.0259** | **0.0019** |

The small standard deviations indicate relatively stable performance across folds.

---

## 🎚️ Threshold Optimization

The competition's primary metric is:

> **Recall at Precision ≥ 90%**

Therefore, using the default classification threshold of `0.50` was not necessarily optimal.

Instead, Out-of-Fold (OOF) predictions were generated from the 5-fold cross-validation process.

The OOF predictions were used to evaluate different thresholds without using the hidden test labels.

### Threshold Selection

The optimal OOF threshold was approximately:

```text
0.16
```

However, its precision was very close to the competition's minimum requirement.

To provide additional margin above the 90% precision floor, the final threshold was set to:

```text
0.20
```

### OOF Performance at Threshold = 0.20

| Metric    |     Result |
| --------- | ---------: |
| Precision | **0.9156** |
| Recall    | **0.9779** |

This threshold was selected as a more conservative choice rather than targeting the minimum precision boundary exactly.

---

## 🏆 Final Model

The final Random Forest model was trained on the complete cleaned training dataset.

```text
Training records: 8,827
Number of estimators: 300
Random state: 42
```

The final prediction pipeline is:

```text
Cleaned Data
     ↓
Preprocessing Pipeline
     ↓
Random Forest
     ↓
Probability of Malignancy
     ↓
Threshold = 0.20
     ↓
B / M Prediction
```

The test labels were not used during training or threshold selection.

---

## 📊 Final Validation Summary

| Metric                |              Result |
| --------------------- | ------------------: |
| Random Forest ROC-AUC |          **0.9891** |
| 5-Fold Mean ROC-AUC   | **0.9892 ± 0.0023** |
| Mean Brier Score      | **0.0259 ± 0.0019** |
| Final Threshold       |            **0.20** |
| OOF Precision         |          **0.9156** |
| OOF Recall            |          **0.9779** |

> These are **validation/OOF results**, not hidden test-set results.

---

## ⚠️ Limitations

The model may perform differently when exposed to data with:

* Different distributions from the training data.
* Different measurement procedures or units.
* New clinical-center characteristics.
* Different missing-value patterns.
* Changes in administrative variables.
* Patient populations not represented in the training data.

In addition, the `biopsy_followup_code` variable is an administrative feature, so its predictive usefulness may depend on when and how the code is recorded.

---

## 🩺 Medical Disclaimer

This project is intended **only for educational and hackathon purposes**.

The model has not undergone clinical validation, prospective testing, regulatory evaluation, or external clinical validation.

It should **not** be used to diagnose breast cancer or make clinical decisions.

For reliable medical information, consult qualified healthcare professionals and trusted medical organizations.

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Jupyter Notebook
* Random Forest
* Logistic Regression

---

## 🙌 Acknowledgment

We thank **DataCamp** for organizing the PinkAI 2026 Hackathon and providing an opportunity to apply machine learning skills to a healthcare-focused problem.

The project also highlights the importance of **breast cancer awareness and early detection**, while emphasizing the responsible use of AI in healthcare.

---

## 📚 Reference

For general breast cancer information and awareness:

**World Health Organization (WHO) — Breast Cancer**

https://www.who.int/news-room/fact-sheets/detail/breast-cancer

---

## 📜 License

This project was created for educational and hackathon purposes.

Please refer to the competition rules and dataset terms before redistributing any competition-provided data or materials.
