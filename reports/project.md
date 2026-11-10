# 🔬 Machine Learning for Particle Physics
## ⚛️ Higgs Signal vs Background Classification


**An Exploratory Study of Machine Learning Applications in High-Energy Physics**


---

## 📌 1. Abstract

This project explores the application of classical machine learning algorithms to distinguish simulated Higgs boson signal events from background events.

Using a subset of **100,000 events** from the UCI HIGGS dataset, three classification algorithms were developed, trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

🏆 **Key Result:** Random Forest achieved the best performance, with **71.49% test accuracy** and a **ROC-AUC score of 0.793519**.

Feature importance analysis consistently identified reconstructed mass variables, particularly `m_bb`, `m_wwbb` and `m_wbb`, as influential predictors.

---

## 🌌 2. Introduction

Particle physics experiments generate large volumes of collision-event data. Identifying rare physical processes among numerous background events is an important challenge in high-energy physics.

Machine learning provides computational techniques for identifying patterns in complex datasets and distinguishing events belonging to different classes.

🎯 **Project Objective**

Develop and evaluate machine learning classifiers to distinguish simulated Higgs signal events from background events using the HIGGS dataset.

---

## 📊 3. Dataset Description

The dataset was obtained from the [UCI Machine Learning Repository – HIGGS](https://archive.ics.uci.edu/dataset/280/higgs).

| Dataset Property | Description |
|---|---|
| Dataset | HIGGS |
| Total dataset size | Approximately 11 million events |
| Selected subset | First 100,000 rows |
| Input features | 28 |
| Target variable | Binary classification |
| Signal class | `1` |
| Background class | `0` |

### 🔍 Feature Categories

The dataset contains two broad categories of input variables:

- **Low-level features:** Particle kinematic quantities such as transverse momentum, pseudorapidity, azimuthal angle and b-tagging information.
- **High-level features:** Reconstructed quantities such as invariant masses of particle combinations.

⚠️ **Scientific note:** The HIGGS dataset contains simulated Monte Carlo events, not raw experimental LHC detector data.

---

## ⚙️ 4. Methodology

The project followed a structured machine-learning workflow.

### 4.1 🔎 Data Exploration

- Loaded the selected dataset using Pandas.
- Examined dataset dimensions, feature names and data types.
- Investigated class distribution and missing values.
- Explored feature distributions using visualisations.

### 4.2 🧹 Data Preprocessing

- Separated input features (`X`) and target labels (`y`).
- Applied an 80:20 stratified train-test split.
- Used `random_state=42` for reproducibility.
- Applied `StandardScaler` to features used by Logistic Regression.
- Used unscaled features for Decision Tree and Random Forest.

### 4.3 🤖 Model Development

Three classification algorithms were implemented:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Baseline linear classification |
| Decision Tree | Learning nonlinear decision rules |
| Random Forest | Ensemble-based classification |

### 4.4 📈 Model Evaluation

Model performance was assessed using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Receiver Operating Characteristic (ROC) curves
- Area Under the ROC Curve (ROC-AUC)

### 4.5 🔬 Feature Interpretation

Three complementary methods were used:

1. Random Forest impurity-based feature importance.
2. Logistic Regression coefficient analysis.
3. Permutation importance using ROC-AUC as the scoring metric.

---

## 🏆 5. Experimental Results

The following results were obtained using the held-out test dataset.

### 5.1 Model Performance Comparison

| Model | Test Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 64.51% | 0.687549 |
| Decision Tree | 67.96% | 0.741849 |
| **Random Forest** | **71.49%** | **0.793519** |

📌 Interpretation of evaluation metrics

- **Accuracy:** Proportion of correctly classified test events at the selected classification threshold.
- **Precision:** Proportion of predicted positive events that are actually positive.
- **Recall:** Proportion of actual positive events correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to rank signal events above background events across classification thresholds.


🥇 **Best-performing model: Random Forest**

Random Forest achieved the highest accuracy and ROC-AUC among the three evaluated algorithms.

### 5.2 📊 Feature Importance Analysis

The three interpretation methods consistently identified reconstructed mass variables among the most influential features.

| Feature | Random Forest Importance | Permutation Importance |
|---|---:|---:|
| `m_bb` | 0.165954 | 0.099627 |
| `m_wwbb` | 0.086804 | 0.077549 |
| `m_wbb` | 0.085753 | 0.051912 |
| `m_jjj` | 0.066675 | 0.033984 |
| `jet_1_pt` | 0.059442 | 0.032638 |

**Key observation:** The `m_bb` feature achieved the highest Random Forest importance score of **0.165954** and the highest permutation importance score of **0.099627**.

These results suggest that reconstructed mass variables contain useful predictive information for distinguishing simulated signal and background events within the selected sample.

Feature importance represents predictive associations rather than causal physical relationships.

---

## 💡 6. Discussion

The experimental results demonstrate that Random Forest outperformed Logistic Regression and Decision Tree on the selected dataset.

This suggests that an ensemble of decision trees can capture useful nonlinear relationships within the input features.

The consistency between the three feature interpretation methods provides additional insight into the variables used by the classifiers.

However, correlated input features can influence feature importance estimates and make individual feature contributions difficult to interpret independently.

---

## ⚠️ 7. Limitations

The following limitations should be considered when interpreting the results:

1. **Limited dataset:** Only the first 100,000 rows of the approximately 11-million-event dataset were analysed.
2. **Sequential sampling:** The selected subset was not randomly sampled from the full dataset.
3. **Limited model comparison:** Only three classical machine learning algorithms were evaluated.
4. **Hyperparameter optimisation:** Extensive hyperparameter tuning was not performed.
5. **Simulated data:** The analysis uses Monte Carlo simulations rather than raw experimental detector data.
6. **Generalisation:** Performance on real experimental LHC data has not been established.
7. **Feature correlations:** Correlations between variables can influence feature importance and coefficient interpretation.

---

## 🚀 8. Future Work

Potential extensions of this project include:

- Analysing a larger and randomly sampled subset of the HIGGS dataset.
- Performing systematic hyperparameter optimisation.
- Exploring Gradient Boosting and XGBoost classifiers.
- Investigating neural networks and deep learning approaches.
- Applying cross-validation for more robust performance estimation.
- Studying feature correlations and model interpretability in greater depth.
- Investigating how simulation-based models could be evaluated against experimental data, subject to appropriate validation.

---

## 🎓 9. Conclusion

This project implemented an introductory machine-learning workflow for simulated Higgs signal-background classification.

Among the three evaluated algorithms, Random Forest achieved the best performance, with **71.49% test accuracy** and a **ROC-AUC score of 0.793519**.

Feature importance analysis highlighted reconstructed mass variables, particularly `m_bb`, `m_wwbb` and `m_wbb`, as influential predictors.

The project provided practical experience in scientific data exploration, preprocessing, classification, model evaluation and feature interpretation.

It also highlighted the importance of documenting dataset limitations and interpreting machine learning results carefully in scientific applications.

---

## 📚 10. References

1. Baldi, P., Sadowski, P. and Whiteson, D. (2014). *Searching for Exotic Particles in High-Energy Physics with Deep Learning*. Nature Communications, 5, 4308.  
   https://doi.org/10.1038/ncomms5308

2. UCI Machine Learning Repository. *HIGGS Dataset*.  
   https://archive.ics.uci.edu/dataset/280/higgs

3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

---

**🔬 Exploring the intersection of Machine Learning and Particle Physics ⚛️**
