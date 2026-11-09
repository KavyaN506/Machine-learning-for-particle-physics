# 🔬 Machine Learning for Particle Physics
## ⚛️ Higgs Signal vs Background Classification

### 🌟 Project Overview

This project applies classical machine learning algorithms to a particle physics classification problem: distinguishing simulated Higgs boson signal events from background events.

Using a 100,000-event subset of the UCI HIGGS dataset, I developed a machine-learning pipeline, trained multiple classifiers, evaluated their performance, and analysed the features that contributed most to classification.

🎯 **Objective:** Predict whether a particle collision event belongs to the simulated Higgs signal class or the background class.

### 📊 Dataset

- **Dataset:** UCI HIGGS Dataset
- **Source:** [UCI Machine Learning Repository – HIGGS](https://archive.ics.uci.edu/dataset/280/higgs)
- **Selected data:** First 100,000 rows
- **Input features:** 28
- **Target:** Binary label (`1` = signal, `0` = background)

The HIGGS dataset contains simulated Monte Carlo events. It is not a collection of raw experimental LHC collision data.

### 🛠️ Technologies Used

- 🐍 Python
- 🔢 NumPy
- 🐼 Pandas
- 📈 Matplotlib
- 🤖 Scikit-learn
- ☁️ Google Colab
- 🐙 Git and GitHub

### ⚙️ Methodology

1. **🔍 Data loading and exploration:** Loaded the first 100,000 rows and examined the dataset's structure, features and class distribution.
2. **🧹 Preprocessing:** Separated input features from the target label and used an 80:20 stratified train-test split.
3. **📏 Feature scaling:** Applied StandardScaler to the training and test features for Logistic Regression. Tree-based models used unscaled features.
4. **🧠 Model training:** Trained and compared three classification algorithms:
   - Logistic Regression
   - Decision Tree
   - Random Forest
5. **📊 Model evaluation:** Compared test accuracy and used classification metrics, confusion matrices and ROC analysis.
6. **🔬 Feature interpretation:** Analysed Random Forest feature importance, Logistic Regression coefficients and permutation importance.

### 🏆 Model Performance

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 64.51% |
| Decision Tree | 67.96% |
| Random Forest | **71.49%** |

🥇 Random Forest achieved the highest test accuracy among the three evaluated models.

*ROC-AUC values and additional evaluation results will be included after verification of the measured Day 8 outputs.*

### 🔎 Feature Importance Analysis

Three approaches were used to investigate influential input features:

- Random Forest impurity-based feature importance
- Logistic Regression coefficient analysis
- Permutation importance using ROC-AUC as the scoring metric

The most influential features were consistently among the reconstructed mass variables:

| Feature | Random Forest Importance | Permutation Importance |
|---|---:|---:|
| `m_bb` | 0.165954 | 0.099627 |
| `m_wwbb` | 0.086804 | 0.077549 |
| `m_wbb` | 0.085753 | 0.051912 |
| `m_jjj` | 0.066675 | 0.033984 |
| `jet_1_pt` | 0.059442 | 0.032638 |

💡 These results indicate that reconstructed mass variables contributed substantially to classification within the selected dataset. Feature importance describes predictive associations, not causal physical relationships.

### 📁 Repository Structure

```text
machine-learning-for-particle-physics/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 notebooks/
│   └── higgs_classification.ipynb
│
├── 📂 reports/
│   └── project_report.md
│
└── 📂 results/
    ├── random_forest_feature_importance.csv
    ├── logistic_regression_coefficients.csv
    ├── permutation_feature_importance.csv
    └── 📂 plots/
```

### 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/machine-learning-for-particle-physics.git
```

2. Navigate into the project:

```bash
cd machine-learning-for-particle-physics
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Open the notebook in Jupyter or Google Colab and execute the cells in order.

The notebook downloads the dataset from the UCI repository. An internet connection is required for the initial data download.

### ⚠️ Limitations

- The analysis uses only the first 100,000 rows of the approximately 11-million-event dataset.
- This is a sequential subset rather than a random sample of the full dataset.
- Only three classical machine learning algorithms were evaluated.
- Extensive hyperparameter optimisation was not performed.
- The analysis uses simulated Monte Carlo events, not raw experimental detector data.
- Generalisation to real experimental LHC data has not been established.
- Feature importance can be affected by correlations between input variables and should not be interpreted as causal evidence.

### 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Handling and exploring scientific datasets.
- Preparing data for machine learning.
- Training and comparing classification models.
- Evaluating model performance.
- Interpreting model predictions using feature importance techniques.
- Identifying methodological limitations in a scientific ML workflow.

### 🙏 Acknowledgement

The dataset used in this project is the HIGGS dataset provided through the UCI Machine Learning Repository.

This project is an educational exploration of machine learning techniques applied to a particle physics classification problem.
