# Fraud Detection Kaggle


A comprehensive machine learning project for credit card fraud detection using Apache Spark and PySpark MLlib. This project implements multiple classification models to detect fraudulent transactions in a highly imbalanced dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Data](#data)

## 🎯 Overview

This project implements a fraud detection system that identifies fraudulent credit card transactions using machine learning. The solution handles class imbalance, implements time-based data splitting to prevent temporal leakage, and evaluates multiple models to find the best performer.

### Key Features

- **Time-based Data Splitting**: Prevents temporal leakage by splitting data chronologically
- **Multiple ML Models**: Logistic Regression, Gradient Boosting Trees, and Random Forest
- **Comprehensive Evaluation**: ROC-AUC, PR-AUC, Precision, Recall, F1-Score metrics
- **Feature Importance Analysis**: Model interpretation and feature contribution analysis
- **Distributed Processing**: Uses Apache Spark for handling large-scale data

## 📁 Project Structure

```
Fraud-Detection-Kaggle/
├── notebooks/
│   ├── 00_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 01_feature_engineering.ipynb    # Feature creation and engineering
│   ├── 02_train_model.ipynb            # Model training
│   ├── 03_model_evaluation.ipynb       # Model evaluation and metrics
│   └── 04_model_interpretation.ipynb   # Feature importance and model interpretation
├── data/
│   ├── raw/                            # Raw data files
│   ├── features/                       # Engineered features (parquet)
│   └── predictions/                    # Model predictions
├── models/                             # Trained models
│   ├── lr_baseline_model/
│   ├── gbt_baseline_model/
│   └── rf_baseline_model/
└── README.md
```

## 🛠 Technologies Used

- **Python 3.11**
- **Apache Spark / PySpark 4.1.0**: Distributed data processing and ML
- **PySpark MLlib**: Machine learning algorithms
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Additional metrics and utilities

## 📦 Installation

### Prerequisites

- Python 3.8+
- Java 8 or 11 (required for Spark)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Fraud-Detection-Kaggle
```

2. Install required packages:
```bash
pip install pyspark pandas numpy matplotlib seaborn scikit-learn scipy statsmodels gdown
```

Or install from a requirements file if available:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Notebooks

Execute the notebooks in the following order:

1. **00_EDA.ipynb** - Exploratory Data Analysis
   - Load and explore the dataset
   - Analyze fraud distribution
   - Identify data quality issues

2. **01_feature_engineering.ipynb** - Feature Engineering
   - Create temporal features (hour, day of week, night/weekend flags)
   - Generate logarithmic transformations
   - Save engineered features to parquet format

3. **02_train_model.ipynb** - Model Training
   - Load engineered features
   - Split data temporally (Train: Jan-May, Validation: June, Test: July)
   - Train three models:
     - Logistic Regression
     - Gradient Boosting Trees
     - Random Forest
   - Save trained models

4. **03_model_evaluation.ipynb** - Model Evaluation
   - Load trained models and validation data
   - Generate predictions
   - Evaluate metrics: ROC-AUC, PR-AUC, Precision, Recall, F1-Score
   - Create comprehensive visualizations

5. **04_model_interpretation.ipynb** - Model Interpretation
   - Extract feature importances
   - Analyze model coefficients
   - Compare feature importance across models
   - Generate insights and rankings

### Data Splitting Strategy

The dataset spans from **January 1, 2023 to July 2, 2023**:

- **Training Set**: January - May 2023 (~80% of data)
- **Validation Set**: June 2023 (~15% of data)
- **Test Set**: July 2023 (~5% of data)

This time-based split simulates real-world fraud detection scenarios where models predict future transactions.

## 📊 Notebooks

### 00_EDA.ipynb
- Data loading and schema exploration
- Fraud distribution analysis (13.45% fraud rate)
- Missing value checks
- Statistical summaries

### 01_feature_engineering.ipynb
- **Temporal Features**:
  - `TX_HOUR`: Hour of transaction
  - `IS_NIGHT`: Night transactions (0-6 AM)
  - `IS_WEEKEND`: Weekend transactions
- **Amount Features**:
  - `TX_AMOUNT`: Original transaction amount
  - `LOG_TX_AMOUNT`: Log-transformed amount
- **Time Features**:
  - `TX_TIME_SECONDS`: Seconds since dataset start
  - `TX_TIME_DAYS`: Days since dataset start

### 02_train_model.ipynb
- Feature vectorization and scaling
- Model training with PySpark ML pipelines
- Model persistence for reuse

### 03_model_evaluation.ipynb
- Comprehensive metric evaluation
- Visualizations:
  - Metrics comparison charts
  - ROC-AUC and PR-AUC comparisons
  - Precision/Recall/F1 analysis
  - Confusion matrices
  - Radar charts

### 04_model_interpretation.ipynb
- Feature importance extraction
- Model coefficient analysis
- Comparative feature rankings
- Cross-model agreement analysis

## 📈 Model Performance

All models are evaluated on the validation set using multiple metrics:

### Key Metrics

- **ROC-AUC**: Measures overall classification ability
- **PR-AUC**: More informative for imbalanced datasets
- **Precision**: Proportion of predicted frauds that are actual frauds
- **Recall**: Proportion of actual frauds correctly identified
- **F1-Score**: Harmonic mean of precision and recall

### Expected Performance

Models typically achieve:
- ROC-AUC: > 0.98
- PR-AUC: > 0.96
- Precision: > 0.99
- Recall: > 0.98
- F1-Score: > 0.99

*(Run the evaluation notebook to see actual results)*

## 🔍 Key Findings

### Data Characteristics

- **Class Imbalance**: Significant imbalance with ~86.55% non-fraud and ~13.45% fraud transactions
- **Temporal Patterns**: Fraud may exhibit time-based patterns (hour, day of week)
- **Amount Distribution**: Transaction amounts vary widely, requiring log transformation

### Feature Importance Insights

Based on model interpretation (run `04_model_interpretation.ipynb`):

- Top features typically include:
  - Transaction amount features (`TX_AMOUNT`, `LOG_TX_AMOUNT`)
  - Temporal features (`TX_TIME_SECONDS`, `TX_TIME_DAYS`)
  - Time-of-day features (`TX_HOUR`, `IS_NIGHT`)

### Model Comparison

- **Logistic Regression**: Fast, interpretable, good baseline
- **Gradient Boosting Trees**: Often best performance, captures non-linear patterns
- **Random Forest**: Robust, handles overfitting well

## 📂 Data

### Input Data

The project expects transaction data with the following schema:
- `TRANSACTION_ID`: Unique transaction identifier
- `TX_DATETIME`: Transaction timestamp
- `CUSTOMER_ID`: Customer identifier
- `TERMINAL_ID`: Terminal identifier
- `TX_AMOUNT`: Transaction amount
- `TX_FRAUD`: Fraud label (0 = non-fraud, 1 = fraud)

### Output Files

- **Engineered Features**: `data/features/fraud_features_v2.parquet`
- **Trained Models**: Saved in `models/` directory
- **Predictions**: Saved in `data/predictions/` directory

## 🎓 Best Practices Implemented

1. **Time-based Splitting**: Prevents data leakage by maintaining temporal order
2. **Feature Engineering**: Creates meaningful features from raw data
3. **Model Persistence**: Saves models for reuse and deployment
4. **Comprehensive Evaluation**: Uses multiple metrics suited for imbalanced data
5. **Model Interpretation**: Analyzes feature importance for insights
6. **Modular Notebooks**: Separates concerns (training vs. evaluation)

## 📝 Notes

- Ensure sufficient memory for Spark operations (especially for large datasets)
- Models are saved in Spark ML format and can be loaded for inference
- All paths are relative to the project root directory
- The project uses distributed computing capabilities of Spark for scalability

## 🔗 Resources

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Kaggle Fraud Detection Competitions](https://www.kaggle.com/search?q=fraud+detection)

## 📄 License

[Add your license here]

## 👤 Author

[Chau Thinh - chauthinh00710@gmail.com]

---

**Note**: This project was developed for educational purposes and demonstrates best practices in fraud detection using machine learning.

