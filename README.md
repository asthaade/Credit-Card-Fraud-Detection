# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Machine Learning](https://img.shields.io/badge/Focus-Anomaly%20Detection-green)

## 📌 Project Overview
This project aims to solve a critical problem for financial institutions: identifying fraudulent credit card transactions. Using a highly imbalanced dataset, we implement a **Random Forest Classifier** to distinguish between legitimate and fraudulent activities based on transaction patterns.

The goal is to minimize financial loss while ensuring that genuine customers are not inconvenienced by false alarms (False Positives).

## 🛠️ Technologies Used & Why?
* **Python:** The primary language for its extensive ecosystem of data science libraries.
* **Pandas & NumPy:** For efficient data manipulation and numerical computations.
* **Matplotlib & Seaborn:** Used for data visualization (Heatmaps, distribution plots) to understand feature correlations.
* **Scikit-Learn:** Provides the `RandomForestClassifier` and robust evaluation metrics.
* **Jupyter Notebook / VS Code:** Development environment.

## 🧠 Algorithm: Random Forest Classifier
The model uses an ensemble of Decision Trees. Each tree is trained on a random subset of data (bootstrapping) and a random subset of features.
- **Formula for Prediction:** The final classification is decided by a "majority vote" across all trees.
- **Handling Imbalance:** By using multiple trees, the model reduces variance and is less likely to be biased by the majority class (Valid transactions).

## 📊 Dataset Description
The project uses the **Credit Card Fraud Detection Dataset** (Kaggle).
- **Total Transactions:** 284,807
- **Fraudulent Cases:** 492 (Highly imbalanced: ~0.17%)
- **Features:** 31 (Time, Amount, Class, and V1-V28 which are PCA-transformed features for privacy).

## 🚀 Project Workflow

### 1. Exploratory Data Analysis (EDA)
We analyze the distribution of valid vs. fraudulent transactions. A **Correlation Heatmap** is generated to identify relationships between variables like `Amount`, `Time`, and the PCA features.


### 2. Data Preparation
- Separating features ($X$) and target ($Y$).
- Splitting data into **Training (80%)** and **Testing (20%)** sets.
- No scaling was required as Random Forest is robust to feature scaling.

### 3. Model Building
We utilized the **Random Forest Classifier**. This model was chosen because:
- It handles high-dimensional data well.
- It is resistant to overfitting.
- It provides high accuracy even in imbalanced datasets.

### 4. Evaluation Metrics
In fraud detection, accuracy is not enough. We focused on:
- **Precision:** How many predicted frauds were actually fraud?
- **Recall:** How many actual frauds did we catch?
- **F1-Score:** The harmonic mean of Precision and Recall.
- **Confusion Matrix:** To visualize True Positives vs. False Positives.


## 📈 Results
The model achieved:
- **Accuracy:** ~99.95%
- **Precision:** ~0.98
- **Recall:** ~0.79
- **MCC:** ~0.88

## 📂 Project Structure
```text
├── images/             # generated images (output)
├── src/                # Python scripts
├── README.md           # Project documentation
