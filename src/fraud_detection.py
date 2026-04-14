import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Loading the dataset

df = pd.read_csv("creditcard.csv")
print(df.head())
print(df.describe())

# Analyzing Class Distribution

fraud_cases = df[df['Class'] == 1]
valid_cases = df[df['Class'] == 0]
outlier_fraction = len(fraud_cases) / float(len(valid_cases))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(df[df['Class'] == 1])))
print('Valid Transactions: {}'.format(len(df[df['Class'] == 0])))

# Exploring Transaction Amounts

print("Fraud transaction amount details")
fraud_cases.Amount.describe()
print("Valid transaction amount details")
valid_cases.Amount.describe()

# Plotting Correlation Matrix

correlation_matrix = df.corr()
figure = plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, vmax=.8, square=True)
plt.show()

# Preparing Data

features = df.drop(['Class'], axis=1)
target = df["Class"]
print(features.shape)
print(target.shape)
X_values = features.values
Y_values = target.values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_values, Y_values, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier

# Building and Training the Model

rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)
Y_pred = rf_model.predict(X_test)

# Evaluating the Model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
acc = accuracy_score(Y_test, Y_pred)
prec = precision_score(Y_test, Y_pred)
rec = recall_score(Y_test, Y_pred)
f1_val = f1_score(Y_test, Y_pred)
mcc_score = matthews_corrcoef(Y_test, Y_pred)
print("Model Evaluation Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1_val:.4f}")
print(f"Matthews Correlation Coefficient: {mcc_score:.4f}")
conf_mat = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
