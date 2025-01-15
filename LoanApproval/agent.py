import pandas as pd
import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from lightgbm import LGBMClassifier

# Load the dataset
train_data = pd.read_csv("./Data/loan-train.csv")
test_data = pd.read_csv("./Data/loan-test.csv")

# Combine the datasets
data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

print("Normalizing data.")

# Data Normalization
scaler = MinMaxScaler()
data['ApplicantIncome_Normalized'] = scaler.fit_transform(data[['ApplicantIncome']])
data['CoapplicantIncome_Normalized'] = scaler.fit_transform(data[['CoapplicantIncome']])

# Dropping the unnormalized data.
data = data.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

# Handle missing values
data.fillna({
    'Gender': data['Gender'].mode()[0],
    'Married': data['Married'].mode()[0],
    'Dependents': data['Dependents'].mode()[0],
    'Self_Employed': data['Self_Employed'].mode()[0],
    'LoanAmount': data['LoanAmount'].median(),
    'Loan_Amount_Term': data['Loan_Amount_Term'].median(),
    'Credit_History': data['Credit_History'].mode()[0]
}, inplace=True)

data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)


# Features and target variable
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)  # Encode target as 0 and 1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Performance for RandomForestClassifier model.")
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()


# Hyperparameter tuning
# Define parameter grid

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred_best)
roc_auc = roc_auc_score(y_test, y_pred_proba_best)

print("Performance of the hyper parameter tuned RandomForestClassifier model.");
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Light GBM Model

lgb_model = LGBMClassifier(n_estimators=15, max_depth=3, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)


y_pred_lgbm = lgb_model.predict(X_test)
y_pred_proba_lgbm = lgb_model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred_lgbm)
roc_auc = roc_auc_score(y_test, y_pred_proba_lgbm)

print(f"Performance for ligth gradient-boosting machine model.")
print("Prediction with LGBM model.");
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")









