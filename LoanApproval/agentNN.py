import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load the dataset
train_data = pd.read_csv("./Data/loan-train.csv")
test_data = pd.read_csv("./Data/loan-test.csv")

# Combine the datasets
data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

print("Data Normalization:")
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

# Encode categorical variables to one-hot encoded variables. 
data = pd.get_dummies(data, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

# Features and target variable
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)  # Encode target as 0 and 1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)


# Make predictions
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))
