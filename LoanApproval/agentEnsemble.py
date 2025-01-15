import pandas as pd

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

# LightGBM imports
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

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)


# Light GBM Model

lgb_model = LGBMClassifier(n_estimators=15, max_depth=3, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)

stacked_model = StackingClassifier(
    estimators=[('rf', rf_model), ('gb', lgb_model)],
    final_estimator=LogisticRegression()
)
stacked_model.fit(X_train, y_train)


# Make predictions
y_pred = stacked_model.predict(X_test)
y_pred_proba = stacked_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.2f}")

# ROC-AUC measures the model’s ability to distinguish between classes.
print(f"ROC-AUC Score: {roc_auc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))















