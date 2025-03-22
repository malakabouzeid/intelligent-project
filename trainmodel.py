from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import dump
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('heart.csv')
print("Dataset successfully loaded!")

# Step 1: Prepare Data (80/20 train-test split)
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data successfully split into training and testing sets.")

# Step 2: Train a Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Initial Decision Tree model trained.")

# Step 3: Hyperparameter Tuning
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
print("Performing hyperparameter tuning...")
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Use the best parameters found
best_dt = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Load validation data
validation_data = X_test.copy()
validation_labels = y_test.copy()

# Evaluate the trained model
y_pred = best_dt.predict(validation_data)
print("Validation Metrics:")
print(classification_report(validation_labels, y_pred))

# Save the trained model
dump(best_dt, 'optimized_decision_tree_model.joblib')
print("Optimized Decision Tree model saved as 'optimized_decision_tree_model.joblib'.")

