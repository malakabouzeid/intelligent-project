from experta import KnowledgeEngine, Rule, Fact, P
from rules import HeartDisease
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def collect_user_data():
    age = int(input("Enter your age: "))
    bp = int(input("Enter your blood pressure (e.g., 120): "))
    cholesterol_level = int(input("Enter your cholesterol level (e.g., 200): "))
    bmi_value = float(input("Enter your BMI (e.g., 24.5): "))
    regular_exercise = input("Do you exercise regularly (yes/no): ").strip().lower()
    smoking_habit = input("Do you smoke (yes/no): ").strip().lower()
    family_heart_disease = input("Do you have a family history of heart disease (yes/no): ").strip().lower()
    diabetic = input("Do you have diabetes (yes/no): ").strip().lower()

    return {
        "age": age,
        "bloodPressure": bp,
        "cholesterol": cholesterol_level,
        "BMI": bmi_value,
        "exercise": "regular" if regular_exercise == "yes" else "none",
        "smoking": smoking_habit,
        "family_history": family_heart_disease,
        "diabetes": diabetic,
    }

def evaluate_risk(patient_info):
    print(type(patient_info), patient_info)
    if patient_info["cholesterol"] > 240 and patient_info["bloodPressure"] > 140:
        print("This patient has a HIGH risk of heart disease.")
    elif patient_info["BMI"] < 25 and patient_info["exercise"] == "regular":
        print("This patient has a LOW risk of heart disease.")
    else:
        print("This patient has a MODERATE risk of heart disease.")

# Gather user input
patient_info = collect_user_data()

# Predict risk based on user input
evaluate_risk(patient_info)

# Initialize expert system
expert_engine = HeartDisease()
expert_engine.reset()

# Declare facts for inference
for key, value in patient_info.items():
    expert_engine.declare(Fact(**{key: value}))

# Execute the expert system
expert_engine.run()

# Train Decision Tree Model
from trainmodel import validation_data, validation_labels, best_dt, X_test, X_train, y_test, y_train

dt_predictions = best_dt.predict(validation_data)

def expert_system_predict(row):
    if row['oldpeak'] > 2.0 and row['ca'] > 1:
        return 1  # High risk
    elif row['thalach'] > 150 and row['exang'] == 0:
        return 0  # Low risk
    else:
        return 0  # Moderate risk

expert_predictions = validation_data.apply(expert_system_predict, axis=1)

print("Decision Tree Model Performance:")
print(classification_report(validation_labels, dt_predictions))

print("Expert System Performance:")
print(classification_report(validation_labels, expert_predictions))

y_pred = best_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Report:")
print(classification_report(y_test, y_pred))

dump(best_dt, 'decision_tree_model.joblib')

plt.figure(figsize=(12, 8))
plot_tree(best_dt, feature_names=validation_data.columns, class_names=["No Risk", "High Risk"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

print("Explainability Notes:")
print("""
- Decision Tree:
    - Derived from training data, adapts to patterns in the dataset.
    - Can be visualized and analyzed to understand decision-making.
- Expert System:
    - Based on human-defined rules, limited by the scope of domain knowledge.
    - More interpretable but less flexible than data-driven models.
""")
