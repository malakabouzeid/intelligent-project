from experta import KnowledgeEngine, Rule, Fact, P
from rules import HeartDisease

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
