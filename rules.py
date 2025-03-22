from experta import KnowledgeEngine, Rule, Fact, P
from rules import HeartDisease
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Define an Expert System Class
class CardioRiskExpert(KnowledgeEngine):
    @Rule(Fact(activity=P(lambda x: x == "regular")) & Fact(BMI=P(lambda x: x < 25)))
    def low_risk(self):
        print("Good job! Your regular exercise and healthy BMI contribute to a lower risk.")

    @Rule(Fact(bp=P(lambda x: x > 140)) & Fact(smoker=P(lambda x: x == "yes")))
    def high_risk_smoking(self):
        print("Caution: High blood pressure and smoking significantly raise your heart disease risk.")

    @Rule(Fact(cholesterol=P(lambda x: x > 240)) & Fact(age=P(lambda x: x > 50)))
    def high_cholesterol_risk(self):
        print("Attention: Elevated cholesterol levels and age put you at a higher risk of heart disease.")

    @Rule(Fact(bp=P(lambda x: x > 160)))
    def very_high_bp(self):
        print("Urgent: Extremely high blood pressure requires immediate medical intervention.")

    @Rule(Fact(BMI=P(lambda x: x > 30)) & Fact(activity=P(lambda x: x == "none")))
    def obesity_risk(self):
        print("Health Alert: Obesity and lack of exercise greatly increase your heart disease risk.")

    @Rule(Fact(sex="male") & Fact(age=P(lambda x: x > 45)))
    def male_risk(self):
        print("Be Aware: As a male over 45, your risk for heart disease is slightly elevated.")

    @Rule(Fact(sex="female") & Fact(age=P(lambda x: x > 55)))
    def female_risk(self):
        print("Take Care: Women over 55 have a higher chance of developing heart-related conditions.")

    @Rule(Fact(cholesterol=P(lambda x: x <= 200)) & Fact(BMI=P(lambda x: x < 25)))
    def healthy_profile(self):
        print("Great news! Your cholesterol and BMI are in a healthy range.")

    @Rule(Fact(family_history="yes") & Fact(smoker=P(lambda x: x == "yes")))
    def family_history_risk(self):
        print("Serious Concern: A family history of heart disease and smoking increase your risk significantly.")

    @Rule(Fact(diabetes="yes") & Fact(cholesterol=P(lambda x: x > 220)))
    def diabetes_cholesterol_risk(self):
        print("Warning: Diabetes combined with high cholesterol raises your risk of heart disease.")

    @Rule()
    def default(self):
        print("No major risk factors detected. Keep maintaining a healthy lifestyle!")

# Create an instance of the expert system
expert_engine = CardioRiskExpert()
expert_engine.reset()