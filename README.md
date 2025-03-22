# intelligent-project
Heart Disease Risk Assessment System
Project Overview

This project is a Heart Disease Risk Assessment System that utilizes machine learning and expert system rules to predict a patient's risk of heart disease. The system incorporates a Decision Tree Classifier trained on medical data and an Expert System built using predefined medical knowledge.
Features

 Predicts heart disease risk using Decision Tree Classification
 Provides expert-based risk assessment using rule-based logic
 Supports hyperparameter tuning to optimize model performance
 Generates classification reports for model evaluation
 Visualizes the Decision Tree structure for explainability
Technologies & Libraries Used

The project is implemented in Python using the following libraries:

    pandas → Data handling and preprocessing

    numpy → Numerical computations

    scikit-learn → Machine learning model (Decision Tree Classifier)

    experta → Rule-based expert system

    matplotlib → Visualization (Decision Tree plotting)

    joblib → Model saving and loading

How It Works

    Data Preparation: The dataset (heart.csv) is loaded and split into training and testing sets.

    Model Training: A Decision Tree Classifier is trained on the dataset.

    Hyperparameter Tuning: Grid Search is used to optimize model parameters.

    Expert System Rules: A rule-based system evaluates patient risk based on medical knowledge.

    Evaluation & Visualization: The model is tested, and its performance is analyzed with classification reports and decision tree diagrams.
