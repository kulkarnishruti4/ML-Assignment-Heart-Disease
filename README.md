
# Heart Disease Classification – Machine Learning Assignment 2

## a. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict the presence of heart disease based on clinical attributes.

Heart disease remains one of the leading causes of mortality worldwide. Early detection using predictive analytics can assist medical professionals in diagnosis and decision-making. This project compares six classification algorithms and evaluates their performance using multiple evaluation metrics.

---

## b. Dataset Description

The dataset used in this project is the UCI Heart Disease dataset, combining records from multiple hospitals (Cleveland, Hungary, Switzerland, and VA Long Beach).

### Dataset Characteristics

- Number of Instances: 920+
- Number of Features: 13
- Target Variable: Heart Disease Status  
  - 0 → No heart disease  
  - 1 → Presence of heart disease  

### Feature List

1. age  
2. sex  
3. cp (chest pain type)  
4. trestbps (resting blood pressure)  
5. chol (serum cholesterol)  
6. fbs (fasting blood sugar)  
7. restecg (resting ECG results)  
8. thalach (maximum heart rate achieved)  
9. exang (exercise induced angina)  
10. oldpeak (ST depression)  
11. slope (slope of ST segment)  
12. ca (number of major vessels)  
13. thal (thalassemia)

Missing values were handled using median imputation. The original multi-class target variable was converted into a binary classification problem.

---

## c. Models Used

The following six machine learning models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model Name         | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|-----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression   | 0.8261   | 0.8939  | 0.8431    | 0.8431  | 0.8431   | 0.6480  |
| Decision Tree         | 0.7717   | 0.7606  | 0.7586    | 0.8627  | 0.8073   | 0.5368  |
| KNN                   | 0.8424   | 0.8677  | 0.8411    | 0.8823  | 0.8612   | 0.6801  |
| Naive Bayes           | 0.8261   | 0.8875  | 0.8365    | 0.8529  | 0.8447   | 0.6473  |
| Random Forest         | 0.8370   | 0.9183  | 0.8396    | 0.8725  | 0.8558   | 0.6691  |
| XGBoost               | 0.8043   | 0.8786  | 0.8000    | 0.8627  | 0.8302   | 0.6026  |

---

## Observations on Model Performance

| ML Model Name       | Observation about Model Performance |
|---------------------|--------------------------------------|
| Logistic Regression | Demonstrates strong performance, indicating that the dataset has reasonable linear separability. Balanced precision and recall. |
| Decision Tree       | Slightly lower performance compared to other models. May be sensitive to data splits and prone to overfitting. |
| KNN                 | Achieved strong recall and balanced performance. Sensitive to feature scaling but performs well after normalization. |
| Naive Bayes         | Performs competitively despite independence assumptions. Indicates meaningful statistical structure in features. |
| Random Forest       | Achieved the highest AUC and strong MCC. Most robust and stable model among all implementations. |
| XGBoost             | Strong performance overall but slightly below Random Forest in this dataset. |
