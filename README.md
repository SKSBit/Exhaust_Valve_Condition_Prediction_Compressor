# Exhaust Valve Condition Prediction using ML

## 1. Problem Statement

Condition monitoring of compressor equipment is important for maintaining reliable and efficient operation in industrial environments.

An exhaust valve in a compressor can operate under different conditions. In this project, machine learning classification models are used to classify the condition of an exhaust valve as either:

- Clean
- Dirty

The objective is to develop and compare multiple machine learning classification models using compressor operating and condition-monitoring parameters.

The following five classification algorithms are implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

---

## 2. Dataset Description
The dataset contains compressor operating and condition-monitoring parameters used to classify the condition of an exhaust valve.

### Original dataset file
 `data.csv`  

### Problem Type

Binary Classification

### Target Variable

`exvalve`

The target variable represents the exhaust valve condition:

- `Clean`
- `Dirty`

For machine learning classification, the target is encoded into two classes.

### Input Features

The following 20 compressor operating and condition-monitoring variables are used as predictors:

```text
rpm
motor_power
torque
outlet_pressure_bar
air_flow
noise_db
outlet_temp
wpump_outlet_press
water_inlet_temp
water_outlet_temp
wpump_power
water_flow
oilpump_power
oil_tank_temp
gaccx
gaccy
gaccz
haccx
haccy
haccz