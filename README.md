Exhaust Valve Condition Prediction using Machine Learning
**5(a). Problem Statement**
Condition monitoring of compressor equipment is important for
maintaining reliable and efficient operation of compressors.
An exhaust valve a critical part of compressor operations.
A dirty exhaust valve critically reduces Compressors efficiency.

In this project, machine learning classification models are used to
classify the condition of an exhaust valve as either Clean or
Dirty.
The objective is to develop and compare multiple machine learning
classification models using compressor operating and
condition-monitoring parameters.
The following classification models were implemented and evaluated:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Random Forest Classifier (Ensemble)
The models were evaluated using Accuracy, AUC, Precision, Recall, F1
Score and Matthews Correlation Coefficient (MCC).
**5(b). Dataset Description**
Original Dataset File
`data.csv`
Problem Type
Binary Classification
Target Variable
`exvalve`
The target variable represents the exhaust valve condition:
Clean
Dirty
Input Features
The following 20 compressor operating and condition-monitoring variables
are used as predictors:

rpm..
motor\_power..
torque..
outlet\_pressure\_bar..
air\_flow..
noise\_db..
outlet\_temp..
wpump\_outlet\_press..
water\_inlet\_temp..
water\_outlet\_temp..
wpump\_power..
water\_flow..
oilpump\_power..
oil\_tank\_temp..
gaccx..
gaccy..
gaccz..
haccx..
haccy..
haccz..

Dataset Split
Parameter        Value
---
Training data    80%
Test data        20%
Random State     42
Stratification   Yes
The resulting test set contains 200 observations.
Feature Scaling
Feature scaling was performed using `StandardScaler`. The scaler was
fitted on the training data and then applied to the training and test
data.
Test Data
`test\_data.csv` contains the 200 observations from the test set and is
used for model evaluation and the Streamlit application.
**5(c). GitHub Repository Link**
GitHub Repository:
https://github.com/SKSBit/Exhaust_Valve_Condition_Prediction_Compressor

**5(d). Models Used**
The following classification models were implemented and evaluated on
the same dataset:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Random Forest Classifier (Ensemble)





Comparison Table

| ML Model Name            | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| :----------------------- | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression      |   1.0000 | 1.0000 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree            |   0.9950 | 0.9969 |    0.9756 | 1.0000 | 0.9877 | 0.9846 |
| KNN                      |   0.9450 | 0.9645 |    1.0000 | 0.7250 | 0.8406 | 0.8236 |
| Gaussian Naive Bayes     |   0.8250 | 0.9223 |    0.5362 | 0.9250 | 0.6789 | 0.6101 |
| Random Forest (Ensemble) |   0.9950 | 1.0000 |    1.0000 | 0.9750 | 0.9873 | 0.9843 |

__Additional Analysis of Logistic Regression__

Because Logistic Regression achieved perfect performance, an additional
experiment was performed without the `air\_flow` feature.

Metric        Result

Accuracy      0.7900
AUC           0.8588
Precision     0.4615
Recall        0.3000
F1            0.3636
MCC           0.2527

The substantial reduction in performance indicates that `air\_flow` is a
highly influential predictor for distinguishing between Clean and Dirty
exhaust valve conditions.


**Observations on Model Performance**

| ML Model Name            | Observation about model performance                                                                                                                                       |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression      | Achieved perfect performance on the test dataset. The unusually high performance was investigated because `air_flow` is strongly associated with exhaust valve condition. |
| Decision Tree            | Achieved 99.5% Accuracy and 99.69% AUC. It correctly identified all 40 Dirty cases and 159 of the 160 Clean cases.                                                        |
| KNN                      | Achieved 94.5% Accuracy and 96.45% AUC. Precision was 1.0000, but Recall was 0.7250.                                                                                      |
| Gaussian Naive Bayes     | Produced the lowest overall performance. Recall was 0.9250, while Precision was 0.5362.                                                                                   |
| Random Forest (Ensemble) | Achieved 99.5% Accuracy and AUC of 1.0000. It correctly classified 39 of 40 Dirty cases and all 160 Clean cases.                                                          |
| **Overall Winner**       | **Logistic Regression** based on the six calculated evaluation metrics.                                                                                                   |


