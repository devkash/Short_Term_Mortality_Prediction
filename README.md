# Short Term Mortality Prediction

## Goal

The goal of this project is to improve the standard severity illness scoring systems by using the MIMIC-III dataset, of roughly 40,000 critical care patients, to develop a machine learning model that can accurately short term predict mortality in a set of ICU patients spanning a wide range of conditions and backgrounds.

In attempting to maximize the performance of the model, different combinations of input features drawing from vitals and demographic information, ICD-9 diagnosis codes, and disease severity scores such as the Apache II, SAPS II, and SOFA scores are explored. Additionally, the “EXPIRE_FLAG” provided in the MIMIC-III database is used to extract mortality labels for each patient. Using the available input features and these target mortality labels, a model that makes a binary prediction on the mortality status of a given patient is developed. Several tables from the MIMIC dataset will be used to gather these features; particularly, data from the “CHARTEVENTS”, “LABEVENTS”, “DIAGNOSES_ICD”, “D_ITEMS”, and “PRESCRIPTIONS” tables will be aggregated together to link each patient to their lab tests, what they have been diagnosed with, and also the current medications they are on. Both filtering and heuristic feature selection processes will then be used to reduce the dimensionality of the input feature map for a more efficient and precise model.

Three sets of machine learning models, Logistic Regression, Multilayer Perceptron, Random Forest, and Support Vector Machine, were developed to predict patient mortality and each set included a different set of features.

Feature Sets
1. Scores from severity illness scores SAPSII, OSASIS, and APSIII
2. Sub-scores from SAPSII, OSASIS, and APSIII
3. Raw features


## Performance of Machine Learning models

### First Feature Set: Severity Illness Scores from SAPSII, OSASIS, and APSIII
![](performance/Illness_scores/logistic_regression.png)
![](performance/Illness_scores/mlp.png)
![](performance/Illness_scores/random_forest.png)
![](performance/Illness_scores/svm.png)
![](performance/Illness_scores/confusion_matrix.png)

### Second Feature Set: Sub_scores from Severity Illness Scores
![](performance/Subscores/logistic_regression.png)
![](performance/Subscores/mlp.png)
![](performance/Subscores/random_forest.png)
![](performance/Subscores/svm.png)
![](performance/Subscores/confusion_matrix.png)

### Third Feature Set: Raw Measurements Recorded for patients
![](performance/Raw_features/all_models.png)
