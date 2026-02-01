# Predicting Patient Discomfort in Hospital Infusion Rooms from Environmental Sensors

This MATLAB project builds and evaluates models to predict excessive patient discomfort in hospital infusion rooms using only environmental sensor data.  

Two models are implemented and compared:
- Logistic Regression (LR) – linear, interpretable baseline  
- Random Forest (RF) – non‑linear ensemble model  

The code performs exploratory data analysis (EDA), preprocessing, 10‑fold cross‑validation, test‑set evaluation, and ROC comparison.

---

1. Files and Their Roles

All scripts assume the Kaggle dataset (e.g. medical_environment_comfort.csv) is in the current folder or a data/ subfolder (adjust paths inside scripts if needed).

a) eda_preprocessing.m

-Purpose:

Performs exploratory data analysis and preprocessing.

-Main steps:

   - Perform Exploratory Data Analysis (EDA)
   - Check class imbalance
   - Visualize feature distributions
   - Compute summary statistics
   - Produce correlation heatmap
   - Perform Z-score normalization
   - Save normalized data for modelling stage

-Produces:
   - normalized_features.mat = Standardized feature matrix (X_norm) and target labels (Y).

%  NOTE:
%  This file contains NO TRAINING and NO TESTING. 

> Run this script first to prepare the data that all later scripts use.

---

b) train_models_with_CV.m

-Purpose:

Trains both Logistic Regression and Random Forest models with 10‑fold cross‑validation and class weighting.

-Main steps:

- Loads the preprocessed training data produced by 'eda_preprocessing.m'.
- Trains:
  - Logistic Regression (LR) with:
    - Standardized features
    - Class weights to counter the 90/10 class imbalance
  - Random Forest (RF) with:
    - A specified number of trees (bagged ensemble)
    - The same class weighting scheme
- Performs **10‑fold cross‑validation** for each model.
- Computes and displays cross‑validated metrics:
  - Precision, Recall, F1‑score
  - Average training AUC and error
  - Training time and (optionally) prediction time
- Optionally generates “overfitting check” bar plots (10‑CV vs. train metrics).
- Saves the trained LR and RF models to the workspace (and/or .mat files) to be used by the testing scripts.

- Produces:
   - LR_best.mat = Stores the final, optimized Logistic Regression model object
   - RF_best.mat = Stores the final Random Forest model object
   - train_data.mat = Contains the training features (Xtrain) and labels (Ytrain).
   - test_data.mat  = Contains the unseen test features (Xtest) and labels (Ytest).
   - CV_metrics.mat = Stores the average 10-fold cross-validation scores (Precision, Recall, and F1) for both models.
   - test_set.mat = Contains unseen test set used for final evaluation (Xtest, Ytest).

> Run this script second to fit both models and obtain the cross‑validation results reported in the poster.

---

c) test_LR_model.m

-Purpose:

Evaluates the Logistic Regression model on the held‑out test set.

-Main steps:

- Load trained Logistic Regression model
- Evaluate on unseen test data only
- Compute test metrics
- Plot confusion matrix and ROC curve
- Prints a concise performance summary in the MATLAB command window.

-Produces:
  - LR_test_metrics.mat = Stores the final test-set metrics (Precision, Recall, and F1-score) for the Logistic Regression model.

> Run this script third to generate the LR confusion matrix and ROC curve used in the results.

---

d) test_RF_model.m

-Purpose:

Evaluates the Random Forest model on the held‑out test set.

-Main steps:

- Load trained Random Forest model
 - Evaluate on unseen test data only
- Compute test metrics
- Plot confusion matrix and ROC curve
- Prints a performance summary showing, for example, the high recall of RF.

-Produces:
  - RF_test_metrics.mat = Stores the final test-set metrics (Precision, Recall, and F1-score) for the Random Forest model. 

> Run this script fourth to generate the RF confusion matrix and ROC curve used in the results.

---

e) combined_ROC.m

-Purpose:

Produces a direct ROC comparison of Logistic Regression vs. Random Forest on the same test set.

-Main steps:

- Load trained LR and RF models
- Load unseen test data
 - Plot combined ROC curves
- Compare AUC values
 - Generate Overfitting Check bar charts (comparing 10-CV vs. Test metrics).
- Verify model stability and generalization across datasets.


> Run this script last to obtain the combined ROC comparison figure.

---

2. Data Overview

- Source: Kaggle – Medical Environment Comfort Prediction dataset  
- Samples: 1,000 infusion‑room sessions  
- Features (11 numeric sensors):
  - Temperature  
  - Relative humidity  
  - Noise level  
  - Air Quality Index (AQI)  
  - CO₂ concentration  
  - O₂ concentration  
  - Air pressure  
  - Lighting intensity  
  - Wind speed  
  - Air flow speed  
  - Particulate pollutant concentration  
- Target: Discomfort (binary)
  - 1 = excessive discomfort  
  - 0 = no excessive discomfort  
- No missing values; all predictors are numeric. The target is highly imbalanced (~90% class 1, 10% class 0); class weights are used during training.

---

3. How to Reproduce the Results

To generate the same type of results as in the report/poster (EDA plots, cross‑validation metrics, test‑set confusion matrices, ROC curves, combined ROC comparison), run the scripts in this exact order:

1) 'eda_preprocessing.m' – loads the raw CSV, performs EDA and preprocessing, and creates the train/test split.  
2) 'train_models_with_CV.m' – trains Logistic Regression and Random Forest with 10‑fold cross‑validation and saves the trained models.  
3) 'test_LR_model.m' – uses the trained LR model to produce LR test‑set metrics, confusion matrix, and ROC curve.
4) 'test_RF_model.m' – uses the trained RF model to produce RF test‑set metrics, confusion matrix, and ROC curve.
5) 'combined_ROC.m' – loads both models’ test predictions and plots the combined ROC comparison (LR vs. RF).

---

4. Requirements

- MATLAB R2025b or later  
- Statistics and Machine Learning Toolbox  
- Bioinformatics Toolbox

No additional external toolboxes are required.
```
