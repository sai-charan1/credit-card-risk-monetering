# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:15:51 2025
@author: saich
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency, f_oneway
import xgboost as xgb
import warnings
import os



# Suppress warnings
warnings.filterwarnings("ignore")



# Load the dataset
a1 = pd.read_excel("D:/Projects/creditcard_risk_management/case_study1.xlsx")
a2 = pd.read_excel("D:/Projects/creditcard_risk_management/case_study2.xlsx")

df1 = a1.copy()
df2 = a2.copy()



# Remove nulls and unwanted columns
df1 = df1[df1['Age_Oldest_TL'] != -99999]
columns_to_be_removed = [
    col for col in df2.columns if df2[df2[col] == -99999].shape[0] > 10000
]
df2 = df2.drop(columns_to_be_removed, axis=1)
for col in df2.columns:
    df2 = df2[df2[col] != -99999]


# Merge the two dataframes
df = pd.merge(df1, df2, how='inner', on='PROSPECTID')



# Identify categorical features
categorical_columns = [
    col for col in df.columns if df[col].dtype == 'object'
]



# Chi-square test for categorical columns
significant_categorical_columns = []
for col in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[col], df['Approved_Flag']))
    if pval <= 0.05:
        significant_categorical_columns.append(col)





# VIF for numerical columns
numeric_columns = [
    col for col in df.columns if df[col].dtype != 'object' and col not in ['PROSPECTID', 'Approved_Flag']
]
vif_data = df[numeric_columns]
columns_to_be_kept = []
for col in numeric_columns:
    vif_value = variance_inflation_factor(vif_data.values, vif_data.columns.get_loc(col))
    if vif_value <= 6:
        columns_to_be_kept.append(col)




# ANOVA for numerical columns
columns_to_be_kept_numerical = []
for col in columns_to_be_kept:
    group_values = [df[col][df['Approved_Flag'] == flag] for flag in df['Approved_Flag'].unique()]
    f_statistic, p_value = f_oneway(*group_values)
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(col)




# Final feature selection
features = columns_to_be_kept_numerical + significant_categorical_columns
df = df[features + ['Approved_Flag']]




# Label encoding for categorical features
df['EDUCATION'] = df['EDUCATION'].replace({
    'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
    'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3
}).astype(int)
df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])



# Prepare data for training
y = df_encoded['Approved_Flag']
X = df_encoded.drop(['Approved_Flag'], axis=1)



# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)



# Hyperparameter tuning and model training with XGBoost
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)



# Evaluate the model
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Best Hyperparameters:", grid_search.best_params_)
print("Test Accuracy:", test_accuracy)



# # Test on the unseen dataset
# a3 = pd.read_excel("D:/Projects/creditcard_risk_management/Unseen_Dataset.xlsx")
# df_unseen = a3[features]
# df_unseen['EDUCATION'] = df_unseen['EDUCATION'].replace({
#     'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
#     'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3
# }).astype(int)
# df_encoded_unseen = pd.get_dummies(df_unseen, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])



# # Prediction on unseen dataset
# X_unseen = df_encoded_unseen
# unseen_predictions = best_model.predict(X_unseen)
# print("Predictions on unseen data:", unseen_predictions)



# Saving the model

import pickle

# Save the model
filename = 'best_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)

# Load the model
with open(filename, 'rb') as file:
    load_model = pickle.load(file)

# Verify the loaded model
print("Model loaded successfully:", load_model)







