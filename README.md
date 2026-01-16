# Synthea-ML-project
This project predicts which hospital patients are at risk of being readmitted within 30 days of discharge. Early identification allows hospitals to provide targeted interventions, reducing readmissions and improving patient outcomes.

We Create 10,000 synthetic patients using Synthea, and using 3 different Models such as Logistic regression, Random Forest, and Gradient Boosting, we then evaluate their performance.

# Reasons for Imports 
```
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
```

pandas: Data manipulation library for working with tabular data (DataFrames)  
numpy: Numerical computing library for mathematical operations and array handling  

datetime: For working with dates and times (calculating days between admissions)  
timedelta: For adding/subtracting time periods (e.g., 30 days, 90 days)  

train_test_split: Splits data into training and testing sets  
cross_val_score: Evaluates model performance using k-fold cross-validation  
GridSearchCV: Systematic hyperparameter tuning (not used in this version but imported for future use  

StandardScaler: Normalizes features to have mean=0 and std=1 (important for logistic regression)  
LabelEncoder: Converts categorical text labels to numeric values (e.g., "M"→0, "F"→1)  

LogisticRegression: Linear model for binary classification (baseline model)  
RandomForestClassifier: Ensemble of decision trees (handles non-linear patterns)  
GradientBoostingClassifier: Sequential tree-based model (often highest performance)  

Synthea:
https://github.com/synthetichealth/synthea
