# Hospital Readmission Prediction with Synthea Data
# Complete implementation from data generation through model building

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                             confusion_matrix, classification_report, auc)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 0: GENERATE SYNTHEA DATA
# ============================================================================
"""
Before running this code, generate Synthea data:

1. Download Synthea from: https://github.com/synthetichealth/synthea
2. Navigate to the synthea directory
3. Run the following command to generate 10,000 patients:
   
   ./run_synthea -p 10000
   
   Or on Windows:
   run_synthea.bat -p 10000

4. The data will be generated in the 'output/csv' folder
5. Update the DATA_PATH variable below to point to your output folder
"""

# Configuration
DATA_PATH = './output/csv/'  # Update this to your Synthea output path
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("HOSPITAL READMISSION PREDICTION PROJECT")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD SYNTHEA DATA
# ============================================================================
print("\n[STEP 1] Loading Synthea data files...")

try:
    # Load all relevant tables
    patients = pd.read_csv(f'{DATA_PATH}patients.csv')
    encounters = pd.read_csv(f'{DATA_PATH}encounters.csv')
    conditions = pd.read_csv(f'{DATA_PATH}conditions.csv')
    procedures = pd.read_csv(f'{DATA_PATH}procedures.csv')
    medications = pd.read_csv(f'{DATA_PATH}medications.csv')
    observations = pd.read_csv(f'{DATA_PATH}observations.csv')
    
    print(f"✓ Patients: {len(patients)} records")
    print(f"✓ Encounters: {len(encounters)} records")
    print(f"✓ Conditions: {len(conditions)} records")
    print(f"✓ Procedures: {len(procedures)} records")
    print(f"✓ Medications: {len(medications)} records")
    print(f"✓ Observations: {len(observations)} records")
except FileNotFoundError as e:
    print(f"\n❌ Error: Could not find Synthea data files.")
    print(f"Please update DATA_PATH to point to your Synthea output/csv folder")
    print(f"Current path: {DATA_PATH}")
    raise

# ============================================================================
# STEP 2: DATA EXPLORATION
# ============================================================================
print("\n[STEP 2] Exploring data structure...")

# Examine encounters to understand visit types
print("\nEncounter types:")
print(encounters['ENCOUNTERCLASS'].value_counts())

# Convert date columns and remove timezone info to avoid comparison issues
encounters['START'] = pd.to_datetime(encounters['START']).dt.tz_localize(None)
encounters['STOP'] = pd.to_datetime(encounters['STOP']).dt.tz_localize(None)
patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE']).dt.tz_localize(None)

# ============================================================================
# STEP 3: DEFINE READMISSION COHORT
# ============================================================================
print("\n[STEP 3] Defining readmission cohort...")

# Filter for inpatient admissions only
inpatient = encounters[encounters['ENCOUNTERCLASS'].isin(['inpatient', 'emergency'])].copy()
inpatient = inpatient.sort_values(['PATIENT', 'START'])

print(f"Total inpatient/emergency encounters: {len(inpatient)}")

# Calculate readmission flag
def identify_readmissions(df, readmit_window=30):
    """Identify 30-day readmissions"""
    df = df.sort_values(['PATIENT', 'START']).reset_index(drop=True)
    df['NEXT_ADMISSION'] = df.groupby('PATIENT')['START'].shift(-1)
    df['DAYS_TO_NEXT'] = (df['NEXT_ADMISSION'] - df['STOP']).dt.days
    df['READMITTED_30'] = (df['DAYS_TO_NEXT'] <= readmit_window) & (df['DAYS_TO_NEXT'] > 0)
    
    # Exclude the last admission for each patient (no opportunity for readmission)
    df['IS_LAST'] = df.groupby('PATIENT').cumcount(ascending=False) == 0
    
    return df[~df['IS_LAST']].copy()

inpatient = identify_readmissions(inpatient)

print(f"\nIndex admissions (excluding last per patient): {len(inpatient)}")
print(f"30-day readmissions: {inpatient['READMITTED_30'].sum()}")
print(f"Readmission rate: {inpatient['READMITTED_30'].mean():.2%}")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 4] Engineering features...")

# 4.1 Patient Demographics
patients['AGE'] = (datetime.now() - patients['BIRTHDATE']).dt.days / 365.25
patient_features = patients[['Id', 'BIRTHDATE', 'GENDER', 'RACE', 'ETHNICITY']].copy()
patient_features.columns = ['PATIENT', 'BIRTHDATE', 'GENDER', 'RACE', 'ETHNICITY']

# 4.2 Calculate age at admission
inpatient = inpatient.merge(patient_features, on='PATIENT', how='left')
inpatient['AGE_AT_ADMISSION'] = (inpatient['START'] - inpatient['BIRTHDATE']).dt.days / 365.25

# 4.3 Index admission features
inpatient['LENGTH_OF_STAY'] = (inpatient['STOP'] - inpatient['START']).dt.days
inpatient['DAY_OF_WEEK'] = inpatient['START'].dt.dayofweek
inpatient['MONTH'] = inpatient['START'].dt.month
inpatient['IS_WEEKEND'] = inpatient['DAY_OF_WEEK'].isin([5, 6]).astype(int)

# 4.4 Prior admission history
def calculate_prior_admissions(df):
    """Calculate number of prior admissions in various windows"""
    df = df.sort_values(['PATIENT', 'START'])
    
    result = []
    for idx, row in df.iterrows():
        patient_history = df[(df['PATIENT'] == row['PATIENT']) & 
                            (df['START'] < row['START'])]
        
        # Prior admissions in different windows
        days_30 = (row['START'] - timedelta(days=30))
        days_90 = (row['START'] - timedelta(days=90))
        days_180 = (row['START'] - timedelta(days=180))
        days_365 = (row['START'] - timedelta(days=365))
        
        prior_30 = len(patient_history[patient_history['START'] >= days_30])
        prior_90 = len(patient_history[patient_history['START'] >= days_90])
        prior_180 = len(patient_history[patient_history['START'] >= days_180])
        prior_365 = len(patient_history[patient_history['START'] >= days_365])
        
        result.append({
            'ENCOUNTER_ID': row['Id'],
            'PRIOR_ADMITS_30': prior_30,
            'PRIOR_ADMITS_90': prior_90,
            'PRIOR_ADMITS_180': prior_180,
            'PRIOR_ADMITS_365': prior_365
        })
    
    return pd.DataFrame(result)

print("Calculating prior admission history (this may take a minute)...")
prior_admits = calculate_prior_admissions(inpatient)
inpatient = inpatient.merge(prior_admits, left_on='Id', right_on='ENCOUNTER_ID', how='left')

# 4.5 Condition-based features
print("Aggregating condition data...")

# Get conditions up to each encounter
conditions['START'] = pd.to_datetime(conditions['START']).dt.tz_localize(None)
condition_counts = conditions.groupby('PATIENT').size().reset_index(name='TOTAL_CONDITIONS')

# High-risk conditions
high_risk_conditions = [
    'diabetes', 'heart failure', 'myocardial infarction', 'kidney', 
    'chronic obstructive', 'hypertension', 'stroke'
]

def count_high_risk(patient_id, encounter_date):
    patient_cond = conditions[(conditions['PATIENT'] == patient_id) & 
                              (conditions['START'] <= encounter_date)]
    desc_lower = patient_cond['DESCRIPTION'].str.lower()
    return sum(desc_lower.str.contains(cond, na=False).any() for cond in high_risk_conditions)

inpatient['HIGH_RISK_CONDITIONS'] = inpatient.apply(
    lambda x: count_high_risk(x['PATIENT'], x['START']), axis=1
)

inpatient = inpatient.merge(condition_counts, on='PATIENT', how='left')
inpatient['TOTAL_CONDITIONS'] = inpatient['TOTAL_CONDITIONS'].fillna(0)

# 4.6 Medication features
print("Aggregating medication data...")
med_counts = medications.groupby('PATIENT').size().reset_index(name='TOTAL_MEDICATIONS')
inpatient = inpatient.merge(med_counts, on='PATIENT', how='left')
inpatient['TOTAL_MEDICATIONS'] = inpatient['TOTAL_MEDICATIONS'].fillna(0)
inpatient['POLYPHARMACY'] = (inpatient['TOTAL_MEDICATIONS'] >= 5).astype(int)

# 4.7 Procedure features
print("Aggregating procedure data...")
proc_counts = procedures.groupby('ENCOUNTER').size().reset_index(name='NUM_PROCEDURES')
inpatient = inpatient.merge(proc_counts, left_on='Id', right_on='ENCOUNTER', how='left')
inpatient['NUM_PROCEDURES'] = inpatient['NUM_PROCEDURES'].fillna(0)

print(f"\n✓ Feature engineering complete. Total features: {len(inpatient.columns)}")

# ============================================================================
# STEP 5: PREPARE MODELING DATASET
# ============================================================================
print("\n[STEP 5] Preparing modeling dataset...")

# Select features for modeling
feature_cols = [
    'AGE_AT_ADMISSION', 'LENGTH_OF_STAY', 'PRIOR_ADMITS_30', 
    'PRIOR_ADMITS_90', 'PRIOR_ADMITS_180', 'PRIOR_ADMITS_365',
    'HIGH_RISK_CONDITIONS', 'TOTAL_CONDITIONS', 'TOTAL_MEDICATIONS',
    'POLYPHARMACY', 'NUM_PROCEDURES', 'DAY_OF_WEEK', 'MONTH', 'IS_WEEKEND',
    'GENDER', 'RACE', 'ETHNICITY'
]

# Create modeling dataset
modeling_data = inpatient[feature_cols + ['READMITTED_30']].copy()

# Handle missing values
modeling_data = modeling_data.fillna(0)

# Encode categorical variables
le_gender = LabelEncoder()
le_race = LabelEncoder()
le_ethnicity = LabelEncoder()

modeling_data['GENDER'] = le_gender.fit_transform(modeling_data['GENDER'])
modeling_data['RACE'] = le_race.fit_transform(modeling_data['RACE'])
modeling_data['ETHNICITY'] = le_ethnicity.fit_transform(modeling_data['ETHNICITY'])

print(f"Final dataset shape: {modeling_data.shape}")
print(f"Target distribution:\n{modeling_data['READMITTED_30'].value_counts()}")

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 6] Splitting data into train and test sets...")

X = modeling_data.drop('READMITTED_30', axis=1)
y = modeling_data['READMITTED_30'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Training readmission rate: {y_train.mean():.2%}")
print(f"Test readmission rate: {y_test.mean():.2%}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 7: MODEL TRAINING
# ============================================================================
print("\n[STEP 7] Training models...")

# 6.1 Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred_proba)
print(f"Logistic Regression AUC: {lr_auc:.4f}")

# 6.2 Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_STATE, 
    class_weight='balanced', max_depth=10
)
rf_model.fit(X_train, y_train)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_proba)
print(f"Random Forest AUC: {rf_auc:.4f}")

# 6.3 Gradient Boosting
print("\nTraining Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100, random_state=RANDOM_STATE, 
    learning_rate=0.1, max_depth=5
)
gb_model.fit(X_train, y_train)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test, gb_pred_proba)
print(f"Gradient Boosting AUC: {gb_auc:.4f}")

# ============================================================================
# STEP 9: MODEL EVALUATION AND VISUALIZATION
# ============================================================================
print("\n[STEP 7] Evaluating models...")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ROC Curves
ax1 = axes[0, 0]
for model_name, pred_proba, color in [
    ('Logistic Regression', lr_pred_proba, 'blue'),
    ('Random Forest', rf_pred_proba, 'green'),
    ('Gradient Boosting', gb_pred_proba, 'red')
]:
    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    auc_score = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})', color=color, lw=2)

ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves - Model Comparison')
ax1.legend()
ax1.grid(alpha=0.3)

# Precision-Recall Curves
ax2 = axes[0, 1]
for model_name, pred_proba, color in [
    ('Logistic Regression', lr_pred_proba, 'blue'),
    ('Random Forest', rf_pred_proba, 'green'),
    ('Gradient Boosting', gb_pred_proba, 'red')
]:
    precision, recall, _ = precision_recall_curve(y_test, pred_proba)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, label=f'{model_name} (AUC={pr_auc:.3f})', color=color, lw=2)

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves')
ax2.legend()
ax2.grid(alpha=0.3)

# Feature Importance (Random Forest)
ax3 = axes[1, 0]
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

ax3.barh(range(len(feature_importance)), feature_importance['importance'])
ax3.set_yticks(range(len(feature_importance)))
ax3.set_yticklabels(feature_importance['feature'])
ax3.set_xlabel('Importance')
ax3.set_title('Top 10 Feature Importances (Random Forest)')
ax3.invert_yaxis()

# Confusion Matrix (Best model - GB)
ax4 = axes[1, 1]
gb_pred = (gb_pred_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, gb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title('Confusion Matrix (Gradient Boosting)')

plt.tight_layout()
plt.savefig('readmission_model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Evaluation plots saved as 'readmission_model_evaluation.png'")

# Print classification report for best model
print("\n" + "="*80)
print("FINAL MODEL PERFORMANCE (Gradient Boosting)")
print("="*80)
print(classification_report(y_test, gb_pred, target_names=['No Readmission', 'Readmission']))

# Feature importance summary
print("\n" + "="*80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*80)
print(feature_importance.to_string(index=False))

print("\n" + "="*80)

print("="*80)
print("\nNext steps:")
print("1. Review the evaluation plots in 'readmission_model_evaluation.png'")
print("2. Consider hyperparameter tuning for better performance")
print("3. Implement cross-validation for more robust evaluation")
print("4. Add SHAP values for better model interpretability")
print("5. Deploy the model with a prediction pipeline")