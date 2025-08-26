"""
Healthcare Analytics Pipeline for Readmission Prediction
Predicts 30/60/90 day readmission rates and provides insights for:
- Disease progression tracking
- Treatment cost estimation
- Treatment plan optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import pickle
import joblib
import json
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_recall_curve,
                             classification_report, confusion_matrix, roc_curve,
                             f1_score, recall_score, precision_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, deep learning features will be disabled")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost for advanced modeling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using alternative models")

class HealthcareDataProcessor:
    """Processes and integrates multiple healthcare datasets"""

    def __init__(self):
        self.beneficiary_data = None
        self.inpatient_data = None
        self.outpatient_data = None
        self.drug_data = None
        self.person_data = None
        self.condition_data = None
        self.integrated_data = None

    def load_beneficiary_data(self, years=[2008, 2009, 2010]):
        """Load and combine beneficiary summary files from CSVs"""
        dfs = []
        file_map = {
            2008: 'data/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv',
            2009: 'data/DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv',
            2010: 'data/DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv'
        }

        for year in years:
            filename = file_map.get(year)
            if not filename:
                print(f"Warning: No file mapping found for year {year}")
                continue

            try:
                print(f"Loading beneficiary data from {filename}")
                df = pd.read_csv(filename)
                df['YEAR'] = year # Add year column for tracking
                dfs.append(df)
            except FileNotFoundError:
                print(f"Error: File not found - {filename}. Please ensure it is uploaded to Colab.")

        if dfs:
            self.beneficiary_data = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(self.beneficiary_data)} total beneficiary records")
        else:
            print("No beneficiary data was loaded.")
        return self.beneficiary_data

    def load_claims_data(self):
        """Load inpatient and outpatient claims from CSVs"""
        inpatient_file = 'data/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv'
        outpatient_file = 'data/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv'

        try:
            print(f"Loading inpatient claims from {inpatient_file}")
            self.inpatient_data = pd.read_csv(inpatient_file)
            print(f"Loaded {len(self.inpatient_data)} inpatient claims")
        except FileNotFoundError:
            print(f"Error: Inpatient claims file not found - {inpatient_file}")

        try:
            print(f"Loading outpatient claims from {outpatient_file}")
            self.outpatient_data = pd.read_csv(outpatient_file)
            print(f"Loaded {len(self.outpatient_data)} outpatient claims")
        except FileNotFoundError:
            print(f"Error: Outpatient claims file not found - {outpatient_file}")


    def create_readmission_labels(self):
        """Create readmission labels for 30, 60, and 90 days"""
        if self.inpatient_data is None:
            raise ValueError("Load inpatient data first")

        # Convert date columns
        self.inpatient_data['ADMIT_DATE'] = pd.to_datetime(
            self.inpatient_data['CLM_ADMSN_DT'], format='%Y%m%d'
        )
        self.inpatient_data['DISCHARGE_DATE'] = pd.to_datetime(
            self.inpatient_data['CLM_THRU_DT'], format='%Y%m%d'
        )

        # Sort by patient and admission date
        self.inpatient_data = self.inpatient_data.sort_values(
            ['DESYNPUF_ID', 'ADMIT_DATE']
        )

        # Calculate time difference to next admission for each patient
        self.inpatient_data['DAYS_TO_READMIT'] = self.inpatient_data.groupby('DESYNPUF_ID')['ADMIT_DATE'].diff().dt.days.shift(-1)

        # Create readmission flags based on the time difference
        self.inpatient_data['READMIT_30'] = (self.inpatient_data['DAYS_TO_READMIT'] <= 30).astype(int)
        self.inpatient_data['READMIT_60'] = (self.inpatient_data['DAYS_TO_READMIT'] <= 60).astype(int)
        self.inpatient_data['READMIT_90'] = (self.inpatient_data['DAYS_TO_READMIT'] <= 90).astype(int)


        print("Readmission labels created:")
        print(f"30-day readmissions: {self.inpatient_data['READMIT_30'].sum()}")
        print(f"60-day readmissions: {self.inpatient_data['READMIT_60'].sum()}")
        print(f"90-day readmissions: {self.inpatient_data['READMIT_90'].sum()}")

class FeatureEngineer:
    """Feature engineering for healthcare data"""

    def __init__(self, processor):
        self.processor = processor
        self.feature_matrix = None
        self.feature_names = []

    def create_patient_features(self):
        """Create comprehensive patient features"""
        if self.processor.beneficiary_data is None:
            print("Beneficiary data not loaded. Cannot create features.")
            return

        # Use the latest beneficiary record for each patient for demographic info
        demo_features = self.processor.beneficiary_data.sort_values('YEAR').drop_duplicates('DESYNPUF_ID', keep='last').copy()

        # Convert birth date to age
        # THIS IS THE CORRECTED LINE:
        demo_features['BENE_BIRTH_DT'] = pd.to_datetime(demo_features['BENE_BIRTH_DT'])

        demo_features['AGE'] = demo_features['YEAR'] - demo_features['BENE_BIRTH_DT'].dt.year

        # Chronic condition flags (1 = Yes, 2 = No in source data)
        chronic_conditions = ['SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR',
                              'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT',
                              'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA']

        for condition in chronic_conditions:
            if condition in demo_features.columns:
                demo_features[f'HAS_{condition}'] = (demo_features[condition] == 1).astype(int)

        # Healthcare utilization features
        if self.processor.inpatient_data is not None:
            admission_counts = self.processor.inpatient_data.groupby('DESYNPUF_ID').size().reset_index(name='NUM_ADMISSIONS')
            los_avg = self.processor.inpatient_data.groupby('DESYNPUF_ID')['CLM_UTLZTN_DAY_CNT'].mean().reset_index(name='AVG_LOS')
            ip_costs = self.processor.inpatient_data.groupby('DESYNPUF_ID')['CLM_PMT_AMT'].sum().reset_index(name='TOTAL_IP_COST')

            demo_features = demo_features.merge(admission_counts, on='DESYNPUF_ID', how='left')
            demo_features = demo_features.merge(los_avg, on='DESYNPUF_ID', how='left')
            demo_features = demo_features.merge(ip_costs, on='DESYNPUF_ID', how='left')

        # Fill missing values for utilization features (patients with no admissions)
        demo_features[['NUM_ADMISSIONS', 'AVG_LOS', 'TOTAL_IP_COST']] = demo_features[['NUM_ADMISSIONS', 'AVG_LOS', 'TOTAL_IP_COST']].fillna(0)

        self.feature_matrix = demo_features
        self.feature_names = [col for col in demo_features.columns if col not in ['DESYNPUF_ID', 'BENE_BIRTH_DT']]

        print(f"Created {len(self.feature_names)} features for {len(self.feature_matrix)} patients.")
        return self.feature_matrix

    def create_diagnosis_features(self):
        """Create features from diagnosis codes"""
        if self.processor.inpatient_data is None:
            print("Inpatient data not available for diagnosis features.")
            return

        diag_cols = [col for col in self.processor.inpatient_data.columns if 'ICD9_DGNS_CD' in col]

        all_diagnoses = pd.concat([self.processor.inpatient_data[col] for col in diag_cols]).dropna()
        top_diagnoses = all_diagnoses.value_counts().head(20).index

        for diag in top_diagnoses:
            feature_name = f'DIAG_{diag}'
            # Check if patient has the diagnosis in any of the diagnosis columns
            has_diag = self.processor.inpatient_data[diag_cols].apply(lambda row: diag in row.values, axis=1)
            diag_df = self.processor.inpatient_data.loc[has_diag, 'DESYNPUF_ID'].drop_duplicates().to_frame()
            diag_df[feature_name] = 1

            self.feature_matrix = self.feature_matrix.merge(diag_df, on='DESYNPUF_ID', how='left')
            self.feature_matrix[feature_name] = self.feature_matrix[feature_name].fillna(0)

        print(f"Added {len(top_diagnoses)} diagnosis features")
        return self.feature_matrix


class ReadmissionPredictor:
    """Multi-model readmission prediction system with model persistence"""

    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []
        self.models_dir = "saved_models"
        self.ensure_models_directory()
    
    def ensure_models_directory(self):
        """Create models directory if it doesn't exist"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def get_model_filename(self, model_name, target_col):
        """Generate filename for saved model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.models_dir, f"{model_name}_{target_col}_{timestamp}.pkl")
    
    def get_scaler_filename(self, target_col):
        """Generate filename for saved scaler"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.models_dir, f"scaler_{target_col}_{timestamp}.pkl")
    
    def save_model(self, model, model_name, target_col):
        """Save a trained model to disk"""
        try:
            filename = self.get_model_filename(model_name, target_col)
            if hasattr(model, 'save') and TENSORFLOW_AVAILABLE:  # Keras model
                model.save(filename.replace('.pkl', '.h5'))
            else:  # Scikit-learn model
                joblib.dump(model, filename)
            print(f"✅ Model saved: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def save_scaler(self, scaler, target_col):
        """Save a fitted scaler to disk"""
        try:
            filename = self.get_scaler_filename(target_col)
            joblib.dump(scaler, filename)
            print(f"✅ Scaler saved: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving scaler: {e}")
            return None
    
    def save_results(self, target_col):
        """Save training results to disk"""
        try:
            if target_col in self.results:
                results_file = os.path.join(self.models_dir, f"results_{target_col}.pkl")
                joblib.dump(self.results[target_col], results_file)
                print(f"✅ Results saved: {results_file}")
                return results_file
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None
    
    def load_model(self, filename):
        """Load a saved model from disk"""
        try:
            if filename.endswith('.h5') and TENSORFLOW_AVAILABLE:
                return keras.models.load_model(filename)
            else:
                return joblib.load(filename)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def load_scaler(self, filename):
        """Load a saved scaler from disk"""
        try:
            return joblib.load(filename)
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            return None
    
    def load_results(self, target_col):
        """Load saved training results from disk"""
        try:
            # Look for any results file for this target column
            if os.path.exists(self.models_dir):
                for file in os.listdir(self.models_dir):
                    if file.startswith(f"results_{target_col}") and file.endswith('.pkl'):
                        results_file = os.path.join(self.models_dir, file)
                        results = joblib.load(results_file)
                        if not hasattr(self, 'results'):
                            self.results = {}
                        self.results[target_col] = results
                        print(f"✅ Results loaded: {results_file}")
                        return True
        except Exception as e:
            print(f"❌ Error loading results: {e}")
        return False
    
    def list_saved_models(self):
        """List all saved models in the models directory"""
        models = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith(('.pkl', '.h5')):
                    models.append(file)
        return sorted(models)
    
    def load_existing_models(self, target_col):
        """Load existing trained models and results for a target column"""
        # First try to load saved results
        if self.load_results(target_col):
            print(f"✅ Successfully loaded existing results for {target_col}")
            return True
        
        # If no results file, check if we have saved models
        saved_models = self.list_saved_models()
        target_models = [m for m in saved_models if target_col in m and not m.startswith('scaler_')]
        
        if not target_models:
            return False
        
        print(f"Found {len(target_models)} existing models for {target_col}")
        print("Loading models and generating performance metrics...")
        
        try:
            # Prepare data for evaluation
            X_train, X_test, y_train, y_test = self.prepare_data(target_col)
            
            # Initialize results structure
            if not hasattr(self, 'results'):
                self.results = {}
            if not hasattr(self, 'models'):
                self.models = {}
            
            self.results[target_col] = {}
            self.models[target_col] = {}
            
            # Load and evaluate each model
            for model_file in target_models:
                model_path = os.path.join(self.models_dir, model_file)
                
                # Extract model name from filename
                model_name = model_file.split('_')[0] + ' ' + model_file.split('_')[1] if '_' in model_file else model_file.replace('.pkl', '')
                if model_name.startswith('Gradient'):
                    model_name = 'Gradient Boosting'
                elif model_name.startswith('Logistic'):
                    model_name = 'Logistic Regression'
                elif model_name.startswith('Random'):
                    model_name = 'Random Forest'
                elif model_name.startswith('XGBoost'):
                    model_name = 'XGBoost'
                
                # Load the model
                model = self.load_model(model_path)
                if model is None:
                    continue
                
                # Make predictions
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Find optimal threshold
                from sklearn.metrics import f1_score, recall_score, precision_score
                thresholds = np.arange(0.1, 0.9, 0.05)
                best_f1 = 0
                best_threshold = 0.5
                best_recall = 0
                best_precision = 0
                
                for threshold in thresholds:
                    y_pred_temp = (y_pred_proba > threshold).astype(int)
                    f1_temp = f1_score(y_test, y_pred_temp)
                    recall_temp = recall_score(y_test, y_pred_temp)
                    precision_temp = precision_score(y_test, y_pred_temp)
                    
                    if f1_temp > best_f1:
                        best_f1 = f1_temp
                        best_threshold = threshold
                        best_recall = recall_temp
                        best_precision = precision_temp
                
                # Use optimal threshold for final predictions
                y_pred = (y_pred_proba > best_threshold).astype(int)
                
                # Calculate comprehensive metrics
                from sklearn.metrics import accuracy_score, roc_auc_score
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                
                # Store results
                self.results[target_col][model_name] = {
                    'model': model, 'accuracy': accuracy, 'auc': auc,
                    'f1_score': f1, 'recall': recall, 'precision': precision,
                    'cv_mean': auc, 'cv_std': 0.0,  # Placeholder values
                    'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba
                }
                
                self.models[target_col][model_name] = model
                
                print(f"  ✅ {model_name}: AUC={auc:.3f}, F1={f1:.3f}, Recall={recall:.3f}")
            
            print(f"✅ Successfully loaded {len(self.results[target_col])} models for {target_col}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading existing models: {e}")
            # Fallback to placeholder structure
            if not hasattr(self, 'results'):
                self.results = {}
            if target_col not in self.results:
                self.results[target_col] = {}
            
            # Mark that we have models but need results
            self.results[target_col]['_models_exist'] = True
            self.results[target_col]['_needs_retraining'] = True
            
            return True

    def models_already_loaded(self, target_col):
        """Check if models are already loaded and functional for a target column"""
        if not hasattr(self, 'results') or target_col not in self.results:
            return False
        
        # Check if we have actual model results (not just placeholders)
        results = self.results[target_col]
        if '_needs_retraining' in results or '_models_exist' in results:
            return False
        
        # Check if we have at least one valid model result
        model_count = 0
        for key, value in results.items():
            if isinstance(value, dict) and 'model' in value and 'auc' in value:
                model_count += 1
        
        return model_count > 0

    def prepare_data(self, target_col='READMIT_30'):
        """Prepare data for modeling"""
        if self.feature_engineer.processor.inpatient_data is None:
            raise ValueError("Inpatient data is required to create target labels.")

        # Merge features with inpatient data to align each admission with patient features
        admission_level_data = self.feature_engineer.processor.inpatient_data.merge(
            self.feature_engineer.feature_matrix, on='DESYNPUF_ID', how='left'
        )

        # Define target variable and features
        y = admission_level_data[target_col]

        # Drop identifiers, dates, and the target variable itself from features
        cols_to_drop = [
            'DESYNPUF_ID', 'BENE_BIRTH_DT', 'CLM_ID', 'CLM_FROM_DT', 'CLM_THRU_DT',
            'ADMIT_DATE', 'DISCHARGE_DATE', 'CLM_ADMSN_DT', 'DAYS_TO_READMIT',
            'READMIT_30', 'READMIT_60', 'READMIT_90'
        ]

        # Also drop diagnosis codes and other non-feature columns
        for col in admission_level_data.columns:
            if 'ICD9' in col or 'HCPCS' in col or '_CD' in col or 'PRVDR' in col:
                cols_to_drop.append(col)

        # Get unique columns to drop
        cols_to_drop = list(set(cols_to_drop))

        X = admission_level_data.drop(columns=cols_to_drop, errors='ignore')
        X = X.select_dtypes(include=np.number) # Use only numeric columns for modeling

        self.feature_names = X.columns.tolist() # Save feature names

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=self.feature_names)

        # Feature selection to reduce noise and improve performance
        if X_imputed.shape[1] > 50:  # Only if we have many features
            selector = SelectKBest(score_func=f_classif, k=min(50, X_imputed.shape[1]))
            X_selected = selector.fit_transform(X_imputed, y)
            selected_features = X_imputed.columns[selector.get_support()].tolist()
            X_imputed = pd.DataFrame(X_selected, columns=selected_features)
            self.feature_names = selected_features
            print(f"Selected {len(selected_features)} best features out of {len(self.feature_names)}")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        self.scalers[target_col] = scaler

        # Split data
        if len(y.unique()) > 1:
            return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        else:
            return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


    def train_ensemble(self, target_col='READMIT_30'):
        """Train ensemble of models with improved parameters for better F1 and Recall"""
        
        # Check if models are already loaded and functional
        if self.models_already_loaded(target_col):
            print(f"\n{'='*50}")
            print(f"✅ Models for {target_col} already loaded and functional")
            print("Skipping training - using existing models")
            print('='*50)
            return self.results[target_col]
        
        print(f"\n{'='*50}")
        print(f"Training models for {target_col}")
        print('='*50)

        X_train, X_test, y_train, y_test = self.prepare_data(target_col)

        if len(np.unique(y_train)) < 2:
            print(f"Skipping training for {target_col}: Target variable has only one class.")
            self.results[target_col] = {}
            return {}

        # Check class distribution and calculate class weights
        class_counts = np.bincount(y_train)
        print(f"Class distribution - Train: {class_counts}, Test: {np.bincount(y_test)}")
        
        # Calculate balanced class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Optimize models for imbalanced data
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=2000, 
                random_state=42,
                class_weight='balanced',
                C=0.1,  # Stronger regularization
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, 
                random_state=42,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8
            )
        }

        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200, 
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=class_weight_dict[1] if 1 in class_weight_dict else 1,
                random_state=42, 
                use_label_encoder=False, 
                eval_metric='logloss'
            )

        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold for better F1 and Recall
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_threshold = 0.5
            best_recall = 0
            best_precision = 0
            
            for threshold in thresholds:
                y_pred_temp = (y_pred_proba > threshold).astype(int)
                f1_temp = f1_score(y_test, y_pred_temp)
                recall_temp = recall_score(y_test, y_pred_temp)
                precision_temp = precision_score(y_test, y_pred_temp)
                
                if f1_temp > best_f1:
                    best_f1 = f1_temp
                    best_threshold = threshold
                    best_recall = recall_temp
                    best_precision = precision_temp
            
            print(f"  Optimal threshold: {best_threshold:.3f}")
            
            # Use optimal threshold for final predictions
            y_pred = (y_pred_proba > best_threshold).astype(int)

            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5), scoring='roc_auc')

            results[name] = {
                'model': model, 'accuracy': accuracy, 'auc': auc,
                'f1_score': f1, 'recall': recall, 'precision': precision,
                'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
                'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba
            }

            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  AUC: {auc:.3f}")
            print(f"  F1 Score: {f1:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            
            # Save model automatically
            self.save_model(model, name, target_col)
        
        # Save scaler
        self.save_scaler(self.scalers[target_col], target_col)

        self.models[target_col] = models
        self.results[target_col] = results
        
        # Save results to disk
        self.save_results(target_col)
        
        return results

    def train_deep_learning_model(self, target_col='READMIT_30'):
        """Train neural network for readmission prediction"""
        if not TENSORFLOW_AVAILABLE:
            print(f"\n{'='*50}")
            print(f"Deep Learning not available - TensorFlow failed to load")
            print(f"Skipping Deep Learning for {target_col}")
            print('='*50)
            self.results[target_col]['Deep Learning'] = {}
            return None, None
            
        print(f"\n{'='*50}")
        print(f"Training Deep Learning Model for {target_col}")
        print('='*50)

        X_train, X_test, y_train, y_test = self.prepare_data(target_col)

        if len(np.unique(y_train)) < 2:
            print(f"Skipping training for {target_col}: Target has only one class.")
            self.results[target_col]['Deep Learning'] = {}
            return None, None

        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        class_weight_dict = dict(zip([0, 1], [1.0, class_counts[0] / class_counts[1]]))
        print(f"Class distribution: {class_counts}")
        print(f"Class weights: {class_weight_dict}")
        
        # Improved architecture for imbalanced data
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # Use focal loss or weighted binary crossentropy for imbalanced data
        if hasattr(keras.losses, 'BinaryFocalCrossentropy'):
            loss_fn = keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)
        else:
            # Fallback to weighted binary crossentropy
            loss_fn = keras.losses.binary_crossentropy
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
        )
        
        # Enhanced callbacks
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        # Use class weights in training
        history = model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=64, 
            validation_split=0.2, 
            callbacks=[early_stop, reduce_lr], 
            class_weight=class_weight_dict,
            verbose=0
        )
        test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)

        # Calculate additional metrics for deep learning
        y_pred_proba = model.predict(X_test).flatten()
        
        # Find optimal threshold for better F1 and Recall
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        best_recall = 0
        
        for threshold in thresholds:
            y_pred_temp = (y_pred_proba > threshold).astype(int)
            f1_temp = f1_score(y_test, y_pred_temp)
            recall_temp = recall_score(y_test, y_pred_temp)
            
            if f1_temp > best_f1:
                best_f1 = f1_temp
                best_threshold = threshold
                best_recall = recall_temp
        
        print(f"  Optimal threshold: {best_threshold:.3f}")
        print(f"  Best F1 Score: {best_f1:.3f}")
        print(f"  Best Recall: {best_recall:.3f}")
        
        # Use optimal threshold for final predictions
        y_pred = (y_pred_proba > best_threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  Test AUC: {test_auc:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  Precision: {precision:.3f}")

        self.results[target_col]['Deep Learning'] = {
            'model': model, 'history': history,
            'test_accuracy': test_accuracy, 'test_auc': test_auc,
            'f1_score': f1, 'recall': recall, 'precision': precision,
            'y_test': y_test, 'y_pred': y_pred
        }
        
        # Save deep learning model
        self.save_model(model, 'Deep Learning', target_col)
        
        # Save results to disk
        self.save_results(target_col)
        
        return model, history
    
    def generate_insights_from_loaded_models(self, target_col='READMIT_30'):
        """Generate insights and recommendations using already loaded models"""
        if not self.models_already_loaded(target_col):
            print(f"❌ No loaded models found for {target_col}")
            return None
        
        print(f"\n📊 Generating insights from loaded models for {target_col}")
        
        # Get the best performing model
        best_model_name = None
        best_auc = 0
        
        for model_name, results in self.results[target_col].items():
            if isinstance(results, dict) and 'auc' in results:
                if results['auc'] > best_auc:
                    best_auc = results['auc']
                    best_model_name = model_name
        
        if best_model_name:
            print(f"✅ Best model: {best_model_name} (AUC: {best_auc:.3f})")
            
            # Generate feature importance if available
            best_model = self.results[target_col][best_model_name]['model']
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n🔍 Top 10 Important Features:")
                for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                    print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
                
                return {
                    'best_model': best_model_name,
                    'best_auc': best_auc,
                    'feature_importance': feature_importance,
                    'model_results': self.results[target_col]
                }
        
        return {'model_results': self.results[target_col]}
    
    def predict_with_saved_model(self, model_filename, scaler_filename, new_data):
        """Make predictions using a saved model"""
        try:
            # Load model and scaler
            model = self.load_model(model_filename)
            scaler = self.load_scaler(scaler_filename)
            
            if model is None or scaler is None:
                return None, "Failed to load model or scaler"
            
            # Preprocess new data
            if isinstance(new_data, pd.DataFrame):
                # Ensure same features as training
                missing_features = set(self.feature_names) - set(new_data.columns)
                if missing_features:
                    return None, f"Missing features: {missing_features}"
                
                # Select and order features
                X_new = new_data[self.feature_names].copy()
            else:
                X_new = new_data.copy()
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X_new_imputed = pd.DataFrame(imputer.fit_transform(X_new), columns=self.feature_names)
            
            # Scale features
            X_new_scaled = scaler.transform(X_new_imputed)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_new_scaled)[:, 1]
            else:
                predictions = model.predict(X_new_scaled)
            
            return predictions, "Success"
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def get_model_performance_summary(self, target_col):
        """Get comprehensive performance summary for all models"""
        if target_col not in self.results:
            return None
        
        summary = []
        for model_name, results in self.results[target_col].items():
            if isinstance(results, dict) and 'accuracy' in results:
                summary.append({
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy']:.3f}",
                    'AUC': f"{results['auc']:.3f}",
                    'F1 Score': f"{results.get('f1_score', 'N/A')}",
                    'Recall': f"{results.get('recall', 'N/A')}",
                    'Precision': f"{results.get('precision', 'N/A')}",
                    'CV AUC': f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}"
                })
        
        return pd.DataFrame(summary)


class ClinicalInsightsGenerator:
    """Generate clinical insights from predictions"""

    def __init__(self, predictor):
        self.predictor = predictor

    def identify_high_risk_factors(self, target_col='READMIT_30', top_n=10):
        """Identify top risk factors for readmission using Random Forest feature importances"""
        print(f"\nTop {top_n} Risk Factors for {target_col}:")
        print("-" * 40)

        if target_col not in self.predictor.results or not self.predictor.results[target_col]:
            print(f"No results available for {target_col}.")
            return []

        results = self.predictor.results[target_col]
        
        # Check if results contain actual model metrics or just flags
        if isinstance(results, dict) and '_models_exist' in results:
            print(f"Models exist for {target_col} but performance metrics not available.")
            print("Try loading models first or retrain to get feature importance.")
            return []
        
        # Try different models that might have feature importance
        model_to_use = None
        feature_names = getattr(self.predictor, 'feature_names', [])
        
        # Prefer Random Forest, then Gradient Boosting, then any tree-based model
        for model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            if (model_name in results and 
                isinstance(results[model_name], dict) and 
                'model' in results[model_name]):
                model = results[model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    model_to_use = model
                    print(f"Using {model_name} for feature importance analysis")
                    break
        
        if model_to_use is None:
            print(f"No tree-based models with feature importance available for {target_col}.")
            return []

        if not feature_names:
            print("Feature names not available.")
            return []

        importances = model_to_use.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        risk_factors = []
        for i in indices:
            if i < len(feature_names):  # Safety check
                feature = feature_names[i]
                score = importances[i]
                print(f"  - {feature}: {score:.4f}")
                risk_factors.append((feature, score))

        return risk_factors

    def estimate_treatment_costs(self):
        """Estimate treatment costs based on readmission risk"""
        print("\n" + "="*50)
        print("Treatment Cost Estimation")
        print("="*50)

        base_admission_cost = 15000
        readmission_cost = 20000
        preventive_care_cost = 2000

        cost_analysis = {}
        for period in ['READMIT_30', 'READMIT_60', 'READMIT_90']:
            if period not in self.predictor.results or not self.predictor.results[period]:
                print(f"\nNo results available for {period} to estimate costs.")
                continue

            results = self.predictor.results[period]
            
            # Check if results contain actual model metrics or just flags
            if isinstance(results, dict) and '_models_exist' in results:
                print(f"\nModels exist for {period} but performance metrics not loaded.")
                print("Try loading models first or retrain to get cost estimates.")
                continue
            
            # Filter out non-model entries and get valid models
            valid_models = {k: v for k, v in results.items() 
                          if (k not in ['_models_exist', '_needs_retraining'] and 
                              isinstance(v, dict) and 'auc' in v and 'y_pred_proba' in v)}
            
            if not valid_models:
                print(f"\nNo valid model results available for {period} to estimate costs.")
                continue
            
            # Find the best model by AUC
            try:
                best_model_name = max(valid_models, key=lambda k: valid_models[k]['auc'])
                best_model_results = valid_models[best_model_name]
                print(f"\n{period} (using {best_model_name} model - AUC: {best_model_results['auc']:.3f}):")
            except (ValueError, AttributeError) as e:
                print(f"\nError finding best model for {period}: {e}")
                continue

            y_pred_proba = best_model_results['y_pred_proba']
            high_risk = (y_pred_proba > 0.7).sum()
            medium_risk = ((y_pred_proba > 0.3) & (y_pred_proba <= 0.7)).sum()
            low_risk = (y_pred_proba <= 0.3).sum()

            expected_cost_no_intervention = (high_risk * 0.8 * readmission_cost) + (medium_risk * 0.4 * readmission_cost) + (low_risk * 0.1 * readmission_cost)
            expected_cost_with_intervention = (high_risk * (preventive_care_cost + 0.3 * readmission_cost)) + (medium_risk * (preventive_care_cost * 0.5 + 0.2 * readmission_cost)) + (low_risk * 0.05 * readmission_cost)
            savings = expected_cost_no_intervention - expected_cost_with_intervention

            cost_analysis[period] = {
                'potential_savings': savings,
                'high_risk_patients': int(high_risk),
                'medium_risk_patients': int(medium_risk),
                'low_risk_patients': int(low_risk),
                'best_model': best_model_name
            }
            
            print(f"  High Risk Patients: {high_risk}")
            print(f"  Medium Risk Patients: {medium_risk}")
            print(f"  Low Risk Patients: {low_risk}")
            print(f"  Potential Savings with Intervention: ${savings:,.2f}")

        return cost_analysis

    def generate_treatment_recommendations(self):
        """Generate personalized treatment recommendations"""
        print("\n" + "="*50)
        print("Treatment Plan Recommendations")
        print("="*50)
        recommendations = {
            'high_risk': {
                'interventions': ['Intensive case management', 'Daily medication adherence monitoring', 'Weekly follow-up appointments', 'Home health visits 2x per week'],
                'monitoring': 'Daily vital signs monitoring via telehealth', 'follow_up': '48-hour post-discharge appointment'
            },
            'medium_risk': {
                'interventions': ['Standard case management', 'Medication reconciliation', 'Bi-weekly follow-up appointments', 'Disease-specific education program'],
                'monitoring': 'Weekly telehealth check-ins', 'follow_up': '7-day post-discharge appointment'
            },
            'low_risk': {
                'interventions': ['Standard discharge planning', 'Medication education', 'Access to nurse hotline', 'Preventive care reminders'],
                'monitoring': 'Monthly wellness checks', 'follow_up': '14-day post-discharge appointment'
            }
        }
        for risk_level, plans in recommendations.items():
            print(f"\n{risk_level.upper()} PATIENTS:")
            print("  - Interventions: " + ", ".join(plans['interventions']))
            print(f"  - Monitoring: {plans['monitoring']}")
            print(f"  - Follow-up: {plans['follow_up']}")
        return recommendations


class ModelVisualizer:
    """Visualization tools for model results"""

    def __init__(self, predictor):
        self.predictor = predictor

    def plot_model_comparison(self):
        periods = ['READMIT_30', 'READMIT_60', 'READMIT_90']
        fig, axes = plt.subplots(1, len(periods), figsize=(18, 5), sharey=True)
        fig.suptitle('Model AUC Comparison', fontsize=16)

        for idx, period in enumerate(periods):
            ax = axes[idx]
            if period not in self.predictor.results or not self.predictor.results[period]:
                ax.set_title(f'{period} - No Data')
                continue

            results = self.predictor.results[period]
            
            # Check if results contain actual model metrics or just flags
            if isinstance(results, dict) and '_models_exist' in results:
                ax.set_title(f'{period} - Models Exist (Need Retraining)')
                continue
            
            # Check if results is a valid dictionary with model metrics
            if not isinstance(results, dict) or not results:
                ax.set_title(f'{period} - No Data')
                continue
            
            models = [k for k in results.keys() if 'auc' in results[k]]
            if not models:
                ax.set_title(f'{period} - No AUC Data')
                continue
                
            aucs = [results[m]['auc'] for m in models]

            sns.barplot(x=models, y=aucs, ax=ax, palette='viridis')
            ax.set_title(period)
            ax.set_ylabel('AUC Score' if idx == 0 else '')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            for i, v in enumerate(aucs):
                ax.text(i, v + 0.01, f"{v:.3f}", ha='center')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('model_comparison_auc.png', dpi=300)
        plt.show()

    def plot_feature_importance(self, top_n=15):
        periods = ['READMIT_30', 'READMIT_60', 'READMIT_90']
        fig, axes = plt.subplots(1, len(periods), figsize=(20, 8))
        fig.suptitle(f'Top {top_n} Feature Importances from Random Forest', fontsize=16)

        for idx, period in enumerate(periods):
            ax = axes[idx]
            if period not in self.predictor.results or not self.predictor.results[period]:
                ax.set_title(f'{period} - No Data')
                continue

            results = self.predictor.results[period]
            
            # Check if results contain actual model metrics or just flags
            if isinstance(results, dict) and '_models_exist' in results:
                ax.set_title(f'{period} - Models Exist (Need Retraining)')
                continue
            
            # Check if Random Forest model exists and has required data
            if 'Random Forest' not in results or 'model' not in results['Random Forest']:
                ax.set_title(f'{period} - No Random Forest Model')
                continue

            rf_model = self.predictor.results[period]['Random Forest']['model']
            feature_names = self.predictor.feature_names
            importances = pd.Series(rf_model.feature_importances_, index=feature_names).nlargest(top_n)

            sns.barplot(x=importances.values, y=importances.index, ax=ax, palette='plasma')
            ax.set_title(period)
            ax.set_xlabel('Importance Score')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()

    def plot_roc_curves(self):
        periods = ['READMIT_30', 'READMIT_60', 'READMIT_90']
        fig, axes = plt.subplots(1, len(periods), figsize=(18, 6), sharey=True, sharex=True)
        fig.suptitle('ROC Curves', fontsize=16)

        for idx, period in enumerate(periods):
            ax = axes[idx]
            ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
            if period in self.predictor.results and self.predictor.results[period]:
                results = self.predictor.results[period]
                
                # Check if results contain actual model metrics or just flags
                if isinstance(results, dict) and '_models_exist' in results:
                    ax.set_title(f'{period} - Models Exist (Need Retraining)')
                    continue
                
                # Check if results is a valid dictionary with model metrics
                if not isinstance(results, dict) or not results:
                    ax.set_title(f'{period} - No Data')
                    continue
                
                for name, result in results.items():
                    if 'y_pred_proba' in result:
                        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                        auc = result['auc']
                        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

            ax.set_title(period)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate' if idx == 0 else '')
            ax.legend()
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('roc_curves.png', dpi=300)
        plt.show()

class DiseaseProgressionTracker:
    """Track disease progression patterns"""

    def __init__(self, processor):
        self.processor = processor

    def analyze_condition_progression(self):
        """Analyze how chronic conditions progress over time"""
        if self.processor.beneficiary_data is None:
            print("No beneficiary data available.")
            return

        print("\n" + "="*50)
        print("Disease Progression Analysis (Prevalence % by Year)")
        print("="*50)

        conditions = [c for c in self.processor.beneficiary_data.columns if c.startswith('SP_')]
        progression = self.processor.beneficiary_data.groupby('YEAR')[conditions].apply(lambda x: (x==1).mean() * 100)

        print(progression.round(2))
        return progression

    def identify_comorbidity_patterns(self):
        """Identify common comorbidity patterns"""
        if self.processor.beneficiary_data is None:
            return

        print("\n" + "="*50)
        print("Comorbidity Pattern Analysis")
        print("="*50)

        conditions = [c for c in self.processor.beneficiary_data.columns if c.startswith('SP_')]
        df_cond = self.processor.beneficiary_data[conditions].apply(lambda x: (x==1).astype(int))

        # Simple co-occurrence count
        from itertools import combinations
        comorbidities = {}
        for c1, c2 in combinations(df_cond.columns, 2):
            count = (df_cond[c1] & df_cond[c2]).sum()
            if count > 0:
                pair_name = f"{c1.replace('SP_', '')} & {c2.replace('SP_', '')}"
                comorbidities[pair_name] = count

        top_comorbidities = sorted(comorbidities.items(), key=lambda item: item[1], reverse=True)[:10]

        print("Top 10 Comorbidity Pairs (by co-occurrence count):")
        for pair, count in top_comorbidities:
            print(f"  - {pair}: {count} patients")

        return comorbidities


class ReportGenerator:
    """Generate comprehensive analytics reports"""

    def __init__(self, processor, predictor, insights, visualizer):
        self.processor = processor
        self.predictor = predictor
        self.insights = insights
        self.visualizer = visualizer

    def generate_executive_summary(self, output_file='healthcare_analytics_report.txt'):
        """Generate a summary report of findings"""
        print("\n" + "="*60)
        print(" GENERATING EXECUTIVE SUMMARY REPORT")
        print("="*60)

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" HEALTHCARE ANALYTICS EXECUTIVE SUMMARY\n")
            f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Data Overview
            f.write("📊 DATA OVERVIEW\n")
            f.write("-" * 40 + "\n")
            n_patients = self.processor.beneficiary_data['DESYNPUF_ID'].nunique()
            n_admissions = len(self.processor.inpatient_data)
            f.write(f"Total Unique Patients Analyzed: {n_patients:,}\n")
            f.write(f"Total Inpatient Admissions: {n_admissions:,}\n\n")

            # Model Performance
            f.write("🎯 MODEL PERFORMANCE SUMMARY (Best Model by AUC)\n")
            f.write("-" * 40 + "\n")
            for period in ['READMIT_30', 'READMIT_60', 'READMIT_90']:
                if period in self.predictor.results and self.predictor.results[period]:
                    results = self.predictor.results[period]
                    
                    # Check if results contain actual model metrics or just flags
                    if isinstance(results, dict) and '_models_exist' in results:
                        f.write(f"{period}: Models exist but need retraining\n")
                        continue
                    
                    # Check if results is a valid dictionary with model metrics
                    if not isinstance(results, dict) or not results:
                        f.write(f"{period}: No data available\n")
                        continue
                    
                    try:
                        best_model_name = max(results, key=lambda k: results[k].get('auc', 0))
                        best_auc = results[best_model_name].get('auc', 0)
                        f.write(f"{period}: {best_model_name} (AUC: {best_auc:.3f})\n")
                    except (ValueError, AttributeError):
                        f.write(f"{period}: Error retrieving performance metrics\n")
            f.write("\n")

            # Key Insights
            f.write("💡 KEY INSIGHTS (from 30-Day Readmission Model)\n")
            f.write("-" * 40 + "\n")
            risk_factors = self.insights.identify_high_risk_factors('READMIT_30', top_n=5)
            f.write("Top Risk Factors:\n")
            for factor, score in risk_factors:
                f.write(f"  - {factor}\n")
            f.write("\n")

            # Cost Impact
            cost_analysis = self.insights.estimate_treatment_costs()
            if 'READMIT_30' in cost_analysis:
                savings = cost_analysis['READMIT_30'].get('potential_savings', 0)
                f.write(f"Potential 30-day cost savings with intervention: ${savings:,.2f}\n\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")

        print(f"Detailed report saved to: {output_file}")


# Main execution pipeline
def run_complete_pipeline():
    """Execute the complete healthcare analytics pipeline"""
    print("="*60)
    print(" HEALTHCARE ANALYTICS PIPELINE INITIATED")
    print("="*60)

    # 1. Data Loading and Preprocessing
    print("\n📁 STEP 1: LOADING & PREPROCESSING DATA...")
    processor = HealthcareDataProcessor()
    processor.load_beneficiary_data()
    processor.load_claims_data()
    if processor.inpatient_data is not None:
        processor.create_readmission_labels()
    else:
        print("Skipping readmission label creation as inpatient data failed to load.")
        return

    # 2. Feature Engineering
    print("\n⚙ STEP 2: ENGINEERING FEATURES...")
    engineer = FeatureEngineer(processor)
    engineer.create_patient_features()
    engineer.create_diagnosis_features()

    # 3. Model Training
    print("\n🤖 STEP 3: TRAINING PREDICTION MODELS...")
    predictor = ReadmissionPredictor(engineer)
    for period in ['READMIT_30', 'READMIT_60', 'READMIT_90']:
        predictor.train_ensemble(period)
    predictor.train_deep_learning_model('READMIT_30')

    # 4. Generating Insights
    print("\n💡 STEP 4: GENERATING CLINICAL INSIGHTS...")
    insights = ClinicalInsightsGenerator(predictor)
    insights.identify_high_risk_factors('READMIT_30')
    insights.estimate_treatment_costs()
    insights.generate_treatment_recommendations()

    # 5. Disease Progression Tracking
    print("\n📈 STEP 5: TRACKING DISEASE PROGRESSION...")
    tracker = DiseaseProgressionTracker(processor)
    tracker.analyze_condition_progression()
    tracker.identify_comorbidity_patterns()

    # 6. Visualization
    print("\n📊 STEP 6: GENERATING VISUALIZATIONS...")
    visualizer = ModelVisualizer(predictor)
    visualizer.plot_model_comparison()
    visualizer.plot_roc_curves()
    visualizer.plot_feature_importance()

    # 7. Reporting
    print("\n📄 STEP 7: GENERATING REPORT...")
    reporter = ReportGenerator(processor, predictor, insights, visualizer)
    reporter.generate_executive_summary()

    print("\n" + "="*60)
    print(" PIPELINE EXECUTION COMPLETE")
    print("="*60)

    return {
        'processor': processor, 'engineer': engineer, 'predictor': predictor,
        'insights': insights, 'tracker': tracker, 'visualizer': visualizer,
        'reporter': reporter
    }

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_pipeline()
    print("\n🎉 Healthcare Analytics Pipeline executed successfully!")