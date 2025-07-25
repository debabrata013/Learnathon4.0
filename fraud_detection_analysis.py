#!/usr/bin/env python3
"""
Auto Insurance Fraud Detection Model
====================================

This script implements a comprehensive fraud detection system following the steps outlined in README.md:
1. Data Understanding & Exploration
2. Data Preprocessing & Feature Engineering
3. Model Building & Evaluation
4. Final Predictions & Submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FraudDetectionPipeline:
    def __init__(self, data_path='dataset/'):
        self.data_path = data_path
        self.train_data = None
        self.test_data = None
        self.submission_template = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def load_data(self):
        """Load and combine training datasets"""
        print("📊 Loading datasets...")
        
        # Load training datasets
        train1 = pd.read_csv(f'{self.data_path}Auto Insurance Fraud Claims (1).csv')
        train2 = pd.read_csv(f'{self.data_path}Auto Insurance Fraud Claims 02.csv')
        
        # Combine training data
        self.train_data = pd.concat([train1, train2], ignore_index=True)
        
        # Load test data (unseen)
        self.test_data = pd.read_csv(f'{self.data_path}Auto Insurance Fraud Claims (2).csv')
        
        # Load submission template
        self.submission_template = pd.read_csv(f'{self.data_path}Auto Insurance Fraud Claims Results.csv')
        
        print(f"✅ Training data shape: {self.train_data.shape}")
        print(f"✅ Test data shape: {self.test_data.shape}")
        print(f"✅ Submission template shape: {self.submission_template.shape}")
        
        return self
    
    def explore_data(self):
        """Comprehensive Exploratory Data Analysis"""
        print("\n🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print("\n📋 Dataset Info:")
        print(f"Training samples: {len(self.train_data):,}")
        print(f"Features: {len(self.train_data.columns)}")
        print(f"Memory usage: {self.train_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Target variable analysis
        fraud_counts = self.train_data['Fraud_Ind'].value_counts()
        fraud_pct = self.train_data['Fraud_Ind'].value_counts(normalize=True) * 100
        
        print(f"\n🎯 Target Variable Distribution:")
        print(f"Non-Fraud (N): {fraud_counts['N']:,} ({fraud_pct['N']:.1f}%)")
        print(f"Fraud (Y): {fraud_counts['Y']:,} ({fraud_pct['Y']:.1f}%)")
        print(f"Class Imbalance Ratio: {fraud_counts['N'] / fraud_counts['Y']:.1f}:1")
        
        # Missing values analysis
        print(f"\n❓ Missing Values Analysis:")
        missing_data = self.train_data.isnull().sum()
        missing_pct = (missing_data / len(self.train_data)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        
        print(missing_df[missing_df['Missing_Count'] > 0].head(10))
        
        # Data types
        print(f"\n📊 Data Types:")
        dtype_counts = self.train_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} columns")
        
        return self
    
    def visualize_data(self):
        """Create comprehensive visualizations"""
        print("\n📈 Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Auto Insurance Fraud Detection - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        fraud_counts = self.train_data['Fraud_Ind'].value_counts()
        axes[0,0].pie(fraud_counts.values, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', 
                     colors=['lightblue', 'lightcoral'])
        axes[0,0].set_title('Fraud Distribution')
        
        # 2. Age distribution by fraud
        sns.boxplot(data=self.train_data, x='Fraud_Ind', y='Age_Insured', ax=axes[0,1])
        axes[0,1].set_title('Age Distribution by Fraud Status')
        
        # 3. Policy premium by fraud
        sns.boxplot(data=self.train_data, x='Fraud_Ind', y='Policy_Premium', ax=axes[0,2])
        axes[0,2].set_title('Policy Premium by Fraud Status')
        
        # 4. Total claim amount by fraud
        sns.boxplot(data=self.train_data, x='Fraud_Ind', y='Total_Claim', ax=axes[1,0])
        axes[1,0].set_title('Total Claim Amount by Fraud Status')
        
        # 5. Vehicle cost by fraud
        sns.boxplot(data=self.train_data, x='Fraud_Ind', y='Vehicle_Cost', ax=axes[1,1])
        axes[1,1].set_title('Vehicle Cost by Fraud Status')
        
        # 6. Correlation heatmap (top numerical features)
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns[:10]
        corr_matrix = self.train_data[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("\n🔧 PREPROCESSING DATA")
        print("=" * 50)
        
        # Make copies to avoid modifying original data
        train_processed = self.train_data.copy()
        test_processed = self.test_data.copy()
        
        # 1. Handle missing values
        print("🔧 Handling missing values...")
        
        # For numerical columns: fill with median
        numerical_cols = train_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'Fraud_Ind':  # Don't fill target variable
                median_val = train_processed[col].median()
                train_processed[col].fillna(median_val, inplace=True)
                if col in test_processed.columns:
                    test_processed[col].fillna(median_val, inplace=True)
        
        # For categorical columns: fill with mode
        categorical_cols = train_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Fraud_Ind':  # Don't fill target variable
                mode_val = train_processed[col].mode()[0] if len(train_processed[col].mode()) > 0 else 'Unknown'
                train_processed[col].fillna(mode_val, inplace=True)
                if col in test_processed.columns:
                    test_processed[col].fillna(mode_val, inplace=True)
        
        # 2. Feature Engineering
        print("🔧 Engineering new features...")
        
        # Convert date columns
        date_cols = ['Bind_Date1', 'Policy_Start_Date', 'Policy_Expiry_Date', 'Accident_Date', 'Claims_Date', 'DL_Expiry_Date']
        
        for col in date_cols:
            if col in train_processed.columns:
                train_processed[col] = pd.to_datetime(train_processed[col], errors='coerce')
                if col in test_processed.columns:
                    test_processed[col] = pd.to_datetime(test_processed[col], errors='coerce')
        
        # Create new features
        if 'Policy_Start_Date' in train_processed.columns and 'Accident_Date' in train_processed.columns:
            train_processed['Days_Policy_to_Accident'] = (train_processed['Accident_Date'] - train_processed['Policy_Start_Date']).dt.days
            test_processed['Days_Policy_to_Accident'] = (test_processed['Accident_Date'] - test_processed['Policy_Start_Date']).dt.days
        
        if 'Accident_Date' in train_processed.columns and 'Claims_Date' in train_processed.columns:
            train_processed['Days_Accident_to_Claim'] = (train_processed['Claims_Date'] - train_processed['Accident_Date']).dt.days
            test_processed['Days_Accident_to_Claim'] = (test_processed['Claims_Date'] - test_processed['Accident_Date']).dt.days
        
        # Claim ratios
        if 'Total_Claim' in train_processed.columns and 'Policy_Premium' in train_processed.columns:
            train_processed['Claim_to_Premium_Ratio'] = train_processed['Total_Claim'] / (train_processed['Policy_Premium'] + 1)
            test_processed['Claim_to_Premium_Ratio'] = test_processed['Total_Claim'] / (test_processed['Policy_Premium'] + 1)
        
        if 'Vehicle_Claim' in train_processed.columns and 'Vehicle_Cost' in train_processed.columns:
            train_processed['Vehicle_Claim_Ratio'] = train_processed['Vehicle_Claim'] / (train_processed['Vehicle_Cost'] + 1)
            test_processed['Vehicle_Claim_Ratio'] = test_processed['Vehicle_Claim'] / (test_processed['Vehicle_Cost'] + 1)
        
        # 3. Encode categorical variables
        print("🔧 Encoding categorical variables...")
        
        # Get categorical columns (excluding target and ID columns)
        categorical_cols = train_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['Fraud_Ind', 'Claim_ID']]
        
        # Label encode categorical variables
        for col in categorical_cols:
            if col in train_processed.columns:
                le = LabelEncoder()
                # Fit on combined data to ensure consistent encoding
                combined_values = pd.concat([train_processed[col].astype(str), test_processed[col].astype(str)])
                le.fit(combined_values)
                
                train_processed[col] = le.transform(train_processed[col].astype(str))
                if col in test_processed.columns:
                    test_processed[col] = le.transform(test_processed[col].astype(str))
                
                self.encoders[col] = le
        
        # 4. Handle remaining non-numeric columns
        for col in train_processed.columns:
            if train_processed[col].dtype == 'object' and col not in ['Fraud_Ind', 'Claim_ID']:
                train_processed[col] = pd.to_numeric(train_processed[col], errors='coerce')
                if col in test_processed.columns:
                    test_processed[col] = pd.to_numeric(test_processed[col], errors='coerce')
        
        # 5. Prepare final datasets
        # Remove ID columns and target from features
        feature_cols = [col for col in train_processed.columns if col not in ['Fraud_Ind', 'Claim_ID']]
        
        X_train = train_processed[feature_cols]
        y_train = train_processed['Fraud_Ind'].map({'N': 0, 'Y': 1})
        X_test = test_processed[feature_cols]
        
        # Handle any remaining NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # 6. Scale numerical features
        print("🔧 Scaling numerical features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        self.scalers['standard'] = scaler
        
        print(f"✅ Preprocessed training shape: {X_train_scaled.shape}")
        print(f"✅ Preprocessed test shape: {X_test_scaled.shape}")
        print(f"✅ Target distribution: {y_train.value_counts().to_dict()}")
        
        return X_train_scaled, y_train, X_test_scaled
    
    def build_models(self, X_train, y_train):
        """Build and train multiple models"""
        print("\n🤖 BUILDING MODELS")
        print("=" * 50)
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Split data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 1. Logistic Regression
        print("🔧 Training Logistic Regression...")
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(X_train_split, y_train_split)
        self.models['logistic_regression'] = lr_model
        
        # 2. Random Forest
        print("🔧 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_split, y_train_split)
        self.models['random_forest'] = rf_model
        
        # 3. XGBoost
        print("🔧 Training XGBoost...")
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_split, y_train_split)
        self.models['xgboost'] = xgb_model
        
        # Evaluate models on validation set
        print("\n📊 Model Performance on Validation Set:")
        for name, model in self.models.items():
            y_pred = model.predict(X_val_split)
            y_pred_proba = model.predict_proba(X_val_split)[:, 1]
            
            auc_score = roc_auc_score(y_val_split, y_pred_proba)
            print(f"\n{name.upper()}:")
            print(f"  AUC Score: {auc_score:.4f}")
            print(f"  Classification Report:")
            print(classification_report(y_val_split, y_pred, target_names=['Non-Fraud', 'Fraud']))
        
        return self
    
    def evaluate_models(self, X_train, y_train):
        """Comprehensive model evaluation with cross-validation"""
        print("\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 50)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        for name, model in self.models.items():
            print(f"\n🔍 Evaluating {name.upper()}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"  Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Plot model comparison
        self.plot_model_comparison(results)
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name.upper()}")
        print(f"   AUC Score: {results[best_model_name]['cv_mean']:.4f}")
        
        return best_model, best_model_name
    
    def plot_model_comparison(self, results):
        """Plot model comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        models = list(results.keys())
        means = [results[model]['cv_mean'] for model in models]
        stds = [results[model]['cv_std'] for model in models]
        
        ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_title('Model Comparison - Cross-Validation AUC')
        ax1.set_ylabel('AUC Score')
        ax1.set_ylim(0.5, 1.0)
        
        # Box plot of CV scores
        cv_data = [results[model]['cv_scores'] for model in models]
        ax2.boxplot(cv_data, labels=models)
        ax2.set_title('Cross-Validation Score Distribution')
        ax2.set_ylabel('AUC Score')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_predictions(self, best_model, X_test):
        """Generate final predictions"""
        print("\n🎯 GENERATING FINAL PREDICTIONS")
        print("=" * 50)
        
        # Generate predictions
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'Claim_ID': self.test_data['Claim_ID'],
            'Fraud_Ind': ['Y' if pred == 1 else 'N' for pred in y_pred]
        })
        
        # Save predictions
        submission.to_csv('fraud_predictions.csv', index=False)
        
        print(f"✅ Predictions saved to 'fraud_predictions.csv'")
        print(f"📊 Prediction distribution:")
        print(submission['Fraud_Ind'].value_counts())
        print(f"📊 Fraud rate in predictions: {(submission['Fraud_Ind'] == 'Y').mean():.2%}")
        
        return submission
    
    def run_complete_pipeline(self):
        """Run the complete fraud detection pipeline"""
        print("🚀 STARTING FRAUD DETECTION PIPELINE")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Visualize data
        self.visualize_data()
        
        # Preprocess data
        X_train, y_train, X_test = self.preprocess_data()
        
        # Build models
        self.build_models(X_train, y_train)
        
        # Evaluate models
        best_model, best_model_name = self.evaluate_models(X_train, y_train)
        
        # Generate predictions
        submission = self.generate_predictions(best_model, X_test)
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✅ Best model: {best_model_name}")
        print(f"✅ Predictions saved to: fraud_predictions.csv")
        print(f"✅ Visualizations saved to: fraud_detection_eda.png, model_comparison.png")
        
        return submission

def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = FraudDetectionPipeline()
    
    # Run complete pipeline
    submission = pipeline.run_complete_pipeline()
    
    return submission

if __name__ == "__main__":
    submission = main()
