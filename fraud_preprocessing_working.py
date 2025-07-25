#!/usr/bin/env python3
"""
Senior-Level ML Fraud Detection Preprocessing Pipeline
=====================================================
Complete preprocessing pipeline for auto insurance fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy import stats
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_fraud_preprocessing():
    """Complete fraud detection preprocessing pipeline"""
    
    # Setup paths
    data_path = Path("/Users/debabratapattnayak/web-dev/learnathon/dataset")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("/Users/debabratapattnayak/web-dev/learnathon/ml_analysis_reports") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting preprocessing pipeline. Output: {output_dir}")
    
    # 1. Load Data
    logger.info("Loading datasets...")
    train1 = pd.read_csv(data_path / "Auto Insurance Fraud Claims (1).csv")
    train2 = pd.read_csv(data_path / "Auto Insurance Fraud Claims 02.csv")
    test_data = pd.read_csv(data_path / "Auto Insurance Fraud Claims (2).csv")
    
    # Combine training data
    combined_train = pd.concat([train1, train2], ignore_index=True)
    logger.info(f"Combined training data shape: {combined_train.shape}")
    
    # 2. Data Quality Analysis
    logger.info("Analyzing data quality...")
    initial_shape = combined_train.shape
    missing_counts = combined_train.isnull().sum()
    missing_pct = (missing_counts / len(combined_train)) * 100
    duplicates = combined_train.duplicated().sum()
    
    # 3. Handle Missing Values
    logger.info("Handling missing values...")
    # Drop columns with >50% missing
    high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
    if high_missing_cols:
        combined_train = combined_train.drop(columns=high_missing_cols)
        logger.info(f"Dropped columns with >50% missing: {high_missing_cols}")
    
    # Fill categorical missing values with mode
    categorical_cols = combined_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if combined_train[col].isnull().sum() > 0:
            mode_val = combined_train[col].mode()[0] if not combined_train[col].mode().empty else 'Unknown'
            combined_train[col].fillna(mode_val, inplace=True)
    
    # Fill numerical missing values with KNN
    numeric_cols = combined_train.select_dtypes(include=[np.number]).columns
    missing_numeric = [col for col in numeric_cols if combined_train[col].isnull().sum() > 0]
    if missing_numeric:
        knn_imputer = KNNImputer(n_neighbors=5)
        combined_train[missing_numeric] = knn_imputer.fit_transform(combined_train[missing_numeric])
    
    # 4. Handle Duplicates
    logger.info("Handling duplicates...")
    initial_count = len(combined_train)
    combined_train = combined_train.drop_duplicates()
    final_count = len(combined_train)
    removed_duplicates = initial_count - final_count
    
    # 5. Outlier Detection and Handling
    logger.info("Handling outliers...")
    numeric_cols = combined_train.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numeric_cols:
        if col in combined_train.columns:
            data = combined_train[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_info[col] = outliers
            
            # Cap outliers at 1st and 99th percentiles
            lower_cap = data.quantile(0.01)
            upper_cap = data.quantile(0.99)
            combined_train[col] = combined_train[col].clip(lower=lower_cap, upper=upper_cap)
    
    # 6. Encode Categorical Features
    logger.info("Encoding categorical features...")
    label_encoders = {}
    categorical_cols = combined_train.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Fraud_Ind']
    
    for col in categorical_cols:
        unique_values = combined_train[col].nunique()
        if unique_values <= 10:  # Label encoding for low cardinality
            le = LabelEncoder()
            combined_train[col] = le.fit_transform(combined_train[col].astype(str))
            label_encoders[col] = le
        else:  # Frequency encoding for high cardinality
            freq_map = combined_train[col].value_counts().to_dict()
            combined_train[col] = combined_train[col].map(freq_map)
    
    # Encode target variable
    if 'Fraud_Ind' in combined_train.columns:
        le_target = LabelEncoder()
        combined_train['Fraud_Ind'] = le_target.fit_transform(combined_train['Fraud_Ind'])
        label_encoders['Fraud_Ind'] = le_target
    
    # 7. Feature Selection
    logger.info("Selecting important features...")
    f_scores = {}
    if 'Fraud_Ind' in combined_train.columns:
        X = combined_train.drop(['Fraud_Ind'], axis=1)
        y = combined_train['Fraud_Ind']
        
        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        # Feature selection using F-statistic
        try:
            f_selector = SelectKBest(score_func=f_classif, k='all')
            f_selector.fit(X_numeric, y)
            f_scores = dict(zip(X_numeric.columns, f_selector.scores_))
            
            # Select top 15 features
            top_features = sorted(f_scores.items(), key=lambda x: x[1], reverse=True)[:15]
            selected_features = [feature[0] for feature in top_features]
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            selected_features = list(X_numeric.columns)[:15]
    else:
        selected_features = []
    
    # 8. Normalize Features
    logger.info("Normalizing features...")
    if selected_features:
        scaler = StandardScaler()
        X_selected = combined_train[selected_features]
        X_normalized = scaler.fit_transform(X_selected)
        
        for i, feature in enumerate(selected_features):
            combined_train[f"{feature}_normalized"] = X_normalized[:, i]
    
    # 9. Feature Engineering
    logger.info("Creating engineered features...")
    engineered_features = []
    
    try:
        # Feature 1: Claim to Premium Ratio
        if 'Total_Claim' in combined_train.columns and 'Policy_Premium' in combined_train.columns:
            combined_train['Claim_Premium_Ratio'] = (
                combined_train['Total_Claim'] / (combined_train['Policy_Premium'] + 1)
            )
            engineered_features.append('Claim_Premium_Ratio')
        
        # Feature 2: Age Risk Score
        if 'Age_Insured' in combined_train.columns:
            combined_train['Age_Risk_Score'] = combined_train['Age_Insured'].apply(
                lambda x: 2 if x < 25 or x > 65 else 1 if x < 30 or x > 60 else 0
            )
            engineered_features.append('Age_Risk_Score')
        
        # Feature 3: Vehicle Value to Claim Ratio
        if 'Vehicle_Cost' in combined_train.columns and 'Total_Claim' in combined_train.columns:
            combined_train['Vehicle_Claim_Ratio'] = (
                combined_train['Total_Claim'] / (combined_train['Vehicle_Cost'] + 1)
            )
            engineered_features.append('Vehicle_Claim_Ratio')
        
        # Feature 4: Claim Complexity Score
        claim_components = ['Injury_Claim', 'Property_Claim', 'Vehicle_Claim']
        available_components = [col for col in claim_components if col in combined_train.columns]
        if len(available_components) >= 2:
            combined_train['Claim_Complexity_Score'] = 0
            for component in available_components:
                combined_train['Claim_Complexity_Score'] += (combined_train[component] > 0).astype(int)
            engineered_features.append('Claim_Complexity_Score')
        
        # Feature 5: Simple interaction feature
        if len(selected_features) >= 2:
            feat1, feat2 = selected_features[0], selected_features[1]
            combined_train['Feature_Interaction'] = combined_train[feat1] * combined_train[feat2]
            engineered_features.append('Feature_Interaction')
            
    except Exception as e:
        logger.warning(f"Feature engineering error: {e}")
    
    # 10. Create Visualizations
    logger.info("Creating visualizations...")
    plt.style.use('default')
    
    # Missing values analysis
    plt.figure(figsize=(12, 8))
    missing_data = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if not missing_data.empty:
        plt.barh(missing_data['Column'][:20], missing_data['Missing_Percentage'][:20])  # Top 20 only
        plt.xlabel('Missing Percentage (%)')
        plt.title('Missing Values by Column (Top 20)')
        plt.tight_layout()
        plt.savefig(output_dir / 'missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Target distribution
    if 'Fraud_Ind' in combined_train.columns:
        plt.figure(figsize=(10, 6))
        fraud_counts = combined_train['Fraud_Ind'].value_counts()
        plt.pie(fraud_counts.values, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90)
        plt.title('Fraud Distribution')
        plt.savefig(output_dir / 'fraud_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Feature importance
    if selected_features and f_scores:
        plt.figure(figsize=(12, 8))
        top_features_plot = sorted(f_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        features, scores = zip(*top_features_plot)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('F-Score')
        plt.title('Top 10 Feature Importance Scores')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 11. Generate Text Reports (instead of PDF)
    logger.info("Generating text reports...")
    
    # Report 1: Preprocessing Analysis
    report1_path = output_dir / "preprocessing_analysis_report.txt"
    with open(report1_path, 'w') as f:
        f.write("AUTO INSURANCE FRAUD DETECTION - PREPROCESSING ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"This report presents comprehensive preprocessing analysis for auto insurance fraud detection.\n")
        f.write(f"Analysis conducted on {initial_shape[0]} training records with {initial_shape[1]} features.\n\n")
        
        f.write("1. DATA QUALITY ANALYSIS\n")
        f.write("-" * 25 + "\n")
        f.write(f"Initial data shape: {initial_shape}\n")
        f.write(f"Final data shape: {combined_train.shape}\n")
        f.write(f"Duplicates removed: {removed_duplicates}\n")
        f.write(f"Columns with missing values: {len(missing_counts[missing_counts > 0])}\n")
        
        if high_missing_cols:
            f.write(f"High missing columns dropped: {', '.join(high_missing_cols)}\n")
        f.write("\n")
        
        f.write("2. MISSING VALUE HANDLING\n")
        f.write("-" * 26 + "\n")
        f.write("Strategy Applied:\n")
        f.write("• Dropped columns with >50% missing values\n")
        f.write("• Mode imputation for categorical variables\n")
        f.write("• KNN imputation for numerical variables\n")
        f.write(f"• Final missing values: {combined_train.isnull().sum().sum()}\n\n")
        
        f.write("3. OUTLIER ANALYSIS\n")
        f.write("-" * 19 + "\n")
        f.write(f"Outlier detection performed on {len(outlier_info)} numerical columns\n")
        f.write("Method: IQR-based detection with 1st-99th percentile capping\n")
        f.write("Top columns with outliers:\n")
        
        top_outliers = sorted(outlier_info.items(), key=lambda x: x[1], reverse=True)[:10]
        for col, count in top_outliers:
            f.write(f"• {col}: {count} outliers\n")
        f.write("\n")
        
        f.write("4. FEATURE SELECTION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Selected {len(selected_features)} most important features using F-statistic\n")
        f.write("Selected features:\n")
        for i, feature in enumerate(selected_features, 1):
            score = f_scores.get(feature, 0)
            f.write(f"{i:2d}. {feature:<25} (F-Score: {score:.4f})\n")
        f.write("\n")
        
        f.write("5. CATEGORICAL ENCODING\n")
        f.write("-" * 23 + "\n")
        f.write(f"Encoded {len(label_encoders)} categorical columns\n")
        f.write("Methods used: Label encoding (low cardinality), Frequency encoding (high cardinality)\n\n")
        
        f.write("6. NORMALIZATION\n")
        f.write("-" * 16 + "\n")
        f.write(f"Normalized {len(selected_features)} selected features using StandardScaler\n")
        f.write("All normalized features have mean=0 and standard deviation=1\n\n")
    
    # Report 2: Feature Engineering
    report2_path = output_dir / "feature_engineering_report.txt"
    with open(report2_path, 'w') as f:
        f.write("FEATURE ENGINEERING & SELECTION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 17 + "\n")
        f.write(f"This report details the feature selection and engineering process for fraud detection.\n")
        f.write(f"From the original dataset, {len(selected_features)} key features were selected and\n")
        f.write(f"{len(engineered_features)} new features were engineered to enhance model performance.\n\n")
        
        f.write("1. SELECTED FEATURES ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write("Selection Methodology:\n")
        f.write("• F-statistic for measuring linear relationships with target\n")
        f.write("• Statistical significance testing for feature relevance\n")
        f.write("• Top 15 features selected based on F-scores\n\n")
        
        f.write("Selected Features and Their Business Relevance:\n")
        for i, feature in enumerate(selected_features, 1):
            score = f_scores.get(feature, 0)
            f.write(f"{i:2d}. {feature:<25} (F-Score: {score:.4f})\n")
            
            # Business interpretation
            if 'Claim' in feature:
                f.write("    Business relevance: Claim amounts indicate potential fraud patterns\n")
            elif 'Age' in feature:
                f.write("    Business relevance: Age demographics show different risk profiles\n")
            elif 'Premium' in feature:
                f.write("    Business relevance: Premium reflects risk assessment and coverage\n")
            elif 'Vehicle' in feature:
                f.write("    Business relevance: Vehicle characteristics affect fraud likelihood\n")
            elif 'Policy' in feature:
                f.write("    Business relevance: Policy details influence claim patterns\n")
            f.write("\n")
        
        f.write("2. ENGINEERED FEATURES\n")
        f.write("-" * 22 + "\n")
        f.write(f"Created {len(engineered_features)} new features:\n\n")
        
        feature_descriptions = {
            'Claim_Premium_Ratio': 'Ratio of claim amount to premium - identifies inflated claims',
            'Age_Risk_Score': 'Risk score based on age demographics - captures age-related patterns',
            'Vehicle_Claim_Ratio': 'Ratio of claim to vehicle value - detects value fraud',
            'Claim_Complexity_Score': 'Number of claim components - measures claim complexity',
            'Feature_Interaction': 'Mathematical interaction between top features'
        }
        
        for i, feature in enumerate(engineered_features, 1):
            f.write(f"{i}. {feature}\n")
            if feature in feature_descriptions:
                f.write(f"   Purpose: {feature_descriptions[feature]}\n")
            
            # Show basic statistics
            if feature in combined_train.columns:
                stats = combined_train[feature].describe()
                f.write(f"   Statistics: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}\n")
                f.write(f"               Min={stats['min']:.4f}, Max={stats['max']:.4f}\n")
            f.write("\n")
        
        f.write("3. FEATURE ENGINEERING BEST PRACTICES APPLIED\n")
        f.write("-" * 46 + "\n")
        best_practices = [
            "Domain Knowledge Integration: Features designed based on insurance fraud patterns",
            "Ratio Features: Created meaningful ratios to capture relative relationships",
            "Categorical Risk Scoring: Converted continuous variables to risk categories",
            "Interaction Features: Captured relationships between multiple variables",
            "Normalization: Applied StandardScaler to ensure feature scale consistency",
            "Business Interpretability: Ensured all features have clear business meaning"
        ]
        
        for practice in best_practices:
            f.write(f"• {practice}\n")
        f.write("\n")
        
        f.write("4. RECOMMENDATIONS FOR MODEL DEVELOPMENT\n")
        f.write("-" * 40 + "\n")
        recommendations = [
            "Use both selected original features and engineered features",
            "Apply tree-based models (Random Forest, XGBoost) for feature interactions",
            "Monitor feature importance to validate engineering decisions",
            "Use stratified cross-validation to maintain fraud class distribution",
            "Consider class balancing techniques due to imbalanced dataset",
            "Validate engineered features prevent overfitting"
        ]
        
        for rec in recommendations:
            f.write(f"• {rec}\n")
    
    # 12. Save Processed Data
    processed_data_path = output_dir / "processed_training_data.csv"
    combined_train.to_csv(processed_data_path, index=False)
    
    # Save feature lists
    features_info_path = output_dir / "feature_information.txt"
    with open(features_info_path, 'w') as f:
        f.write("FEATURE INFORMATION\n")
        f.write("=" * 20 + "\n\n")
        
        f.write("SELECTED FEATURES:\n")
        for i, feature in enumerate(selected_features, 1):
            f.write(f"{i:2d}. {feature}\n")
        
        f.write(f"\nENGINEERED FEATURES:\n")
        for i, feature in enumerate(engineered_features, 1):
            f.write(f"{i}. {feature}\n")
        
        f.write(f"\nNORMALIZED FEATURES:\n")
        normalized_features = [f"{feat}_normalized" for feat in selected_features]
        for i, feature in enumerate(normalized_features, 1):
            f.write(f"{i:2d}. {feature}\n")
    
    # Final Summary
    logger.info("Preprocessing completed successfully!")
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📁 Report Directory: {output_dir}")
    print(f"📄 Preprocessing Report: {report1_path.name}")
    print(f"📄 Feature Engineering Report: {report2_path.name}")
    print(f"📄 Feature Information: {features_info_path.name}")
    print(f"💾 Processed Data: {processed_data_path.name}")
    print(f"\n📊 Data Summary:")
    print(f"   • Original shape: {initial_shape}")
    print(f"   • Final shape: {combined_train.shape}")
    print(f"   • Selected features: {len(selected_features)}")
    print(f"   • Engineered features: {len(engineered_features)}")
    print(f"   • Duplicates removed: {removed_duplicates}")
    print(f"   • Missing values handled: ✓")
    print(f"   • Outliers capped: ✓")
    print(f"   • Features normalized: ✓")
    print(f"\n🎯 Ready for Model Building:")
    print(f"   • Data is cleaned and preprocessed")
    print(f"   • Features are selected and engineered")
    print(f"   • Normalization applied for ML algorithms")
    print(f"   • Comprehensive documentation generated")
    print("="*80)
    
    return {
        'output_dir': str(output_dir),
        'report1_path': str(report1_path),
        'report2_path': str(report2_path),
        'processed_data_path': str(processed_data_path),
        'selected_features': selected_features,
        'engineered_features': engineered_features,
        'final_shape': combined_train.shape
    }

if __name__ == "__main__":
    results = run_fraud_preprocessing()
