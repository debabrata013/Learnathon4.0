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
from fpdf import FPDF
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
    
    # Missing values heatmap
    plt.figure(figsize=(12, 8))
    missing_data = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if not missing_data.empty:
        plt.barh(missing_data['Column'], missing_data['Missing_Percentage'])
        plt.xlabel('Missing Percentage (%)')
        plt.title('Missing Values by Column')
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
    if selected_features and 'f_scores' in locals():
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
    
    # 11. Generate PDF Reports
    logger.info("Generating PDF reports...")
    
    # PDF 1: Preprocessing Analysis
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'Auto Insurance Fraud Detection - Preprocessing Analysis', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf1 = PDF()
    pdf1.add_page()
    
    # Executive Summary
    pdf1.set_font('Arial', 'B', 14)
    pdf1.cell(0, 10, 'Executive Summary', 0, 1, 'L')
    pdf1.ln(5)
    
    pdf1.set_font('Arial', '', 11)
    pdf1.multi_cell(0, 8, 
        f"This report presents comprehensive preprocessing analysis for auto insurance fraud detection. "
        f"Analysis conducted on {initial_shape[0]} training records with {initial_shape[1]} features.")
    pdf1.ln(10)
    
    # Data Quality
    pdf1.set_font('Arial', 'B', 12)
    pdf1.cell(0, 10, '1. Data Quality Analysis', 0, 1, 'L')
    pdf1.ln(5)
    
    pdf1.set_font('Arial', '', 10)
    pdf1.multi_cell(0, 6, f"Initial data shape: {initial_shape}")
    pdf1.multi_cell(0, 6, f"Final data shape: {combined_train.shape}")
    pdf1.multi_cell(0, 6, f"Duplicates removed: {removed_duplicates}")
    pdf1.multi_cell(0, 6, f"Columns with missing values: {len(missing_counts[missing_counts > 0])}")
    
    if high_missing_cols:
        pdf1.multi_cell(0, 6, f"High missing columns dropped: {', '.join(high_missing_cols)}")
    
    pdf1.ln(10)
    
    # Missing Value Handling
    pdf1.set_font('Arial', 'B', 12)
    pdf1.cell(0, 10, '2. Missing Value Handling', 0, 1, 'L')
    pdf1.ln(5)
    
    pdf1.set_font('Arial', '', 10)
    pdf1.multi_cell(0, 6, "Strategy Applied:")
    pdf1.multi_cell(0, 6, "• Dropped columns with >50% missing values")
    pdf1.multi_cell(0, 6, "• Mode imputation for categorical variables")
    pdf1.multi_cell(0, 6, "• KNN imputation for numerical variables")
    pdf1.multi_cell(0, 6, f"• Final missing values: {combined_train.isnull().sum().sum()}")
    
    pdf1.ln(10)
    
    # Outlier Analysis
    pdf1.set_font('Arial', 'B', 12)
    pdf1.cell(0, 10, '3. Outlier Analysis', 0, 1, 'L')
    pdf1.ln(5)
    
    pdf1.set_font('Arial', '', 10)
    pdf1.multi_cell(0, 6, f"Outlier detection performed on {len(outlier_info)} numerical columns")
    pdf1.multi_cell(0, 6, "Method: IQR-based detection with 1st-99th percentile capping")
    
    # Show top columns with outliers
    top_outliers = sorted(outlier_info.items(), key=lambda x: x[1], reverse=True)[:5]
    pdf1.multi_cell(0, 6, "Top columns with outliers:")
    for col, count in top_outliers:
        pdf1.multi_cell(0, 6, f"• {col}: {count} outliers")
    
    pdf1.ln(10)
    
    # Feature Selection
    pdf1.set_font('Arial', 'B', 12)
    pdf1.cell(0, 10, '4. Feature Selection', 0, 1, 'L')
    pdf1.ln(5)
    
    pdf1.set_font('Arial', '', 10)
    pdf1.multi_cell(0, 6, f"Selected {len(selected_features)} most important features using F-statistic")
    pdf1.multi_cell(0, 6, "Selected features:")
    for i, feature in enumerate(selected_features[:10], 1):
        pdf1.multi_cell(0, 6, f"{i}. {feature}")
    
    # Save PDF 1
    pdf1_path = output_dir / "preprocessing_analysis_report.pdf"
    pdf1.output(str(pdf1_path))
    
    # PDF 2: Feature Engineering
    pdf2 = PDF()
    pdf2.add_page()
    
    pdf2.set_font('Arial', 'B', 14)
    pdf2.cell(0, 10, 'Feature Engineering Analysis', 0, 1, 'L')
    pdf2.ln(10)
    
    pdf2.set_font('Arial', 'B', 12)
    pdf2.cell(0, 10, '1. Selected Features Analysis', 0, 1, 'L')
    pdf2.ln(5)
    
    pdf2.set_font('Arial', '', 10)
    pdf2.multi_cell(0, 6, f"Total selected features: {len(selected_features)}")
    pdf2.multi_cell(0, 6, "Selection method: F-statistic for classification")
    pdf2.ln(5)
    
    pdf2.multi_cell(0, 6, "Selected features and their business relevance:")
    for i, feature in enumerate(selected_features, 1):
        pdf2.multi_cell(0, 6, f"{i}. {feature}")
        # Add business interpretation
        if 'Claim' in feature:
            pdf2.multi_cell(0, 6, "   Business relevance: Claim amounts indicate potential fraud patterns")
        elif 'Age' in feature:
            pdf2.multi_cell(0, 6, "   Business relevance: Age demographics show different risk profiles")
        elif 'Premium' in feature:
            pdf2.multi_cell(0, 6, "   Business relevance: Premium reflects risk assessment")
        pdf2.ln(2)
    
    pdf2.add_page()
    
    pdf2.set_font('Arial', 'B', 12)
    pdf2.cell(0, 10, '2. Engineered Features', 0, 1, 'L')
    pdf2.ln(5)
    
    pdf2.set_font('Arial', '', 10)
    pdf2.multi_cell(0, 6, f"Created {len(engineered_features)} new features:")
    
    feature_descriptions = {
        'Claim_Premium_Ratio': 'Ratio of claim amount to premium - identifies inflated claims',
        'Age_Risk_Score': 'Risk score based on age demographics - captures age-related patterns',
        'Vehicle_Claim_Ratio': 'Ratio of claim to vehicle value - detects value fraud',
        'Claim_Complexity_Score': 'Number of claim components - measures claim complexity',
        'Feature_Interaction': 'Mathematical interaction between top features'
    }
    
    for i, feature in enumerate(engineered_features, 1):
        pdf2.multi_cell(0, 6, f"{i}. {feature}")
        if feature in feature_descriptions:
            pdf2.multi_cell(0, 6, f"   Purpose: {feature_descriptions[feature]}")
        pdf2.ln(3)
    
    pdf2.ln(10)
    
    pdf2.set_font('Arial', 'B', 12)
    pdf2.cell(0, 10, '3. Recommendations', 0, 1, 'L')
    pdf2.ln(5)
    
    pdf2.set_font('Arial', '', 10)
    recommendations = [
        "Use selected features for initial model training",
        "Apply ensemble methods (Random Forest, XGBoost) for feature interactions",
        "Consider class balancing due to imbalanced fraud distribution",
        "Use cross-validation with stratification",
        "Monitor engineered features for overfitting",
        "Validate feature importance with tree-based models"
    ]
    
    for rec in recommendations:
        pdf2.multi_cell(0, 6, f"• {rec}")
    
    # Save PDF 2
    pdf2_path = output_dir / "feature_engineering_report.pdf"
    pdf2.output(str(pdf2_path))
    
    # 12. Save Processed Data
    processed_data_path = output_dir / "processed_training_data.csv"
    combined_train.to_csv(processed_data_path, index=False)
    
    # Final Summary
    logger.info("Preprocessing completed successfully!")
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📁 Report Directory: {output_dir}")
    print(f"📄 Preprocessing Report (PDF 1): {pdf1_path}")
    print(f"📄 Feature Engineering Report (PDF 2): {pdf2_path}")
    print(f"💾 Processed Data: {processed_data_path}")
    print(f"\n📊 Data Summary:")
    print(f"   • Original shape: {initial_shape}")
    print(f"   • Final shape: {combined_train.shape}")
    print(f"   • Selected features: {len(selected_features)}")
    print(f"   • Engineered features: {len(engineered_features)}")
    print(f"   • Duplicates removed: {removed_duplicates}")
    print(f"   • Missing values handled: ✓")
    print(f"   • Outliers capped: ✓")
    print(f"   • Features normalized: ✓")
    print("="*80)
    
    return {
        'output_dir': str(output_dir),
        'pdf1_path': str(pdf1_path),
        'pdf2_path': str(pdf2_path),
        'processed_data_path': str(processed_data_path),
        'selected_features': selected_features,
        'engineered_features': engineered_features,
        'final_shape': combined_train.shape
    }

if __name__ == "__main__":
    results = run_fraud_preprocessing()
