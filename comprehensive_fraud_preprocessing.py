#!/usr/bin/env python3
"""
Senior-Level ML Fraud Detection Preprocessing Pipeline
=====================================================
Author: Senior AI/ML Specialist
Date: 2025-01-25
Purpose: Comprehensive data preprocessing for auto insurance fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class FraudDetectionPreprocessor:
    """
    Senior-level ML preprocessing pipeline for fraud detection
    Implements industry best practices for data cleaning, feature engineering, and analysis
    """
    
    def __init__(self, data_path: str, output_dir: str = None):
        """
        Initialize the preprocessor with data path and output directory
        
        Args:
            data_path: Path to the dataset folder
            output_dir: Directory to save reports and processed data
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path("ml_analysis_reports")
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.report_dir = self.output_dir / self.timestamp
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.train_data = None
        self.test_data = None
        self.combined_train = None
        self.processed_data = None
        self.selected_features = None
        self.engineered_features = None
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
        # Analysis results storage
        self.preprocessing_summary = {}
        self.feature_analysis = {}
        self.outlier_analysis = {}
        
        logger.info(f"Initialized FraudDetectionPreprocessor with output directory: {self.report_dir}")
    
    def load_data(self) -> None:
        """Load and combine training datasets"""
        try:
            # Load training datasets
            train1_path = self.data_path / "Auto Insurance Fraud Claims (1).csv"
            train2_path = self.data_path / "Auto Insurance Fraud Claims 02.csv"
            test_path = self.data_path / "Auto Insurance Fraud Claims (2).csv"
            
            logger.info("Loading training datasets...")
            train1 = pd.read_csv(train1_path)
            train2 = pd.read_csv(train2_path)
            self.test_data = pd.read_csv(test_path)
            
            # Combine training datasets
            self.combined_train = pd.concat([train1, train2], ignore_index=True)
            
            logger.info(f"Training data shape: {self.combined_train.shape}")
            logger.info(f"Test data shape: {self.test_data.shape}")
            
            # Store initial data info
            self.preprocessing_summary['initial_train_shape'] = self.combined_train.shape
            self.preprocessing_summary['initial_test_shape'] = self.test_data.shape
            self.preprocessing_summary['total_features'] = len(self.combined_train.columns)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        logger.info("Performing comprehensive data quality analysis...")
        
        analysis = {
            'missing_values': {},
            'data_types': {},
            'duplicates': {},
            'basic_stats': {}
        }
        
        # Missing values analysis
        missing_counts = self.combined_train.isnull().sum()
        missing_percentages = (missing_counts / len(self.combined_train)) * 100
        
        analysis['missing_values'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
        
        # Data types analysis
        analysis['data_types'] = self.combined_train.dtypes.to_dict()
        
        # Duplicate analysis
        duplicates = self.combined_train.duplicated()
        analysis['duplicates'] = {
            'count': duplicates.sum(),
            'percentage': (duplicates.sum() / len(self.combined_train)) * 100
        }
        
        # Basic statistics
        numeric_cols = self.combined_train.select_dtypes(include=[np.number]).columns
        analysis['basic_stats'] = self.combined_train[numeric_cols].describe().to_dict()
        
        self.preprocessing_summary['data_quality'] = analysis
        logger.info(f"Data quality analysis completed. Found {len(analysis['missing_values']['columns_with_missing'])} columns with missing values")
        
        return analysis
    
    def handle_missing_values(self) -> None:
        """Advanced missing value handling with multiple strategies"""
        logger.info("Handling missing values with advanced strategies...")
        
        missing_info = self.preprocessing_summary['data_quality']['missing_values']
        
        # Strategy 1: Drop columns with >50% missing values
        high_missing_cols = [col for col, pct in missing_info['percentages'].items() if pct > 50]
        if high_missing_cols:
            logger.info(f"Dropping columns with >50% missing values: {high_missing_cols}")
            self.combined_train = self.combined_train.drop(columns=high_missing_cols)
        
        # Strategy 2: Handle categorical missing values
        categorical_cols = self.combined_train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.combined_train[col].isnull().sum() > 0:
                # Use mode for categorical variables
                mode_value = self.combined_train[col].mode()[0] if not self.combined_train[col].mode().empty else 'Unknown'
                self.combined_train[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        # Strategy 3: Handle numerical missing values with KNN imputation
        numeric_cols = self.combined_train.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if self.combined_train[col].isnull().sum() > 0]
        
        if numeric_cols:
            logger.info(f"Applying KNN imputation to numerical columns: {numeric_cols}")
            knn_imputer = KNNImputer(n_neighbors=5)
            self.combined_train[numeric_cols] = knn_imputer.fit_transform(self.combined_train[numeric_cols])
            self.imputers['knn_numeric'] = knn_imputer
        
        # Update preprocessing summary
        remaining_missing = self.combined_train.isnull().sum().sum()
        self.preprocessing_summary['missing_values_handled'] = {
            'high_missing_dropped': high_missing_cols,
            'remaining_missing_count': remaining_missing,
            'strategy_used': 'Mode for categorical, KNN for numerical'
        }
        
        logger.info(f"Missing value handling completed. Remaining missing values: {remaining_missing}")
    
    def identify_and_handle_duplicates(self) -> None:
        """Identify and handle duplicate records"""
        logger.info("Identifying and handling duplicate records...")
        
        initial_count = len(self.combined_train)
        
        # Remove exact duplicates
        self.combined_train = self.combined_train.drop_duplicates()
        
        # Check for potential duplicates based on key columns
        key_columns = ['Policy_Num', 'Claim_ID', 'Accident_Date']
        key_columns = [col for col in key_columns if col in self.combined_train.columns]
        
        if key_columns:
            potential_duplicates = self.combined_train.duplicated(subset=key_columns, keep=False)
            duplicate_count = potential_duplicates.sum()
            
            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} potential duplicates based on key columns: {key_columns}")
                # Keep first occurrence of duplicates
                self.combined_train = self.combined_train.drop_duplicates(subset=key_columns, keep='first')
        
        final_count = len(self.combined_train)
        removed_count = initial_count - final_count
        
        self.preprocessing_summary['duplicates_handled'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_count': removed_count,
            'removal_percentage': (removed_count / initial_count) * 100
        }
        
        logger.info(f"Duplicate handling completed. Removed {removed_count} duplicate records")
    
    def detect_outliers(self) -> Dict[str, Any]:
        """Advanced outlier detection using multiple methods"""
        logger.info("Performing advanced outlier detection...")
        
        numeric_cols = self.combined_train.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if col in self.combined_train.columns:
                data = self.combined_train[col].dropna()
                
                # Method 1: IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                
                # Method 2: Z-score method
                z_scores = np.abs(stats.zscore(data))
                zscore_outliers = (z_scores > 3).sum()
                
                # Method 3: Modified Z-score method
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                modified_zscore_outliers = (np.abs(modified_z_scores) > 3.5).sum()
                
                outlier_info[col] = {
                    'iqr_outliers': iqr_outliers,
                    'zscore_outliers': zscore_outliers,
                    'modified_zscore_outliers': modified_zscore_outliers,
                    'iqr_bounds': (lower_bound, upper_bound),
                    'total_values': len(data)
                }
        
        self.outlier_analysis = outlier_info
        logger.info(f"Outlier detection completed for {len(numeric_cols)} numerical columns")
        
        return outlier_info
    
    def handle_outliers(self, method: str = 'cap') -> None:
        """Handle outliers using specified method"""
        logger.info(f"Handling outliers using {method} method...")
        
        numeric_cols = self.combined_train.select_dtypes(include=[np.number]).columns
        outliers_handled = {}
        
        for col in numeric_cols:
            if col in self.outlier_analysis:
                data = self.combined_train[col]
                
                if method == 'cap':
                    # Cap outliers at 1st and 99th percentiles
                    lower_cap = data.quantile(0.01)
                    upper_cap = data.quantile(0.99)
                    
                    original_outliers = ((data < lower_cap) | (data > upper_cap)).sum()
                    
                    self.combined_train[col] = data.clip(lower=lower_cap, upper=upper_cap)
                    
                    outliers_handled[col] = {
                        'method': 'capping',
                        'outliers_handled': original_outliers,
                        'bounds': (lower_cap, upper_cap)
                    }
                
                elif method == 'remove':
                    # Remove outliers using IQR method
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (data >= lower_bound) & (data <= upper_bound)
                    outliers_removed = (~outlier_mask).sum()
                    
                    self.combined_train = self.combined_train[outlier_mask]
                    
                    outliers_handled[col] = {
                        'method': 'removal',
                        'outliers_removed': outliers_removed,
                        'bounds': (lower_bound, upper_bound)
                    }
        
        self.preprocessing_summary['outliers_handled'] = outliers_handled
        logger.info(f"Outlier handling completed using {method} method")
#!/usr/bin/env python3
"""
ML Fraud Detection Preprocessing Pipeline - Part 2
=================================================
Feature Engineering, Selection, and Normalization
"""

# Continuation of FraudDetectionPreprocessor class

    def encode_categorical_features(self) -> None:
        """Advanced categorical feature encoding"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = self.combined_train.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Fraud_Ind']  # Exclude target
        
        encoding_info = {}
        
        for col in categorical_cols:
            unique_values = self.combined_train[col].nunique()
            
            if unique_values <= 10:  # Use Label Encoding for low cardinality
                le = LabelEncoder()
                self.combined_train[col] = le.fit_transform(self.combined_train[col].astype(str))
                self.label_encoders[col] = le
                encoding_info[col] = {'method': 'label_encoding', 'unique_values': unique_values}
                
            else:  # Use frequency encoding for high cardinality
                freq_map = self.combined_train[col].value_counts().to_dict()
                self.combined_train[col] = self.combined_train[col].map(freq_map)
                encoding_info[col] = {'method': 'frequency_encoding', 'unique_values': unique_values}
        
        # Encode target variable
        if 'Fraud_Ind' in self.combined_train.columns:
            le_target = LabelEncoder()
            self.combined_train['Fraud_Ind'] = le_target.fit_transform(self.combined_train['Fraud_Ind'])
            self.label_encoders['Fraud_Ind'] = le_target
            encoding_info['Fraud_Ind'] = {'method': 'label_encoding', 'classes': list(le_target.classes_)}
        
        self.preprocessing_summary['categorical_encoding'] = encoding_info
        logger.info(f"Categorical encoding completed for {len(categorical_cols)} columns")
    
    def select_important_features(self, n_features: int = 15) -> List[str]:
        """Advanced feature selection using multiple methods"""
        logger.info(f"Selecting top {n_features} important features...")
        
        # Prepare features and target
        if 'Fraud_Ind' not in self.combined_train.columns:
            logger.error("Target variable 'Fraud_Ind' not found")
            return []
        
        X = self.combined_train.drop(['Fraud_Ind'], axis=1)
        y = self.combined_train['Fraud_Ind']
        
        # Remove non-numeric columns that couldn't be encoded
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        feature_scores = {}
        
        # Method 1: F-statistic
        try:
            f_selector = SelectKBest(score_func=f_classif, k='all')
            f_selector.fit(X_numeric, y)
            f_scores = dict(zip(X_numeric.columns, f_selector.scores_))
            feature_scores['f_statistic'] = f_scores
        except Exception as e:
            logger.warning(f"F-statistic feature selection failed: {e}")
        
        # Method 2: Mutual Information
        try:
            mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
            mi_selector.fit(X_numeric, y)
            mi_scores = dict(zip(X_numeric.columns, mi_selector.scores_))
            feature_scores['mutual_info'] = mi_scores
        except Exception as e:
            logger.warning(f"Mutual information feature selection failed: {e}")
        
        # Method 3: Correlation with target
        try:
            correlations = X_numeric.corrwith(y).abs()
            corr_scores = correlations.to_dict()
            feature_scores['correlation'] = corr_scores
        except Exception as e:
            logger.warning(f"Correlation feature selection failed: {e}")
        
        # Combine scores (average ranking)
        all_features = set()
        for method_scores in feature_scores.values():
            all_features.update(method_scores.keys())
        
        combined_scores = {}
        for feature in all_features:
            scores = []
            for method, method_scores in feature_scores.items():
                if feature in method_scores:
                    # Normalize scores to 0-1 range
                    max_score = max(method_scores.values())
                    min_score = min(method_scores.values())
                    if max_score != min_score:
                        normalized_score = (method_scores[feature] - min_score) / (max_score - min_score)
                    else:
                        normalized_score = 1.0
                    scores.append(normalized_score)
            
            combined_scores[feature] = np.mean(scores) if scores else 0
        
        # Select top features
        selected_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        self.selected_features = [feature[0] for feature in selected_features]
        
        # Store feature analysis
        self.feature_analysis = {
            'feature_scores': feature_scores,
            'combined_scores': combined_scores,
            'selected_features': self.selected_features,
            'selection_method': 'Combined ranking (F-statistic, Mutual Info, Correlation)'
        }
        
        logger.info(f"Feature selection completed. Selected features: {self.selected_features}")
        return self.selected_features
    
    def normalize_features(self) -> None:
        """Normalize selected features using StandardScaler"""
        logger.info("Normalizing selected features...")
        
        if not self.selected_features:
            logger.error("No features selected for normalization")
            return
        
        # Prepare data with selected features
        X_selected = self.combined_train[self.selected_features]
        
        # Fit and transform the scaler
        X_normalized = self.scaler.fit_transform(X_selected)
        
        # Update the dataframe
        for i, feature in enumerate(self.selected_features):
            self.combined_train[f"{feature}_normalized"] = X_normalized[:, i]
        
        # Store normalization info
        self.preprocessing_summary['normalization'] = {
            'method': 'StandardScaler',
            'features_normalized': self.selected_features,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist()
        }
        
        logger.info(f"Feature normalization completed for {len(self.selected_features)} features")
    
    def create_engineered_features(self) -> List[str]:
        """Create 5 new engineered features from selected features"""
        logger.info("Creating 5 new engineered features...")
        
        if not self.selected_features:
            logger.error("No selected features available for engineering")
            return []
        
        engineered_features = []
        
        try:
            # Feature 1: Claim to Premium Ratio
            if 'Total_Claim' in self.selected_features and 'Policy_Premium' in self.selected_features:
                self.combined_train['Claim_Premium_Ratio'] = (
                    self.combined_train['Total_Claim'] / 
                    (self.combined_train['Policy_Premium'] + 1)  # Add 1 to avoid division by zero
                )
                engineered_features.append('Claim_Premium_Ratio')
            
            # Feature 2: Age Risk Score
            if 'Age_Insured' in self.selected_features:
                # Higher risk for very young (<25) and older (>65) drivers
                self.combined_train['Age_Risk_Score'] = self.combined_train['Age_Insured'].apply(
                    lambda x: 2 if x < 25 or x > 65 else 1 if x < 30 or x > 60 else 0
                )
                engineered_features.append('Age_Risk_Score')
            
            # Feature 3: Vehicle Value to Claim Ratio
            if 'Vehicle_Cost' in self.selected_features and 'Total_Claim' in self.selected_features:
                self.combined_train['Vehicle_Claim_Ratio'] = (
                    self.combined_train['Total_Claim'] / 
                    (self.combined_train['Vehicle_Cost'] + 1)
                )
                engineered_features.append('Vehicle_Claim_Ratio')
            
            # Feature 4: Claim Complexity Score
            claim_components = ['Injury_Claim', 'Property_Claim', 'Vehicle_Claim']
            available_components = [col for col in claim_components if col in self.selected_features]
            
            if len(available_components) >= 2:
                # Count non-zero claim components
                self.combined_train['Claim_Complexity_Score'] = 0
                for component in available_components:
                    self.combined_train['Claim_Complexity_Score'] += (
                        self.combined_train[component] > 0).astype(int)
                engineered_features.append('Claim_Complexity_Score')
            
            # Feature 5: Time-based Risk Score
            date_columns = ['Policy_Start_Date', 'Accident_Date', 'Claims_Date']
            available_dates = [col for col in date_columns if col in self.combined_train.columns]
            
            if len(available_dates) >= 2:
                # Convert to datetime if not already
                for col in available_dates:
                    if self.combined_train[col].dtype == 'object':
                        self.combined_train[col] = pd.to_datetime(self.combined_train[col], errors='coerce')
                
                # Calculate days between policy start and accident
                if 'Policy_Start_Date' in available_dates and 'Accident_Date' in available_dates:
                    self.combined_train['Days_Policy_To_Accident'] = (
                        self.combined_train['Accident_Date'] - self.combined_train['Policy_Start_Date']
                    ).dt.days
                    
                    # Risk score: higher risk for very quick claims (< 30 days)
                    self.combined_train['Time_Risk_Score'] = self.combined_train['Days_Policy_To_Accident'].apply(
                        lambda x: 2 if x < 30 else 1 if x < 90 else 0
                    )
                    engineered_features.append('Time_Risk_Score')
        
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
        
        # If we couldn't create 5 features, create simple mathematical combinations
        while len(engineered_features) < 5 and len(self.selected_features) >= 2:
            try:
                # Create interaction features from top selected features
                feature_idx = len(engineered_features)
                if feature_idx < len(self.selected_features) - 1:
                    feat1 = self.selected_features[feature_idx]
                    feat2 = self.selected_features[feature_idx + 1]
                    
                    new_feature_name = f"{feat1}_{feat2}_interaction"
                    self.combined_train[new_feature_name] = (
                        self.combined_train[feat1] * self.combined_train[feat2]
                    )
                    engineered_features.append(new_feature_name)
                else:
                    break
            except Exception as e:
                logger.warning(f"Could not create interaction feature: {e}")
                break
        
        self.engineered_features = engineered_features
        
        # Store feature engineering info
        self.preprocessing_summary['feature_engineering'] = {
            'engineered_features': engineered_features,
            'engineering_methods': [
                'Claim to Premium Ratio',
                'Age Risk Score',
                'Vehicle Value to Claim Ratio', 
                'Claim Complexity Score',
                'Time-based Risk Score'
            ],
            'total_engineered': len(engineered_features)
        }
        
        logger.info(f"Feature engineering completed. Created {len(engineered_features)} new features: {engineered_features}")
        return engineered_features
#!/usr/bin/env python3
"""
ML Fraud Detection Preprocessing Pipeline - Part 3
=================================================
PDF Generation and Visualization Components
"""

# Continuation of FraudDetectionPreprocessor class

    def create_preprocessing_visualizations(self) -> None:
        """Create comprehensive visualizations for preprocessing analysis"""
        logger.info("Creating preprocessing visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)
        
        # 1. Missing Values Heatmap
        plt.figure(figsize=fig_size)
        missing_data = self.combined_train.isnull()
        sns.heatmap(missing_data, cbar=True, cmap='viridis', yticklabels=False)
        plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.report_dir / 'missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Data Types Distribution
        plt.figure(figsize=(12, 6))
        dtype_counts = self.combined_train.dtypes.value_counts()
        plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Data Types Distribution', fontsize=16, fontweight='bold')
        plt.savefig(self.report_dir / 'data_types_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Target Variable Distribution
        if 'Fraud_Ind' in self.combined_train.columns:
            plt.figure(figsize=(10, 6))
            fraud_counts = self.combined_train['Fraud_Ind'].value_counts()
            plt.subplot(1, 2, 1)
            plt.pie(fraud_counts.values, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90)
            plt.title('Fraud Distribution')
            
            plt.subplot(1, 2, 2)
            sns.countplot(data=self.combined_train, x='Fraud_Ind')
            plt.title('Fraud Count Distribution')
            plt.xlabel('Fraud Indicator (0=No, 1=Yes)')
            
            plt.tight_layout()
            plt.savefig(self.report_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Outlier Analysis Visualization
        if self.outlier_analysis:
            numeric_cols = list(self.outlier_analysis.keys())[:6]  # Top 6 for visualization
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    sns.boxplot(data=self.combined_train, y=col, ax=axes[i])
                    axes[i].set_title(f'Outliers in {col}')
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.report_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Feature Importance Visualization
        if self.feature_analysis and 'combined_scores' in self.feature_analysis:
            plt.figure(figsize=(12, 8))
            
            # Get top 15 features for visualization
            top_features = sorted(self.feature_analysis['combined_scores'].items(), 
                                key=lambda x: x[1], reverse=True)[:15]
            
            features, scores = zip(*top_features)
            
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance Score')
            plt.title('Top 15 Feature Importance Scores', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.report_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Preprocessing visualizations created successfully")
    
    def generate_preprocessing_pdf(self) -> str:
        """Generate comprehensive PDF report for preprocessing analysis"""
        logger.info("Generating preprocessing PDF report...")
        
        pdf_path = self.report_dir / "preprocessing_analysis_report.pdf"
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'Auto Insurance Fraud Detection - Preprocessing Analysis', 0, 1, 'C')
                self.ln(10)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.add_page()
        
        # Title and Introduction
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8, 
            f"This report presents a comprehensive analysis of the data preprocessing pipeline "
            f"for auto insurance fraud detection. The analysis was conducted on {self.preprocessing_summary['initial_train_shape'][0]} "
            f"training records with {self.preprocessing_summary['initial_train_shape'][1]} features.")
        pdf.ln(10)
        
        # Data Quality Analysis
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '1. Data Quality Analysis', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        
        # Missing Values Section
        if 'data_quality' in self.preprocessing_summary:
            missing_info = self.preprocessing_summary['data_quality']['missing_values']
            pdf.multi_cell(0, 6, f"Missing Values Analysis:")
            pdf.multi_cell(0, 6, f"• Total columns with missing values: {len(missing_info['columns_with_missing'])}")
            
            if missing_info['columns_with_missing']:
                pdf.multi_cell(0, 6, f"• Columns with highest missing percentages:")
                sorted_missing = sorted(missing_info['percentages'].items(), key=lambda x: x[1], reverse=True)[:5]
                for col, pct in sorted_missing:
                    if pct > 0:
                        pdf.multi_cell(0, 6, f"  - {col}: {pct:.2f}%")
        
        pdf.ln(5)
        
        # Duplicates Section
        if 'duplicates_handled' in self.preprocessing_summary:
            dup_info = self.preprocessing_summary['duplicates_handled']
            pdf.multi_cell(0, 6, f"Duplicate Records Analysis:")
            pdf.multi_cell(0, 6, f"• Initial record count: {dup_info['initial_count']}")
            pdf.multi_cell(0, 6, f"• Final record count: {dup_info['final_count']}")
            pdf.multi_cell(0, 6, f"• Duplicates removed: {dup_info['removed_count']} ({dup_info['removal_percentage']:.2f}%)")
        
        pdf.ln(10)
        
        # Missing Value Handling
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '2. Missing Value Handling Strategy', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        if 'missing_values_handled' in self.preprocessing_summary:
            mv_info = self.preprocessing_summary['missing_values_handled']
            pdf.multi_cell(0, 6, f"Strategy Applied: {mv_info['strategy_used']}")
            
            if mv_info['high_missing_dropped']:
                pdf.multi_cell(0, 6, f"• Columns dropped (>50% missing): {', '.join(mv_info['high_missing_dropped'])}")
            
            pdf.multi_cell(0, 6, f"• Remaining missing values after processing: {mv_info['remaining_missing_count']}")
        
        pdf.ln(10)
        
        # Outlier Analysis
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '3. Outlier Analysis', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        if self.outlier_analysis:
            pdf.multi_cell(0, 6, f"Outlier detection performed on {len(self.outlier_analysis)} numerical columns using:")
            pdf.multi_cell(0, 6, "• IQR Method (1.5 * IQR rule)")
            pdf.multi_cell(0, 6, "• Z-Score Method (threshold: 3)")
            pdf.multi_cell(0, 6, "• Modified Z-Score Method (threshold: 3.5)")
            
            # Show top columns with outliers
            outlier_summary = []
            for col, info in self.outlier_analysis.items():
                outlier_summary.append((col, info['iqr_outliers']))
            
            outlier_summary.sort(key=lambda x: x[1], reverse=True)
            
            pdf.multi_cell(0, 6, f"Top columns with outliers (IQR method):")
            for col, count in outlier_summary[:5]:
                total = self.outlier_analysis[col]['total_values']
                percentage = (count / total) * 100 if total > 0 else 0
                pdf.multi_cell(0, 6, f"• {col}: {count} outliers ({percentage:.2f}%)")
        
        pdf.ln(10)
        
        # Categorical Encoding
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '4. Categorical Feature Encoding', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        if 'categorical_encoding' in self.preprocessing_summary:
            enc_info = self.preprocessing_summary['categorical_encoding']
            
            label_encoded = [col for col, info in enc_info.items() if info['method'] == 'label_encoding']
            freq_encoded = [col for col, info in enc_info.items() if info['method'] == 'frequency_encoding']
            
            pdf.multi_cell(0, 6, f"Encoding Methods Applied:")
            pdf.multi_cell(0, 6, f"• Label Encoding: {len(label_encoded)} columns")
            pdf.multi_cell(0, 6, f"• Frequency Encoding: {len(freq_encoded)} columns")
            
            if label_encoded:
                pdf.multi_cell(0, 6, f"Label Encoded Columns: {', '.join(label_encoded[:5])}")
            if freq_encoded:
                pdf.multi_cell(0, 6, f"Frequency Encoded Columns: {', '.join(freq_encoded[:5])}")
        
        pdf.add_page()
        
        # Feature Selection
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '5. Feature Selection Analysis', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        if self.feature_analysis:
            pdf.multi_cell(0, 6, f"Feature Selection Method: {self.feature_analysis['selection_method']}")
            pdf.multi_cell(0, 6, f"Total features analyzed: {len(self.feature_analysis['combined_scores'])}")
            pdf.multi_cell(0, 6, f"Selected features: {len(self.selected_features)}")
            pdf.ln(5)
            
            pdf.multi_cell(0, 6, "Top Selected Features:")
            for i, feature in enumerate(self.selected_features[:10], 1):
                score = self.feature_analysis['combined_scores'].get(feature, 0)
                pdf.multi_cell(0, 6, f"{i}. {feature} (Score: {score:.4f})")
        
        pdf.ln(10)
        
        # Normalization
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '6. Feature Normalization', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        if 'normalization' in self.preprocessing_summary:
            norm_info = self.preprocessing_summary['normalization']
            pdf.multi_cell(0, 6, f"Normalization Method: {norm_info['method']}")
            pdf.multi_cell(0, 6, f"Features normalized: {len(norm_info['features_normalized'])}")
            pdf.multi_cell(0, 6, "Normalized features have mean=0 and standard deviation=1")
        
        pdf.ln(10)
        
        # Recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '7. Recommendations for Model Building', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        recommendations = [
            "• Use the selected features for initial model training",
            "• Consider ensemble methods (Random Forest, XGBoost) for handling feature interactions",
            "• Apply class balancing techniques due to imbalanced fraud distribution",
            "• Use cross-validation with stratification to maintain class distribution",
            "• Monitor for overfitting given the engineered features",
            "• Consider feature importance from tree-based models for further selection"
        ]
        
        for rec in recommendations:
            pdf.multi_cell(0, 6, rec)
        
        # Save PDF
        pdf.output(str(pdf_path))
        logger.info(f"Preprocessing PDF report saved to: {pdf_path}")
        
        return str(pdf_path)
    
    def generate_feature_engineering_pdf(self) -> str:
        """Generate PDF report for feature engineering and selection"""
        logger.info("Generating feature engineering PDF report...")
        
        pdf_path = self.report_dir / "feature_engineering_report.pdf"
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'Feature Engineering & Selection Analysis', 0, 1, 'C')
                self.ln(10)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.add_page()
        
        # Executive Summary
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Feature Engineering Executive Summary', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8, 
            f"This report details the feature selection and engineering process for the fraud detection model. "
            f"From the original dataset, {len(self.selected_features)} key features were selected and "
            f"{len(self.engineered_features)} new features were engineered to enhance model performance.")
        pdf.ln(10)
        
        # Selected Features Analysis
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '1. Selected Features Analysis', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        if self.feature_analysis:
            pdf.multi_cell(0, 6, f"Selection Methodology:")
            pdf.multi_cell(0, 6, f"• Combined ranking approach using multiple statistical methods")
            pdf.multi_cell(0, 6, f"• F-statistic for linear relationships with target")
            pdf.multi_cell(0, 6, f"• Mutual Information for non-linear relationships")
            pdf.multi_cell(0, 6, f"• Correlation analysis for direct associations")
            pdf.ln(5)
            
            pdf.multi_cell(0, 6, f"Selected Features and Their Importance:")
            for i, feature in enumerate(self.selected_features, 1):
                score = self.feature_analysis['combined_scores'].get(feature, 0)
                pdf.multi_cell(0, 6, f"{i}. {feature}")
                pdf.multi_cell(0, 6, f"   Importance Score: {score:.4f}")
                
                # Add business interpretation
                interpretations = {
                    'Total_Claim': 'Higher claim amounts may indicate fraudulent activity',
                    'Policy_Premium': 'Premium amount reflects risk assessment and coverage level',
                    'Age_Insured': 'Age demographics show different fraud patterns',
                    'Vehicle_Cost': 'Expensive vehicles may be targets for fraud',
                    'Accident_Severity': 'Severity level correlates with claim legitimacy',
                    'Policy_BI': 'Bodily injury coverage affects claim amounts',
                    'Annual_Mileage': 'Mileage patterns can indicate usage fraud'
                }
                
                if feature in interpretations:
                    pdf.multi_cell(0, 6, f"   Business Relevance: {interpretations[feature]}")
                pdf.ln(3)
        
        pdf.add_page()
        
        # Engineered Features
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '2. Engineered Features Analysis', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        if self.engineered_features:
            pdf.multi_cell(0, 6, f"Total Engineered Features: {len(self.engineered_features)}")
            pdf.ln(5)
            
            feature_descriptions = {
                'Claim_Premium_Ratio': {
                    'description': 'Ratio of total claim amount to policy premium',
                    'importance': 'High ratios may indicate inflated claims or premium fraud',
                    'calculation': 'Total_Claim / (Policy_Premium + 1)'
                },
                'Age_Risk_Score': {
                    'description': 'Risk score based on insured age demographics',
                    'importance': 'Young (<25) and older (>65) drivers have different risk profiles',
                    'calculation': 'Categorical scoring: 2 (high risk), 1 (medium risk), 0 (low risk)'
                },
                'Vehicle_Claim_Ratio': {
                    'description': 'Ratio of claim amount to vehicle value',
                    'importance': 'Unusually high ratios may indicate vehicle value fraud',
                    'calculation': 'Total_Claim / (Vehicle_Cost + 1)'
                },
                'Claim_Complexity_Score': {
                    'description': 'Number of different claim components involved',
                    'importance': 'Complex claims with multiple components may require more scrutiny',
                    'calculation': 'Count of non-zero claim components (Injury, Property, Vehicle)'
                },
                'Time_Risk_Score': {
                    'description': 'Risk score based on timing between policy start and accident',
                    'importance': 'Very quick claims (<30 days) may indicate premeditated fraud',
                    'calculation': 'Time-based categorical scoring'
                }
            }
            
            for i, feature in enumerate(self.engineered_features, 1):
                pdf.set_font('Arial', 'B', 10)
                pdf.multi_cell(0, 6, f"{i}. {feature}")
                pdf.set_font('Arial', '', 10)
                
                if feature in feature_descriptions:
                    desc = feature_descriptions[feature]
                    pdf.multi_cell(0, 6, f"   Description: {desc['description']}")
                    pdf.multi_cell(0, 6, f"   Business Importance: {desc['importance']}")
                    pdf.multi_cell(0, 6, f"   Calculation: {desc['calculation']}")
                else:
                    pdf.multi_cell(0, 6, f"   Mathematical combination of selected features")
                    pdf.multi_cell(0, 6, f"   Captures feature interactions for improved model performance")
                
                pdf.ln(5)
        
        # Feature Engineering Best Practices
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '3. Feature Engineering Best Practices Applied', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        best_practices = [
            "Domain Knowledge Integration: Features designed based on insurance fraud patterns",
            "Ratio Features: Created meaningful ratios to capture relative relationships",
            "Categorical Risk Scoring: Converted continuous variables to risk categories",
            "Time-based Features: Incorporated temporal patterns in fraud detection",
            "Interaction Features: Captured relationships between multiple variables",
            "Normalization: Applied StandardScaler to ensure feature scale consistency",
            "Business Interpretability: Ensured all features have clear business meaning"
        ]
        
        for practice in best_practices:
            pdf.multi_cell(0, 6, f"• {practice}")
        
        pdf.ln(10)
        
        # Model Recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '4. Recommendations for Model Development', 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        model_recommendations = [
            "Feature Usage: Use both selected original features and engineered features",
            "Model Types: Tree-based models (Random Forest, XGBoost) work well with these features",
            "Feature Importance: Monitor feature importance to validate engineering decisions",
            "Cross-validation: Use stratified CV to maintain fraud class distribution",
            "Feature Scaling: Normalized features ready for linear models and neural networks",
            "Interpretability: Engineered features enhance model explainability for business users"
        ]
        
        for rec in model_recommendations:
            pdf.multi_cell(0, 6, f"• {rec}")
        
        # Save PDF
        pdf.output(str(pdf_path))
        logger.info(f"Feature engineering PDF report saved to: {pdf_path}")
        
        return str(pdf_path)
    
    def run_complete_preprocessing(self) -> Dict[str, str]:
        """Run the complete preprocessing pipeline"""
        logger.info("Starting complete preprocessing pipeline...")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Analyze data quality
            self.analyze_data_quality()
            
            # Step 3: Handle missing values
            self.handle_missing_values()
            
            # Step 4: Handle duplicates
            self.identify_and_handle_duplicates()
            
            # Step 5: Detect and handle outliers
            self.detect_outliers()
            self.handle_outliers(method='cap')  # Using capping method
            
            # Step 6: Encode categorical features
            self.encode_categorical_features()
            
            # Step 7: Select important features
            self.select_important_features(n_features=15)
            
            # Step 8: Normalize features
            self.normalize_features()
            
            # Step 9: Create engineered features
            self.create_engineered_features()
            
            # Step 10: Create visualizations
            self.create_preprocessing_visualizations()
            
            # Step 11: Generate PDF reports
            pdf1_path = self.generate_preprocessing_pdf()
            pdf2_path = self.generate_feature_engineering_pdf()
            
            # Save processed data
            processed_data_path = self.report_dir / "processed_training_data.csv"
            self.combined_train.to_csv(processed_data_path, index=False)
            
            logger.info("Complete preprocessing pipeline finished successfully!")
            
            return {
                'preprocessing_pdf': pdf1_path,
                'feature_engineering_pdf': pdf2_path,
                'processed_data': str(processed_data_path),
                'report_directory': str(self.report_dir),
                'selected_features': self.selected_features,
                'engineered_features': self.engineered_features
            }
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

# Main execution function
def main():
    """Main function to run the preprocessing pipeline"""
    
    # Initialize preprocessor
    data_path = "/Users/debabratapattnayak/web-dev/learnathon/dataset"
    output_dir = "/Users/debabratapattnayak/web-dev/learnathon/ml_analysis_reports"
    
    preprocessor = FraudDetectionPreprocessor(data_path, output_dir)
    
    # Run complete preprocessing
    results = preprocessor.run_complete_preprocessing()
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Report Directory: {results['report_directory']}")
    print(f"Preprocessing Report: {results['preprocessing_pdf']}")
    print(f"Feature Engineering Report: {results['feature_engineering_pdf']}")
    print(f"Processed Data: {results['processed_data']}")
    print(f"Selected Features: {len(results['selected_features'])}")
    print(f"Engineered Features: {len(results['engineered_features'])}")
    print("="*80)

if __name__ == "__main__":
    main()
