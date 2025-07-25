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
