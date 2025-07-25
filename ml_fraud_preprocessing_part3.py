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
