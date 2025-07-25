#!/usr/bin/env python3
"""
Comprehensive Fraud Detection Model Testing Framework
====================================================
Testing 10 different ML algorithms with comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
from pathlib import Path
import joblib
import time

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score, log_loss
)

# Model imports
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb

warnings.filterwarnings('ignore')

class FraudModelTester:
    """
    Comprehensive model testing framework for fraud detection
    Tests 10 different algorithms with 8+ evaluation metrics
    """
    
    def __init__(self, data_path: str, output_dir: str = None):
        """Initialize the model tester"""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path("model_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        # Results storage
        self.model_results = {}
        self.trained_models = {}
        self.evaluation_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
            'balanced_accuracy', 'matthews_corrcoef', 'log_loss'
        ]
        
        print(f"🚀 Fraud Model Tester initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_and_prepare_data(self):
        """Load and prepare the processed data for modeling"""
        print("\n📊 Loading processed data...")
        
        # Load the updated processed data
        df = pd.read_csv(self.data_path)
        print(f"Data shape: {df.shape}")
        
        # Define feature sets
        selected_features = [
            'Annual_Mileage', 'DiffIN_Mileage', 'Auto_Make', 'Vehicle_Cost',
            'Accident_Severity', 'Garage_Location', 'Collision_Type',
            'authorities_contacted', 'Commute_Discount', 'Witnesses',
            'Umbrella_Limit', 'Policy_State', 'Num_of_Vehicles_Involved',
            'Acccident_State'
        ]
        
        # Use normalized features if available
        normalized_features = [f"{feat}_normalized" for feat in selected_features 
                             if f"{feat}_normalized" in df.columns]
        
        # Use original features if normalized not available
        available_features = []
        for feat in selected_features:
            if f"{feat}_normalized" in df.columns:
                available_features.append(f"{feat}_normalized")
            elif feat in df.columns:
                available_features.append(feat)
        
        # Add engineered features
        engineered_features = [
            'Claim_Premium_Ratio', 'Age_Risk_Score', 'Vehicle_Claim_Ratio',
            'Mileage_Discrepancy_Score', 'Vehicle_Age_Risk'
        ]
        
        for feat in engineered_features:
            if feat in df.columns:
                available_features.append(feat)
        
        print(f"Available features for modeling: {len(available_features)}")
        
        # Prepare features and target
        X = df[available_features].fillna(0)  # Handle any remaining NaN
        y = df['Fraud_Ind']
        
        self.feature_names = available_features
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(self.y_train)}")
        print(f"Class distribution - Test: {np.bincount(self.y_test)}")
        
        return X, y
    
    def initialize_models(self):
        """Initialize all 10 models for testing"""
        print("\n🤖 Initializing models...")
        
        models = {
            'Logistic_Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
            ),
            'K_Nearest_Neighbors': KNeighborsClassifier(
                n_neighbors=5, n_jobs=-1
            ),
            'Naive_Bayes': GaussianNB(),
            'Decision_Tree': DecisionTreeClassifier(
                random_state=42, class_weight='balanced', max_depth=10
            ),
            'Support_Vector_Machine': SVC(
                random_state=42, class_weight='balanced', probability=True, kernel='rbf'
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, eval_metric='logloss', n_jobs=-1,
                scale_pos_weight=3  # Handle class imbalance
            ),
            'SGD_Classifier': SGDClassifier(
                random_state=42, class_weight='balanced', loss='log_loss'
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                random_state=42, n_estimators=100
            )
        }
        
        # Voting Classifier (ensemble of top 3 models)
        voting_models = [
            ('rf', models['Random_Forest']),
            ('xgb', models['XGBoost']),
            ('lr', models['Logistic_Regression'])
        ]
        
        models['Voting_Classifier'] = VotingClassifier(
            estimators=voting_models, voting='soft'
        )
        
        print(f"✅ Initialized {len(models)} models")
        return models
    
    def evaluate_model(self, model, model_name, X_test, y_test, y_pred, y_pred_proba):
        """Comprehensive model evaluation with 8+ metrics"""
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if y_pred_proba is not None else 0,
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba) if y_pred_proba is not None else np.inf
        }
        
        # Additional metrics
        results['precision_fraud'] = precision_score(y_test, y_pred, pos_label=1)
        results['recall_fraud'] = recall_score(y_test, y_pred, pos_label=1)
        results['f1_fraud'] = f1_score(y_test, y_pred, pos_label=1)
        
        return results
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n🏋️ Training and evaluating models...")
        
        models = self.initialize_models()
        results_list = []
        
        for model_name, model in models.items():
            print(f"\n📈 Training {model_name}...")
            start_time = time.time()
            
            try:
                # Train the model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                
                # Get prediction probabilities if available
                try:
                    y_pred_proba = model.predict_proba(self.X_test)
                except:
                    y_pred_proba = None
                
                # Calculate training time
                training_time = time.time() - start_time
                
                # Evaluate model
                results = self.evaluate_model(
                    model, model_name, self.X_test, self.y_test, y_pred, y_pred_proba
                )
                results['training_time'] = training_time
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train, cv=5, scoring='roc_auc'
                )
                results['cv_roc_auc_mean'] = cv_scores.mean()
                results['cv_roc_auc_std'] = cv_scores.std()
                
                # Store results
                results_list.append(results)
                self.trained_models[model_name] = model
                
                print(f"✅ {model_name} completed in {training_time:.2f}s")
                print(f"   Accuracy: {results['accuracy']:.4f}")
                print(f"   ROC-AUC: {results['roc_auc']:.4f}")
                print(f"   F1-Score: {results['f1_score']:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        self.model_results = pd.DataFrame(results_list)
        return self.model_results
    
    def create_model_comparison_visualizations(self):
        """Create comprehensive visualizations for model comparison"""
        print("\n📊 Creating model comparison visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        axes[0, 0].bar(self.model_results['model_name'], self.model_results['accuracy'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # ROC-AUC comparison
        axes[0, 1].bar(self.model_results['model_name'], self.model_results['roc_auc'])
        axes[0, 1].set_title('ROC-AUC Comparison')
        axes[0, 1].set_ylabel('ROC-AUC Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[1, 0].bar(self.model_results['model_name'], self.model_results['f1_score'])
        axes[1, 0].set_title('F1-Score Comparison')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Training Time comparison
        axes[1, 1].bar(self.model_results['model_name'], self.model_results['training_time'])
        axes[1, 1].set_title('Training Time Comparison')
        axes[1, 1].set_ylabel('Training Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Comprehensive Metrics Heatmap
        plt.figure(figsize=(14, 8))
        
        metrics_for_heatmap = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
            'balanced_accuracy', 'matthews_corrcoef'
        ]
        
        heatmap_data = self.model_results.set_index('model_name')[metrics_for_heatmap]
        
        sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlGn', fmt='.3f', 
                   cbar_kws={'label': 'Score'})
        plt.title('Model Performance Metrics Heatmap')
        plt.ylabel('Metrics')
        plt.xlabel('Models')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curves for top 5 models
        plt.figure(figsize=(12, 8))
        
        # Get top 5 models by ROC-AUC
        top_5_models = self.model_results.nlargest(5, 'roc_auc')
        
        for _, row in top_5_models.iterrows():
            model_name = row['model_name']
            if model_name in self.trained_models:
                model = self.trained_models[model_name]
                try:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {row['roc_auc']:.3f})")
                except:
                    continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Top 5 Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'roc_curves_top5.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations created successfully")
    
    def generate_model_recommendation(self):
        """Generate comprehensive model recommendation based on multiple criteria"""
        print("\n🎯 Generating model recommendations...")
        
        # Sort models by different criteria
        best_accuracy = self.model_results.loc[self.model_results['accuracy'].idxmax()]
        best_roc_auc = self.model_results.loc[self.model_results['roc_auc'].idxmax()]
        best_f1 = self.model_results.loc[self.model_results['f1_score'].idxmax()]
        best_fraud_recall = self.model_results.loc[self.model_results['recall_fraud'].idxmax()]
        fastest_training = self.model_results.loc[self.model_results['training_time'].idxmin()]
        
        # Calculate composite score for overall best model
        # Weighted scoring: ROC-AUC (40%), F1 (30%), Balanced Accuracy (20%), Speed (10%)
        self.model_results['composite_score'] = (
            0.4 * self.model_results['roc_auc'] +
            0.3 * self.model_results['f1_score'] +
            0.2 * self.model_results['balanced_accuracy'] +
            0.1 * (1 - self.model_results['training_time'] / self.model_results['training_time'].max())
        )
        
        best_overall = self.model_results.loc[self.model_results['composite_score'].idxmax()]
        
        # Scalability assessment
        scalability_scores = {
            'Logistic_Regression': 9,
            'Random_Forest': 7,
            'K_Nearest_Neighbors': 4,
            'Naive_Bayes': 9,
            'Decision_Tree': 6,
            'Support_Vector_Machine': 3,
            'XGBoost': 8,
            'SGD_Classifier': 10,
            'Gradient_Boosting': 6,
            'Voting_Classifier': 5
        }
        
        self.model_results['scalability_score'] = self.model_results['model_name'].map(scalability_scores)
        
        recommendations = {
            'best_overall': best_overall,
            'best_accuracy': best_accuracy,
            'best_roc_auc': best_roc_auc,
            'best_f1': best_f1,
            'best_fraud_detection': best_fraud_recall,
            'fastest_training': fastest_training,
            'most_scalable': self.model_results.loc[self.model_results['scalability_score'].idxmax()]
        }
        
        return recommendations
    
    def save_results(self):
        """Save all results to files"""
        print("\n💾 Saving results...")
        
        # Save model results
        self.model_results.to_csv(self.output_dir / 'model_comparison_results.csv', index=False)
        
        # Save trained models
        models_dir = self.output_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            joblib.dump(model, models_dir / f'{model_name}.pkl')
        
        print(f"✅ Results saved to {self.output_dir}")
    
    def run_complete_model_testing(self):
        """Run the complete model testing pipeline"""
        print("🚀 Starting Complete Model Testing Pipeline")
        print("=" * 60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train and evaluate models
        results_df = self.train_and_evaluate_models()
        
        # Create visualizations
        self.create_model_comparison_visualizations()
        
        # Generate recommendations
        recommendations = self.generate_model_recommendation()
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 MODEL TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"Models tested: {len(results_df)}")
        print(f"Best overall model: {recommendations['best_overall']['model_name']}")
        print(f"Best ROC-AUC: {recommendations['best_roc_auc']['model_name']} ({recommendations['best_roc_auc']['roc_auc']:.4f})")
        print(f"Best F1-Score: {recommendations['best_f1']['model_name']} ({recommendations['best_f1']['f1_score']:.4f})")
        print(f"Most scalable: {recommendations['most_scalable']['model_name']}")
        
        return results_df, recommendations

def main():
    """Main function to run model testing"""
    
    # Set up paths
    data_path = "/Users/debabratapattnayak/web-dev/learnathon/ml_analysis_reports/updated_2025-07-25_23-19-01/updated_processed_training_data.csv"
    output_dir = "/Users/debabratapattnayak/web-dev/learnathon/model-test/results"
    
    # Initialize and run model tester
    tester = FraudModelTester(data_path, output_dir)
    results_df, recommendations = tester.run_complete_model_testing()
    
    return results_df, recommendations

if __name__ == "__main__":
    results, recommendations = main()
