#!/usr/bin/env python3
"""
Simplified Fraud Detection Model Testing
=======================================
Tests 10 ML algorithms with comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
from pathlib import Path
import time

# Basic sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, matthews_corrcoef, 
    balanced_accuracy_score
)

# Model imports
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the processed data"""
    print("📊 Loading processed data...")
    
    # Load the updated processed data
    data_path = "/Users/debabratapattnayak/web-dev/learnathon/ml_analysis_reports/updated_2025-07-25_23-19-01/updated_processed_training_data.csv"
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Define feature sets (using available features)
    selected_features = [
        'Annual_Mileage', 'DiffIN_Mileage', 'Auto_Make', 'Vehicle_Cost',
        'Accident_Severity', 'Garage_Location', 'Collision_Type',
        'authorities_contacted', 'Commute_Discount', 'Witnesses',
        'Umbrella_Limit', 'Policy_State', 'Num_of_Vehicles_Involved',
        'Acccident_State'
    ]
    
    # Use available features
    available_features = []
    for feat in selected_features:
        if feat in df.columns:
            available_features.append(feat)
        elif f"{feat}_normalized" in df.columns:
            available_features.append(f"{feat}_normalized")
    
    # Add engineered features if available
    engineered_features = [
        'Claim_Premium_Ratio', 'Age_Risk_Score', 'Vehicle_Claim_Ratio',
        'Mileage_Discrepancy_Score', 'Vehicle_Age_Risk'
    ]
    
    for feat in engineered_features:
        if feat in df.columns:
            available_features.append(feat)
    
    print(f"Available features: {len(available_features)}")
    
    # Prepare features and target
    X = df[available_features].fillna(0)
    y = df['Fraud_Ind']
    
    return X, y, available_features

def initialize_models():
    """Initialize all models for testing"""
    print("🤖 Initializing models...")
    
    models = {
        'Logistic_Regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'K_Nearest_Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive_Bayes': GaussianNB(),
        'Decision_Tree': DecisionTreeClassifier(
            random_state=42, class_weight='balanced', max_depth=10
        ),
        'Support_Vector_Machine': SVC(
            random_state=42, class_weight='balanced', probability=True, kernel='rbf'
        ),
        'SGD_Classifier': SGDClassifier(
            random_state=42, class_weight='balanced', loss='log_loss'
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=100
        )
    }
    
    # Try to add XGBoost if available
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=42, eval_metric='logloss', n_jobs=-1,
            scale_pos_weight=3
        )
        print("✅ XGBoost added")
    except ImportError:
        print("⚠️ XGBoost not available, skipping")
    
    # Voting Classifier
    voting_models = [
        ('rf', models['Random_Forest']),
        ('lr', models['Logistic_Regression']),
        ('dt', models['Decision_Tree'])
    ]
    
    models['Voting_Classifier'] = VotingClassifier(
        estimators=voting_models, voting='soft'
    )
    
    print(f"✅ Initialized {len(models)} models")
    return models

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive evaluation metrics"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }
    
    # Add ROC-AUC if probabilities available
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['roc_auc'] = 0.5
    else:
        metrics['roc_auc'] = 0.5
    
    # Fraud-specific metrics
    metrics['precision_fraud'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_fraud'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['f1_fraud'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    return metrics

def train_and_evaluate_models(X, y):
    """Train and evaluate all models"""
    print("\n🏋️ Training and evaluating models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    models = initialize_models()
    results = []
    
    for model_name, model in models.items():
        print(f"\n📈 Training {model_name}...")
        start_time = time.time()
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            metrics['model_name'] = model_name
            metrics['training_time'] = time.time() - start_time
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
                metrics['cv_roc_auc_mean'] = cv_scores.mean()
                metrics['cv_roc_auc_std'] = cv_scores.std()
            except:
                metrics['cv_roc_auc_mean'] = 0.5
                metrics['cv_roc_auc_std'] = 0.0
            
            results.append(metrics)
            
            print(f"✅ {model_name} completed in {metrics['training_time']:.2f}s")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def create_visualizations(results_df, output_dir):
    """Create model comparison visualizations"""
    print("\n📊 Creating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy
    axes[0, 0].bar(results_df['model_name'], results_df['accuracy'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # ROC-AUC
    axes[0, 1].bar(results_df['model_name'], results_df['roc_auc'])
    axes[0, 1].set_title('ROC-AUC Comparison')
    axes[0, 1].set_ylabel('ROC-AUC Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # F1-Score
    axes[1, 0].bar(results_df['model_name'], results_df['f1_score'])
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Training Time
    axes[1, 1].bar(results_df['model_name'], results_df['training_time'])
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Metrics heatmap
    plt.figure(figsize=(14, 8))
    
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'balanced_accuracy']
    heatmap_data = results_df.set_index('model_name')[metrics_cols]
    
    sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlGn', fmt='.3f')
    plt.title('Model Performance Metrics Heatmap')
    plt.ylabel('Metrics')
    plt.xlabel('Models')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations created")

def generate_recommendations(results_df):
    """Generate model recommendations"""
    print("\n🎯 Generating recommendations...")
    
    # Sort by ROC-AUC
    results_sorted = results_df.sort_values('roc_auc', ascending=False)
    
    # Scalability scores
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
    
    results_df['scalability_score'] = results_df['model_name'].map(scalability_scores).fillna(5)
    
    # Calculate deployment score
    results_df['deployment_score'] = (
        0.4 * results_df['roc_auc'] +
        0.3 * results_df['f1_score'] +
        0.2 * results_df['balanced_accuracy'] +
        0.1 * (results_df['scalability_score'] / 10)
    )
    
    best_overall = results_df.loc[results_df['deployment_score'].idxmax()]
    best_roc_auc = results_df.loc[results_df['roc_auc'].idxmax()]
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    most_scalable = results_df.loc[results_df['scalability_score'].idxmax()]
    
    return {
        'best_overall': best_overall,
        'best_roc_auc': best_roc_auc,
        'best_f1': best_f1,
        'most_scalable': most_scalable,
        'results_sorted': results_sorted
    }

def main():
    """Main function"""
    print("🚀 Starting Fraud Detection Model Testing")
    print("=" * 50)
    
    # Create output directory
    output_dir = "/Users/debabratapattnayak/web-dev/learnathon/model-test/results"
    
    try:
        # Load data
        X, y, features = load_data()
        
        # Train and evaluate models
        results_df = train_and_evaluate_models(X, y)
        
        # Create visualizations
        create_visualizations(results_df, output_dir)
        
        # Generate recommendations
        recommendations = generate_recommendations(results_df)
        
        # Save results
        results_df.to_csv(f"{output_dir}/model_results.csv", index=False)
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎉 MODEL TESTING COMPLETED!")
        print("=" * 50)
        
        print(f"\n📊 TOP 5 MODELS BY ROC-AUC:")
        top_5 = recommendations['results_sorted'].head(5)
        for idx, row in top_5.iterrows():
            print(f"{idx+1}. {row['model_name']:<20} ROC-AUC: {row['roc_auc']:.4f} | F1: {row['f1_score']:.4f} | Time: {row['training_time']:.2f}s")
        
        print(f"\n🏆 RECOMMENDATIONS:")
        print(f"Best Overall: {recommendations['best_overall']['model_name']} (Score: {recommendations['best_overall']['deployment_score']:.4f})")
        print(f"Best ROC-AUC: {recommendations['best_roc_auc']['model_name']} ({recommendations['best_roc_auc']['roc_auc']:.4f})")
        print(f"Best F1-Score: {recommendations['best_f1']['model_name']} ({recommendations['best_f1']['f1_score']:.4f})")
        print(f"Most Scalable: {recommendations['most_scalable']['model_name']} ({recommendations['most_scalable']['scalability_score']}/10)")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print("✅ Ready for Streamlit application development!")
        
        return results_df, recommendations
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None

if __name__ == "__main__":
    results, recommendations = main()
