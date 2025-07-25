#!/usr/bin/env python3
"""
Fast Fraud Detection Model Testing (Without SVM)
===============================================
Tests 9 ML algorithms with comprehensive evaluation metrics
Removed SVM for faster execution
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

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score
)

# Model imports (excluding SVM)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def load_and_prepare_data():
    """Load and prepare the processed data for modeling"""
    print("📊 Loading and Preparing Data...")
    
    # Data path
    data_path = "/Users/debabratapattnayak/web-dev/learnathon/ml_analysis_reports/updated_2025-07-25_23-19-01/updated_processed_training_data.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None, None, None
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully: {df.shape}")
        
        # Define feature sets (prioritizing requested features)
        requested_features = ['Annual_Mileage', 'DiffIN_Mileage', 'Auto_Make', 'Vehicle_Cost']
        
        other_important_features = [
            'Accident_Severity', 'Garage_Location', 'Collision_Type',
            'authorities_contacted', 'Commute_Discount', 'Witnesses',
            'Umbrella_Limit', 'Policy_State', 'Num_of_Vehicles_Involved',
            'Acccident_State'
        ]
        
        # Check for normalized versions first, then original
        available_features = []
        
        # Add requested features (prioritized)
        for feat in requested_features:
            if f"{feat}_normalized" in df.columns:
                available_features.append(f"{feat}_normalized")
            elif feat in df.columns:
                available_features.append(feat)
        
        # Add other important features
        for feat in other_important_features:
            if f"{feat}_normalized" in df.columns:
                available_features.append(f"{feat}_normalized")
            elif feat in df.columns:
                available_features.append(feat)
        
        # Add engineered features if available
        engineered_features = [
            'Claim_Premium_Ratio', 'Age_Risk_Score', 'Vehicle_Claim_Ratio',
            'Mileage_Discrepancy_Score', 'Vehicle_Age_Risk'
        ]
        
        for feat in engineered_features:
            if feat in df.columns:
                available_features.append(feat)
        
        print(f"📊 Total features for modeling: {len(available_features)}")
        
        # Prepare features and target
        X = df[available_features].fillna(0)
        y = df['Fraud_Ind']
        
        print(f"📈 Dataset info:")
        print(f"   • Total samples: {len(X)}")
        print(f"   • Features: {len(available_features)}")
        print(f"   • Fraud cases: {y.sum()} ({(y.sum()/len(y)*100):.2f}%)")
        print(f"   • Non-fraud cases: {(y==0).sum()} ({((y==0).sum()/len(y)*100):.2f}%)")
        
        return X, y, available_features
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None, None, None

def initialize_fast_models():
    """Initialize fast models (excluding SVM)"""
    print("🤖 Initializing Fast ML Models (SVM excluded for speed)...")
    
    models = {}
    
    # 1. Logistic Regression
    models['Logistic_Regression'] = LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced'
    )
    print("✅ Logistic Regression")
    
    # 2. Random Forest
    models['Random_Forest'] = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
    print("✅ Random Forest")
    
    # 3. K-Nearest Neighbors
    models['K_Nearest_Neighbors'] = KNeighborsClassifier(n_neighbors=5)
    print("✅ K-Nearest Neighbors")
    
    # 4. Naive Bayes
    models['Naive_Bayes'] = GaussianNB()
    print("✅ Naive Bayes")
    
    # 5. Decision Tree
    models['Decision_Tree'] = DecisionTreeClassifier(
        random_state=42, class_weight='balanced', max_depth=10
    )
    print("✅ Decision Tree")
    
    # 6. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=42, eval_metric='logloss', n_jobs=-1,
            scale_pos_weight=3, n_estimators=100
        )
        print("✅ XGBoost")
    else:
        print("⚠️ XGBoost not available")
    
    # 7. SGD Classifier
    models['SGD_Classifier'] = SGDClassifier(
        random_state=42, class_weight='balanced', loss='log_loss'
    )
    print("✅ SGD Classifier")
    
    # 8. Gradient Boosting
    models['Gradient_Boosting'] = GradientBoostingClassifier(
        random_state=42, n_estimators=100
    )
    print("✅ Gradient Boosting")
    
    # 9. Voting Classifier (ensemble)
    voting_models = [
        ('rf', models['Random_Forest']),
        ('lr', models['Logistic_Regression']),
        ('dt', models['Decision_Tree'])
    ]
    
    models['Voting_Classifier'] = VotingClassifier(
        estimators=voting_models, voting='soft'
    )
    print("✅ Voting Classifier")
    
    print(f"\n🎯 Total models initialized: {len(models)} (SVM excluded for speed)")
    return models

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive evaluation metrics"""
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # ROC-AUC (if probabilities available)
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
    """Train and evaluate all models quickly"""
    print("\n🏋️ Training and Evaluating Models (Fast Mode)...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"   • Training: {X_train.shape} | Test: {X_test.shape}")
    print(f"   • Train fraud: {(y_train.sum()/len(y_train)*100):.2f}% | Test fraud: {(y_test.sum()/len(y_test)*100):.2f}%")
    
    models = initialize_fast_models()
    results = []
    
    total_start_time = time.time()
    
    for i, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] 🚀 Training {model_name}...")
        start_time = time.time()
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['model_name'] = model_name
            metrics['training_time'] = training_time
            
            # Quick cross-validation (2-fold for speed)
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=2, scoring='roc_auc')
                metrics['cv_roc_auc_mean'] = cv_scores.mean()
                metrics['cv_roc_auc_std'] = cv_scores.std()
            except:
                metrics['cv_roc_auc_mean'] = 0.5
                metrics['cv_roc_auc_std'] = 0.0
            
            results.append(metrics)
            
            # Print quick results
            print(f"   ✅ Done in {training_time:.2f}s | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    total_time = time.time() - total_start_time
    results_df = pd.DataFrame(results)
    
    print(f"\n✅ All models trained in {total_time:.2f}s! ({len(results_df)} successful)")
    return results_df

def create_quick_visualizations(results_df, output_dir):
    """Create essential visualizations quickly"""
    print("\n📊 Creating Visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    
    # 1. Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison (Fast Testing)', fontsize=16, fontweight='bold')
    
    # ROC-AUC
    axes[0, 0].bar(results_df['model_name'], results_df['roc_auc'], color='lightgreen')
    axes[0, 0].set_title('ROC-AUC Scores')
    axes[0, 0].set_ylabel('ROC-AUC')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # F1-Score
    axes[0, 1].bar(results_df['model_name'], results_df['f1_score'], color='orange')
    axes[0, 1].set_title('F1-Scores')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # Accuracy
    axes[1, 0].bar(results_df['model_name'], results_df['accuracy'], color='skyblue')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Training Time
    axes[1, 1].bar(results_df['model_name'], results_df['training_time'], color='salmon')
    axes[1, 1].set_title('Training Time')
    axes[1, 1].set_ylabel('Seconds')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fast_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top models ranking
    plt.figure(figsize=(12, 6))
    top_models = results_df.nlargest(5, 'roc_auc')
    
    x_pos = np.arange(len(top_models))
    plt.bar(x_pos - 0.2, top_models['roc_auc'], 0.4, label='ROC-AUC', color='gold')
    plt.bar(x_pos + 0.2, top_models['f1_score'], 0.4, label='F1-Score', color='lightblue')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Top 5 Models Performance', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, top_models['model_name'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_models_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations created")

def generate_fast_recommendations(results_df):
    """Generate quick model recommendations"""
    print("\n🎯 Generating Recommendations...")
    
    # Scalability scores (excluding SVM)
    scalability_scores = {
        'Logistic_Regression': 9,
        'Random_Forest': 7,
        'K_Nearest_Neighbors': 4,
        'Naive_Bayes': 9,
        'Decision_Tree': 6,
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
    
    # Get recommendations
    best_overall = results_df.loc[results_df['deployment_score'].idxmax()]
    best_roc_auc = results_df.loc[results_df['roc_auc'].idxmax()]
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    fastest = results_df.loc[results_df['training_time'].idxmin()]
    most_scalable = results_df.loc[results_df['scalability_score'].idxmax()]
    
    return {
        'best_overall': best_overall,
        'best_roc_auc': best_roc_auc,
        'best_f1': best_f1,
        'fastest': fastest,
        'most_scalable': most_scalable
    }

def print_fast_summary(results_df, recommendations):
    """Print quick results summary"""
    print_header("FAST TESTING RESULTS")
    
    # Sort by deployment score
    results_sorted = results_df.sort_values('deployment_score', ascending=False)
    
    print("🏆 TOP 5 MODELS:")
    print("-" * 50)
    for i, (idx, row) in enumerate(results_sorted.head(5).iterrows(), 1):
        print(f"{i}. {row['model_name']:<20} | Score: {row['deployment_score']:.4f}")
        print(f"   ROC-AUC: {row['roc_auc']:.4f} | F1: {row['f1_score']:.4f} | Time: {row['training_time']:.2f}s")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"🏆 Best Overall: {recommendations['best_overall']['model_name']}")
    print(f"📈 Best ROC-AUC: {recommendations['best_roc_auc']['model_name']} ({recommendations['best_roc_auc']['roc_auc']:.4f})")
    print(f"⚖️ Best F1-Score: {recommendations['best_f1']['model_name']} ({recommendations['best_f1']['f1_score']:.4f})")
    print(f"⚡ Fastest: {recommendations['fastest']['model_name']} ({recommendations['fastest']['training_time']:.2f}s)")
    print(f"🚀 Most Scalable: {recommendations['most_scalable']['model_name']} ({recommendations['most_scalable']['scalability_score']}/10)")

def main():
    """Main function for fast model testing"""
    print_header("FAST FRAUD DETECTION MODEL TESTING")
    print("⚡ SVM excluded for faster execution")
    print(f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}")
    
    output_dir = "/Users/debabratapattnayak/web-dev/learnathon/model-test/results"
    
    try:
        # Load data
        X, y, features = load_and_prepare_data()
        if X is None:
            return None, None
        
        # Train models
        results_df = train_and_evaluate_models(X, y)
        if results_df.empty:
            print("❌ No models trained successfully")
            return None, None
        
        # Create visualizations
        create_quick_visualizations(results_df, output_dir)
        
        # Generate recommendations
        recommendations = generate_fast_recommendations(results_df)
        
        # Print summary
        print_fast_summary(results_df, recommendations)
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results_df.to_csv(f"{output_dir}/fast_model_results.csv", index=False)
        
        print_header("FAST TESTING COMPLETED!")
        best = recommendations['best_overall']
        print(f"🏆 RECOMMENDED: {best['model_name']}")
        print(f"📊 ROC-AUC: {best['roc_auc']:.4f} | F1: {best['f1_score']:.4f}")
        print(f"🚀 Scalability: {best['scalability_score']}/10")
        print(f"📁 Results saved to: {output_dir}")
        
        return results_df, recommendations
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None

if __name__ == "__main__":
    results, recommendations = main()
