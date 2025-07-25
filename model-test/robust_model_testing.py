#!/usr/bin/env python3
"""
Robust Fraud Detection Model Testing
===================================
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
import sys

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, matthews_corrcoef, 
    balanced_accuracy_score, log_loss
)

# Model imports
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def load_and_prepare_data():
    """Load and prepare the processed data for modeling"""
    print_step(1, "Loading and Preparing Data")
    
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
                print(f"✅ Added normalized: {feat}_normalized")
            elif feat in df.columns:
                available_features.append(feat)
                print(f"✅ Added original: {feat}")
            else:
                print(f"⚠️ Feature not found: {feat}")
        
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
                print(f"✅ Added engineered: {feat}")
        
        print(f"\n📊 Total features for modeling: {len(available_features)}")
        
        # Prepare features and target
        X = df[available_features].fillna(0)  # Handle any remaining NaN
        y = df['Fraud_Ind']
        
        # Basic data info
        print(f"📈 Dataset info:")
        print(f"   • Total samples: {len(X)}")
        print(f"   • Features: {len(available_features)}")
        print(f"   • Fraud cases: {y.sum()} ({(y.sum()/len(y)*100):.2f}%)")
        print(f"   • Non-fraud cases: {(y==0).sum()} ({((y==0).sum()/len(y)*100):.2f}%)")
        
        return X, y, available_features
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None, None, None

def initialize_models():
    """Initialize all models for testing"""
    print_step(2, "Initializing ML Models")
    
    models = {}
    
    # 1. Logistic Regression
    models['Logistic_Regression'] = LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced'
    )
    print("✅ Logistic Regression initialized")
    
    # 2. Random Forest
    models['Random_Forest'] = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
    print("✅ Random Forest initialized")
    
    # 3. K-Nearest Neighbors
    models['K_Nearest_Neighbors'] = KNeighborsClassifier(n_neighbors=5)
    print("✅ K-Nearest Neighbors initialized")
    
    # 4. Naive Bayes
    models['Naive_Bayes'] = GaussianNB()
    print("✅ Naive Bayes initialized")
    
    # 5. Decision Tree
    models['Decision_Tree'] = DecisionTreeClassifier(
        random_state=42, class_weight='balanced', max_depth=10
    )
    print("✅ Decision Tree initialized")
    
    # 6. Support Vector Machine
    models['Support_Vector_Machine'] = SVC(
        random_state=42, class_weight='balanced', probability=True, kernel='rbf'
    )
    print("✅ Support Vector Machine initialized")
    
    # 7. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=42, eval_metric='logloss', n_jobs=-1,
            scale_pos_weight=3  # Handle class imbalance
        )
        print("✅ XGBoost initialized")
    else:
        print("⚠️ XGBoost not available, skipping")
    
    # 8. SGD Classifier
    models['SGD_Classifier'] = SGDClassifier(
        random_state=42, class_weight='balanced', loss='log_loss'
    )
    print("✅ SGD Classifier initialized")
    
    # 9. Gradient Boosting
    models['Gradient_Boosting'] = GradientBoostingClassifier(
        random_state=42, n_estimators=100
    )
    print("✅ Gradient Boosting initialized")
    
    # 10. Voting Classifier (ensemble)
    voting_models = [
        ('rf', models['Random_Forest']),
        ('lr', models['Logistic_Regression']),
        ('dt', models['Decision_Tree'])
    ]
    
    models['Voting_Classifier'] = VotingClassifier(
        estimators=voting_models, voting='soft'
    )
    print("✅ Voting Classifier initialized")
    
    print(f"\n🎯 Total models initialized: {len(models)}")
    return models

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
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
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except Exception as e:
            metrics['roc_auc'] = 0.5
            metrics['log_loss'] = np.inf
    else:
        metrics['roc_auc'] = 0.5
        metrics['log_loss'] = np.inf
    
    # Fraud-specific metrics (binary classification)
    metrics['precision_fraud'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_fraud'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['f1_fraud'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    return metrics

def train_and_evaluate_models(X, y):
    """Train and evaluate all models"""
    print_step(3, "Training and Evaluating Models")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"   • Training set: {X_train.shape}")
    print(f"   • Test set: {X_test.shape}")
    print(f"   • Train fraud rate: {(y_train.sum()/len(y_train)*100):.2f}%")
    print(f"   • Test fraud rate: {(y_test.sum()/len(y_test)*100):.2f}%")
    
    models = initialize_models()
    results = []
    
    print(f"\n🏋️ Starting model training...")
    
    for i, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] Training {model_name}...")
        start_time = time.time()
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            metrics['model_name'] = model_name
            metrics['training_time'] = training_time
            
            # Cross-validation (3-fold for speed)
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=3, scoring='roc_auc'
                )
                metrics['cv_roc_auc_mean'] = cv_scores.mean()
                metrics['cv_roc_auc_std'] = cv_scores.std()
            except:
                metrics['cv_roc_auc_mean'] = 0.5
                metrics['cv_roc_auc_std'] = 0.0
            
            results.append(metrics)
            
            # Print results
            print(f"   ✅ Completed in {training_time:.2f}s")
            print(f"      Accuracy: {metrics['accuracy']:.4f}")
            print(f"      ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"      F1-Score: {metrics['f1_score']:.4f}")
            print(f"      Fraud Recall: {metrics['recall_fraud']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    print(f"\n✅ Model training completed! {len(results_df)} models trained successfully.")
    
    return results_df

def create_visualizations(results_df, output_dir):
    """Create comprehensive visualizations"""
    print_step(4, "Creating Visualizations")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy
    axes[0, 0].bar(results_df['model_name'], results_df['accuracy'], color='skyblue')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # ROC-AUC
    axes[0, 1].bar(results_df['model_name'], results_df['roc_auc'], color='lightgreen')
    axes[0, 1].set_title('ROC-AUC Comparison')
    axes[0, 1].set_ylabel('ROC-AUC Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # F1-Score
    axes[1, 0].bar(results_df['model_name'], results_df['f1_score'], color='orange')
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Training Time
    axes[1, 1].bar(results_df['model_name'], results_df['training_time'], color='salmon')
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Performance comparison chart saved")
    
    # 2. Comprehensive Metrics Heatmap
    plt.figure(figsize=(14, 8))
    
    metrics_cols = [
        'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
        'balanced_accuracy', 'matthews_corrcoef'
    ]
    
    heatmap_data = results_df.set_index('model_name')[metrics_cols]
    
    sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlGn', fmt='.3f', 
               cbar_kws={'label': 'Score'})
    plt.title('Model Performance Metrics Heatmap', fontsize=16, fontweight='bold')
    plt.ylabel('Metrics')
    plt.xlabel('Models')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Metrics heatmap saved")
    
    # 3. Top 5 Models Ranking
    plt.figure(figsize=(12, 8))
    
    top_5 = results_df.nlargest(5, 'roc_auc')
    
    x_pos = np.arange(len(top_5))
    plt.bar(x_pos, top_5['roc_auc'], color='gold', alpha=0.7, label='ROC-AUC')
    plt.bar(x_pos, top_5['f1_score'], color='lightblue', alpha=0.7, label='F1-Score')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Top 5 Models - ROC-AUC vs F1-Score', fontsize=16, fontweight='bold')
    plt.xticks(x_pos, top_5['model_name'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'top5_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Top 5 models comparison saved")
    
    print(f"📁 All visualizations saved to: {output_dir}")

def generate_recommendations(results_df):
    """Generate comprehensive model recommendations"""
    print_step(5, "Generating Model Recommendations")
    
    # Scalability scores (1-10 scale)
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
    
    # Interpretability scores (1-10 scale)
    interpretability_scores = {
        'Logistic_Regression': 9,
        'Random_Forest': 7,
        'K_Nearest_Neighbors': 6,
        'Naive_Bayes': 8,
        'Decision_Tree': 10,
        'Support_Vector_Machine': 4,
        'XGBoost': 6,
        'SGD_Classifier': 8,
        'Gradient_Boosting': 5,
        'Voting_Classifier': 4
    }
    
    # Add scores to dataframe
    results_df['scalability_score'] = results_df['model_name'].map(scalability_scores).fillna(5)
    results_df['interpretability_score'] = results_df['model_name'].map(interpretability_scores).fillna(5)
    
    # Calculate composite scores
    results_df['performance_score'] = (
        0.4 * results_df['roc_auc'] +
        0.3 * results_df['f1_score'] +
        0.2 * results_df['balanced_accuracy'] +
        0.1 * results_df['recall_fraud']
    )
    
    results_df['deployment_score'] = (
        0.5 * results_df['performance_score'] +
        0.3 * (results_df['scalability_score'] / 10) +
        0.2 * (results_df['interpretability_score'] / 10)
    )
    
    # Generate recommendations
    recommendations = {
        'best_overall': results_df.loc[results_df['deployment_score'].idxmax()],
        'best_performance': results_df.loc[results_df['performance_score'].idxmax()],
        'best_roc_auc': results_df.loc[results_df['roc_auc'].idxmax()],
        'best_f1': results_df.loc[results_df['f1_score'].idxmax()],
        'best_fraud_detection': results_df.loc[results_df['recall_fraud'].idxmax()],
        'fastest_training': results_df.loc[results_df['training_time'].idxmin()],
        'most_scalable': results_df.loc[results_df['scalability_score'].idxmax()],
        'most_interpretable': results_df.loc[results_df['interpretability_score'].idxmax()]
    }
    
    return recommendations, results_df

def print_results_summary(results_df, recommendations):
    """Print comprehensive results summary"""
    print_step(6, "Results Summary and Recommendations")
    
    # Sort by deployment score
    results_sorted = results_df.sort_values('deployment_score', ascending=False)
    
    print("🏆 TOP 5 MODELS BY DEPLOYMENT SCORE:")
    print("-" * 50)
    for i, (idx, row) in enumerate(results_sorted.head(5).iterrows(), 1):
        print(f"{i}. {row['model_name']:<20}")
        print(f"   Deployment Score: {row['deployment_score']:.4f}")
        print(f"   ROC-AUC: {row['roc_auc']:.4f} | F1: {row['f1_score']:.4f}")
        print(f"   Scalability: {row['scalability_score']}/10 | Interpretability: {row['interpretability_score']}/10")
        print(f"   Training Time: {row['training_time']:.2f}s")
        print()
    
    print("\n🎯 CATEGORY WINNERS:")
    print("-" * 30)
    print(f"🏆 Best Overall: {recommendations['best_overall']['model_name']} (Score: {recommendations['best_overall']['deployment_score']:.4f})")
    print(f"📈 Best ROC-AUC: {recommendations['best_roc_auc']['model_name']} ({recommendations['best_roc_auc']['roc_auc']:.4f})")
    print(f"⚖️ Best F1-Score: {recommendations['best_f1']['model_name']} ({recommendations['best_f1']['f1_score']:.4f})")
    print(f"🔍 Best Fraud Detection: {recommendations['best_fraud_detection']['model_name']} ({recommendations['best_fraud_detection']['recall_fraud']:.4f})")
    print(f"⚡ Fastest Training: {recommendations['fastest_training']['model_name']} ({recommendations['fastest_training']['training_time']:.2f}s)")
    print(f"🚀 Most Scalable: {recommendations['most_scalable']['model_name']} ({recommendations['most_scalable']['scalability_score']}/10)")
    print(f"🔬 Most Interpretable: {recommendations['most_interpretable']['model_name']} ({recommendations['most_interpretable']['interpretability_score']}/10)")

def save_results(results_df, output_dir):
    """Save all results to files"""
    print_step(7, "Saving Results")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    results_df.to_csv(output_dir / 'comprehensive_model_results.csv', index=False)
    print("✅ Model results saved to CSV")
    
    # Save deployment ranking
    deployment_ranking = results_df.sort_values('deployment_score', ascending=False)
    deployment_ranking.to_csv(output_dir / 'deployment_ranking.csv', index=False)
    print("✅ Deployment ranking saved")
    
    # Create summary report
    with open(output_dir / 'model_testing_summary.txt', 'w') as f:
        f.write("FRAUD DETECTION MODEL TESTING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Testing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Tested: {len(results_df)}\n\n")
        
        f.write("TOP 5 MODELS BY DEPLOYMENT SCORE:\n")
        f.write("-" * 40 + "\n")
        for i, (idx, row) in enumerate(deployment_ranking.head(5).iterrows(), 1):
            f.write(f"{i}. {row['model_name']}\n")
            f.write(f"   Deployment Score: {row['deployment_score']:.4f}\n")
            f.write(f"   ROC-AUC: {row['roc_auc']:.4f}\n")
            f.write(f"   F1-Score: {row['f1_score']:.4f}\n")
            f.write(f"   Scalability: {row['scalability_score']}/10\n")
            f.write(f"   Training Time: {row['training_time']:.2f}s\n\n")
    
    print("✅ Summary report saved")
    print(f"📁 All results saved to: {output_dir}")

def main():
    """Main function to run the complete model testing pipeline"""
    print_header("FRAUD DETECTION MODEL TESTING PIPELINE")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set output directory
    output_dir = "/Users/debabratapattnayak/web-dev/learnathon/model-test/results"
    
    try:
        # Step 1: Load data
        X, y, features = load_and_prepare_data()
        if X is None:
            return None, None
        
        # Step 2 & 3: Train and evaluate models
        results_df = train_and_evaluate_models(X, y)
        if results_df.empty:
            print("❌ No models were trained successfully")
            return None, None
        
        # Step 4: Create visualizations
        create_visualizations(results_df, output_dir)
        
        # Step 5: Generate recommendations
        recommendations, results_df = generate_recommendations(results_df)
        
        # Step 6: Print summary
        print_results_summary(results_df, recommendations)
        
        # Step 7: Save results
        save_results(results_df, output_dir)
        
        # Final summary
        print_header("TESTING COMPLETED SUCCESSFULLY!")
        best_model = recommendations['best_overall']
        print(f"🏆 RECOMMENDED MODEL: {best_model['model_name']}")
        print(f"📊 Deployment Score: {best_model['deployment_score']:.4f}")
        print(f"📈 ROC-AUC: {best_model['roc_auc']:.4f}")
        print(f"⚖️ F1-Score: {best_model['f1_score']:.4f}")
        print(f"🚀 Scalability: {best_model['scalability_score']}/10")
        print(f"🔬 Interpretability: {best_model['interpretability_score']}/10")
        print(f"\n✅ Ready for Streamlit application development!")
        
        return results_df, recommendations
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, recommendations = main()
