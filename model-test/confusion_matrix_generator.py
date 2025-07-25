#!/usr/bin/env python3
"""
Confusion Matrix Generator for All Fraud Detection Models
========================================================
Generates confusion matrices for all 9 trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import warnings
from pathlib import Path
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare data for confusion matrix generation"""
    print("📊 Loading data for confusion matrix generation...")
    
    data_path = "/Users/debabratapattnayak/web-dev/learnathon/ml_analysis_reports/updated_2025-07-25_23-19-01/updated_processed_training_data.csv"
    df = pd.read_csv(data_path)
    
    # Define features (same as used in model testing)
    requested_features = ['Annual_Mileage', 'DiffIN_Mileage', 'Auto_Make', 'Vehicle_Cost']
    other_features = [
        'Accident_Severity', 'Garage_Location', 'Collision_Type',
        'authorities_contacted', 'Commute_Discount', 'Witnesses',
        'Umbrella_Limit', 'Policy_State', 'Num_of_Vehicles_Involved',
        'Acccident_State'
    ]
    
    # Get available features
    available_features = []
    for feat in requested_features + other_features:
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
    
    X = df[available_features].fillna(0)
    y = df['Fraud_Ind']
    
    print(f"✅ Data loaded: {X.shape}, Features: {len(available_features)}")
    return X, y

def initialize_models():
    """Initialize all models for confusion matrix generation"""
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        'K_Nearest_Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive_Bayes': GaussianNB(),
        'Decision_Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10),
        'SGD_Classifier': SGDClassifier(random_state=42, class_weight='balanced', loss='log_loss'),
        'Gradient_Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, scale_pos_weight=3)
    
    # Voting Classifier
    voting_models = [
        ('rf', models['Random_Forest']),
        ('lr', models['Logistic_Regression']),
        ('dt', models['Decision_Tree'])
    ]
    models['Voting_Classifier'] = VotingClassifier(estimators=voting_models, voting='soft')
    
    return models

def create_confusion_matrix_plot(y_true, y_pred, model_name, ax):
    """Create a single confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
        annotations.append(row)
    
    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'],
                cbar_kws={'label': 'Count'})
    
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Calculate and display key metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add metrics text
    metrics_text = f'Acc: {accuracy:.3f}\nPrec: {precision:.3f}\nRec: {recall:.3f}\nF1: {f1:.3f}'
    ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def generate_all_confusion_matrices(X, y, models, output_dir):
    """Generate confusion matrices for all models"""
    print("🔄 Training models and generating confusion matrices...")
    
    # Split data (same split as used in model testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate grid size
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Create figure for all confusion matrices
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    model_results = []
    
    for idx, (model_name, model) in enumerate(models.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        print(f"📈 Processing {model_name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Create confusion matrix plot
            create_confusion_matrix_plot(y_test, y_pred, model_name, ax)
            
            # Store results
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            model_results.append({
                'model': model_name,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp,
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            })
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
            ax.text(0.5, 0.5, f'Error: {model_name}\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{model_name} - Error')
    
    # Hide empty subplots
    for idx in range(len(models), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_models_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model_results

def create_confusion_matrix_summary(model_results, output_dir):
    """Create summary table and individual detailed matrices"""
    print("📊 Creating confusion matrix summary...")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(model_results)
    summary_df['f1_score'] = 2 * (summary_df['precision'] * summary_df['recall']) / (summary_df['precision'] + summary_df['recall'])
    summary_df['f1_score'] = summary_df['f1_score'].fillna(0)
    
    # Sort by F1 score
    summary_df = summary_df.sort_values('f1_score', ascending=False)
    
    # Save summary
    summary_df.to_csv(output_dir / 'confusion_matrix_summary.csv', index=False)
    
    # Create detailed summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy comparison
    axes[0, 0].bar(summary_df['model'], summary_df['accuracy'], color='skyblue')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Precision vs Recall
    axes[0, 1].scatter(summary_df['recall'], summary_df['precision'], s=100, alpha=0.7)
    for i, model in enumerate(summary_df['model']):
        axes[0, 1].annotate(model, (summary_df.iloc[i]['recall'], summary_df.iloc[i]['precision']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_xlabel('Recall (Sensitivity)')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Recall')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. F1 Score comparison
    axes[1, 0].bar(summary_df['model'], summary_df['f1_score'], color='lightgreen')
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Confusion Matrix Components
    x = np.arange(len(summary_df))
    width = 0.2
    
    axes[1, 1].bar(x - 1.5*width, summary_df['true_positives'], width, label='True Positives', color='green', alpha=0.7)
    axes[1, 1].bar(x - 0.5*width, summary_df['true_negatives'], width, label='True Negatives', color='blue', alpha=0.7)
    axes[1, 1].bar(x + 0.5*width, summary_df['false_positives'], width, label='False Positives', color='orange', alpha=0.7)
    axes[1, 1].bar(x + 1.5*width, summary_df['false_negatives'], width, label='False Negatives', color='red', alpha=0.7)
    
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Confusion Matrix Components')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(summary_df['model'], rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return summary_df

def print_detailed_results(summary_df):
    """Print detailed confusion matrix results"""
    print("\n" + "="*80)
    print("🎯 CONFUSION MATRIX ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n🏆 TOP 5 MODELS BY F1-SCORE:")
    print("-" * 50)
    
    for i, (idx, row) in enumerate(summary_df.head(5).iterrows(), 1):
        print(f"\n{i}. {row['model']}")
        print(f"   📊 Confusion Matrix:")
        print(f"      True Positives:  {row['true_positives']:,}")
        print(f"      True Negatives:  {row['true_negatives']:,}")
        print(f"      False Positives: {row['false_positives']:,}")
        print(f"      False Negatives: {row['false_negatives']:,}")
        print(f"   📈 Metrics:")
        print(f"      Accuracy:   {row['accuracy']:.4f}")
        print(f"      Precision:  {row['precision']:.4f}")
        print(f"      Recall:     {row['recall']:.4f}")
        print(f"      F1-Score:   {row['f1_score']:.4f}")
        print(f"      Specificity: {row['specificity']:.4f}")
    
    print(f"\n📋 SUMMARY TABLE:")
    print("-" * 30)
    display_cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score']
    print(summary_df[display_cols].round(4).to_string(index=False))

def main():
    """Main function to generate all confusion matrices"""
    print("🎯 CONFUSION MATRIX GENERATOR FOR ALL MODELS")
    print("=" * 60)
    
    # Set output directory
    output_dir = Path("/Users/debabratapattnayak/web-dev/learnathon/model-test/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        X, y = load_data()
        
        # Initialize models
        models = initialize_models()
        print(f"✅ Initialized {len(models)} models")
        
        # Generate confusion matrices
        model_results = generate_all_confusion_matrices(X, y, models, output_dir)
        
        # Create summary analysis
        summary_df = create_confusion_matrix_summary(model_results, output_dir)
        
        # Print results
        print_detailed_results(summary_df)
        
        print(f"\n📁 Results saved to: {output_dir}")
        print("📄 Files created:")
        print("   • all_models_confusion_matrices.png")
        print("   • confusion_matrix_analysis.png")
        print("   • confusion_matrix_summary.csv")
        
        print(f"\n🎉 Confusion matrix analysis completed successfully!")
        
        return summary_df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
