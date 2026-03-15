# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

import shap
from shap import plots
import lime
import lime.lime_tabular

# XAI Quality Assessment Integration
from evaluation_toolkit import SimplifiedXAIEvaluator

# LLM Explanation Integration
from llm_explanation_module import DeepseekLLMExplainer

# Scikit-learn imports
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Other ML library imports
import xgboost as xgb

# data preprocessing


# Helper function to create organized directory structure
def create_model_result_dir(task_type: str, model_name: str) -> str:
    """
    Create organized directory structure for model results.
    
    Args:
        task_type: 'binary' or 'multiclass'
        model_name: Name of the model (e.g., 'DecisionTree', 'RandomForest', etc.)
    
    Returns:
        str: Path to the model-specific directory
    """
    model_dir = os.path.join(RESULT_DIR, task_type, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

# Evaluation sampling configuration (configurable via environment variables)
EVAL_SAMPLE_SIZE = int(os.environ.get('XAI_EVAL_SAMPLE_SIZE', '200'))
EVAL_STRATIFIED = os.environ.get('XAI_EVAL_STRATIFIED', '1') == '1'
cwd = os.getcwd()

# ---- helper: save classification report as image ----
def save_classification_report_image(y_true, y_pred, title, save_path, target_names=None):
    try:
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=target_names if target_names is not None else None,
            output_dict=True,
            zero_division=0
        )
        df_report = pd.DataFrame(report_dict).T
        # format support and metrics
        if 'support' in df_report.columns:
            df_report['support'] = df_report['support'].fillna(0).astype(int)
        for col in ['precision', 'recall', 'f1-score']:
            if col in df_report.columns:
                df_report[col] = df_report[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

        df_display = df_report.reset_index().rename(columns={'index': 'class'})

        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(df_display) + 2)))
        ax.axis('off')
        table = ax.table(cellText=df_display.values,
                         colLabels=df_display.columns,
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as _e:
        # Fallback: save plain text report as image
        try:
            report_text = classification_report(
                y_true,
                y_pred,
                target_names=target_names if target_names is not None else None,
                zero_division=0
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')
            ax.text(0.01, 0.99, report_text, fontsize=10, family='monospace', va='top')
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            pass


# def data_preprocess(data_name):
#     if data_name=='5G-NIDD':
#         df = pd.read_csv(f'{cwd}/{data_name}.csv', encoding='utf-8')
#         # handle NaN
#         MIN_NON_NULL=1215000
#         df_cleaned = df.dropna(axis=1, thresh=MIN_NON_NULL)

#         # high_risk_features = [
#         #     'sTos', 'sDSb', 'sVid', 'dVid',     # Your initial list
#         #     'dTtl', 'dHops', 'DstTCPBase',      # The new audited ones
#         #     'sHops', 'SrcTCPBase'               # Usually linked to the ones above
#         # ]

#         # merge some categories
#         df_cleaned['Proto'] = np.where(df_cleaned['Proto'].isin(['lldp','llc','arp','ipv6-icmp']), 'other', df_cleaned['Proto'])
#         df_cleaned['State'] = np.where(df_cleaned['State'].isin(['ECO','ACC','URP','RSP','TST','NRS']), 'other', df_cleaned['State'])

#         proto_dummies = pd.get_dummies(df_cleaned['Proto'], prefix='Proto')
#         cause_dummies = pd.get_dummies(df_cleaned['Cause'], prefix='Cause')
#         state_dummies = pd.get_dummies(df_cleaned['State'], prefix='State')
#         df_new = pd.concat([df_cleaned, proto_dummies, cause_dummies, state_dummies], axis=1)
#         if 'Seq' in df_new.columns:
#             df_new.drop(columns=['Seq'], inplace=True) # new
#         if 'Offset' in df_new.columns:
#             df_new.drop(columns=['Offset'], inplace=True) # new
#         cols_to_drop = [c for c in ['Proto', 'Cause', 'State', 'sDSb'] if c in df_new.columns]
#         if cols_to_drop:
#             df_new.drop(cols_to_drop, axis=1, inplace=True)
#         df_new=df_new.dropna(axis=0)

#         features=df_new.columns.tolist()
#         for _c in ['Label', 'Attack Type', 'Attack Tool', 'Unnamed: 0']:
#             if _c in features:
#                 features.remove(_c)
#         print(features)
#     elif data_name=='5GC_PFCP':
#         df_new = pd.read_csv(f'{cwd}/5GC_PFCP.csv', encoding='utf-8')
#         df_new['Attack Type'] = df_new['Label'].copy() 
#         df_new['Label'] = np.where(df_new['Label'] == 'Normal', 'Benign', 'Malicious')
#         features=df_new.columns.tolist()
#         # Remove non-numeric columns that shouldn't be used as features
#         for _c in ['Label', 'Attack Type']:
#             if _c in features:
#                 features.remove(_c)
#         print(features)
#     elif data_name=='5GAD':
#         df = pd.read_csv(f'{cwd}/5GAD.csv', encoding='utf-8')
#         df_new=df.drop(['src_ip', 'dst_ip'], axis=1)
#         df_new=df_new.rename(columns={'attack_type':'Attack Type'})
    
#         # Convert tcp_flags to numeric
#         if 'tcp_flags' in df_new.columns:
#             # Create mapping for tcp_flags
#             tcp_flags_mapping = {
#                 '0': 0,
#                 'A': 1,    # ACK
#                 'PA': 2,   # PSH + ACK
#                 'FPA': 3   # FIN + PSH + ACK
#             }
#             df_new['tcp_flags'] = df_new['tcp_flags'].map(tcp_flags_mapping)
#             # Fill any unmapped values with 0
#             df_new['tcp_flags'] = df_new['tcp_flags'].fillna(0)
        
#         # Set up features and labels
#         features = df_new.columns.tolist()
#         # Remove label columns from features
#         for _c in ['label', 'Attack Type']:
#             if _c in features:
#                 features.remove(_c)
#         print(features)
#         # Create binary label: normal=0, attack=1
#         df_new['Label'] = df_new['label'].map({'normal': 'Benign', 'attack': 'Malicious'})
#         df_new=df_new.drop(['label'], axis=1)
#     else:
#         print(f"Data name {data_name} not found")
#         return
#     X = df_new[features]
#     le = LabelEncoder()
#     y = df_new['Label']
#     y = le.fit_transform(y)
#     # Print the encoded labels
#     print("Encoded Labels:", y)
#     return X, y, le,df_new


## NOTE: pipeline entrypoint moved to the end (__main__) for configurable datasets

def train_lightgbm_with_scale_pos_weight(X_train, y_train, X_val, y_val):
    """
    Trains a LightGBM model for binary classification using scale_pos_weight
    to handle class imbalance.
    """
    # 1. Divide the dataset
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )
   
    # 2. Calculate scale_pos_weight for binary classification
    class_counts = np.bincount(y_train)
    scale_pos_weight = class_counts[0] / class_counts[1]

    print(f"Class distribution in training set: Class 0: {class_counts[0]} samples, Class 1: {class_counts[1]} samples")
    print(f"Computed scale_pos_weight: {scale_pos_weight:.4f}")

    # 3. Build LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    # test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    # 4. Set parameters
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'num_leaves': 30,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    # Try GPU first, fallback to CPU if not available
    try:
        params_gpu = dict(params)
        params_gpu.update(tree_learner='serial', device_type='gpu')
        model = lgb.train(
            params_gpu,
            train_data,
            num_boost_round=500,
            # valid_sets=[train_data, test_data],
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )
    except Exception:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            # valid_sets=[train_data, test_data],
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )

    # 5. Train the model
    # model already trained above with GPU-first strategy

    # 6. Make predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 7. Evaluate the model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

    # Create organized directory for LightGBM binary results
    lightgbm_dir = create_model_result_dir('binary', 'LightGBM')
    
    # Save classification report as image
    save_classification_report_image(
        y_true=y_test,
        y_pred=y_pred,
        title='Classification Report (LightGBM Binary)',
        save_path=os.path.join(lightgbm_dir, "classification_report.png"),
        target_names=['Benign', 'Malicious']
    )

    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Binary LightGBM Confusion Matrix')
    plt.savefig(os.path.join(lightgbm_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # return model, X_train, X_test, y_train, y_test
    return model


def shap_plots_Tree(model, X_train, feature_names=None, top_n=30):
    """
    Enhanced version of SHAP visualization that:
    1. Outputs top N important features list.
    2. Fixes compatibility issues with TreeExplainer.
    3. Handles both binary and multi-class scenarios.
    """
    # Initialize explainer - CORRECTED LINE
    # Only pass the model to the explainer.
    # The data can be passed later when calculating shap_values.
    explainer = shap.TreeExplainer(model)


    # Calculate SHAP values for the test data
    shap_values = explainer.shap_values(X_train)

    # The rest of your logic for handling binary/multi-class outputs and plotting is correct.
    # We will use the positive class's SHAP values for summary plots in a binary case.
    if isinstance(shap_values, list):
        # For binary classification, shap_values is a list [class_0_values, class_1_values]
        shap_values_for_summary = shap_values[1]
    else:
        # For regression or single-output models
        shap_values_for_summary = shap_values

    # ========================
    # 1. Feature Importance List (Top N)
    # ========================
    abs_shap = np.abs(shap_values_for_summary).mean(0)

    # Create feature importance DataFrame
    if feature_names is None:
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    n_features = len(feature_names)

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': abs_shap
    })

    # Sort full importance
    importance_sorted = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    # Save all features global importance
    shap_dir = create_model_result_dir('binary', 'LightGBM')
    try:
        with open(os.path.join(shap_dir, "shap_top_features.json"), 'w', encoding='utf-8') as f:
            json.dump([
                {"feature": str(row.Feature), "importance": float(row.Importance)}
                for _, row in importance_sorted.iterrows()
            ], f, ensure_ascii=False, indent=2)
    except Exception as _e:
        print(f"Failed to write shap_top_features.json: {_e}")

    # Compute cumulative importance threshold at 90%
    importance_vals = importance_sorted['Importance'].values
    total = importance_vals.sum() or 1.0
    cum = np.cumsum(importance_vals) / total
    cutoff_idx = int(np.searchsorted(cum, 0.99) + 1)
    print(f"Cumulative importance threshold at 99: {cutoff_idx}")
    top_n_features = max(cutoff_idx, int(0.8 * n_features+1))
    top_n_features_df = importance_sorted.head(top_n_features)
    # Also save cum90 selection
    try:
        with open(os.path.join(shap_dir, "shap_top_feature.json"), 'w', encoding='utf-8') as f:
            json.dump([
                {"feature": str(row.Feature), "importance": float(row.Importance)}
                for _, row in top_n_features_df.iterrows()
            ], f, ensure_ascii=False, indent=2)
    except Exception as _e:
        print(f"Failed to write shap_top_feature.json: {_e}")

    # Derive display top-N from sorted data (for plots)
    available_features = len(importance_sorted)
    actual_top_n = min(top_n, available_features)
    top_n_df = importance_sorted.head(actual_top_n)

    print("="*80)
    print(f"Top {top_n} Features by SHAP Importance")
    print("="*80)
    print(top_n_df.to_string(index=False))
    print("\n")

    # ========================
    # 2. SHAP Summary Plots
    # ========================
    print("Generating SHAP summary plots...")
    # Create organized directory for SHAP results (already created above)
    
    # Bar plot for feature importance
    plt.figure()
    shap.summary_plot(shap_values_for_summary, X_train, plot_type="bar",
                      feature_names=feature_names, show=False, max_display=top_n)
    plt.title("SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_importance_bar.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Summary plot (beeswarm) for SHAP value distribution
    plt.figure()
    shap.summary_plot(shap_values_for_summary, X_train,
                      feature_names=feature_names, show=False, max_display=top_n)
    plt.title("SHAP Value Distribution (Beeswarm Plot)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_beeswarm.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # ========================
    # 3. Return results for further analysis
    # ========================
    return explainer, shap_values, top_n_df, top_n_features_df

# ---------------- XAI Assessment Helper Tools (No Impact on Original Functionality) ----------------
def _build_explanations_from_shap(shap_values_array, top_k=10, max_samples=1000):
    """
    Construct explanation structure required for assessment based on SHAP values.
    - Include ALL features in explanations for comprehensive evaluation
    - Use feature indices (int) as keys to ensure deletion/insertion tests in assessment are executable.
    - Limit sample count to improve assessment efficiency.
    """
    explanations = []
    if shap_values_array is None:
        return explanations
    try:
        num_samples = min(max_samples, shap_values_array.shape[0])
        for i in range(num_samples):
            shap_vals_row = shap_values_array[i]
            # Compatible with possible 2D row vectors
            if hasattr(shap_vals_row, 'ndim') and shap_vals_row.ndim > 1:
                shap_vals_row = np.asarray(shap_vals_row).reshape(-1)
            
            # Include all features, not just top-k features
            feature_importance = {}
            for idx in range(len(shap_vals_row)):
                try:
                    value = shap_vals_row[idx]
                    if hasattr(value, 'item'):
                        value = value.item()  # Convert numpy scalar to Python scalar
                    feature_importance[int(idx)] = float(value)
                except (ValueError, TypeError):
                    feature_importance[int(idx)] = 0.0
            
            explanations.append({
                'feature_importance': feature_importance,
                'text_explanation': f'SHAP-based explanation for sample {i}',
                'rules': [],
                'confidence': 0.0,
                'prediction_probability': 0.0,
                'comparison_baseline': 'zero_baseline'
            })
    except Exception:
        pass
    return explanations


class _PredictProbaAdapter:
    """
    Provide unified probability output interface for models without predict_proba (e.g., Keras).
    Optionally apply scaler internally to match the input space during model training.
    """
    def __init__(self, base_model, scaler=None, uses_scaled=False):
        self.base_model = base_model
        self.scaler = scaler
        self.uses_scaled = uses_scaled

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        if self.uses_scaled and self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)
        if hasattr(self.base_model, 'predict_proba'):
            probs = self.base_model.predict_proba(X_arr)
        else:
            # Keras and other models
            probs = self.base_model.predict(X_arr, verbose=0)
        probs = np.asarray(probs)
        
        # Fix: Unified handling of output shapes
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        
        # Handle binary classification single column output (sigmoid)
        if probs.shape[1] == 1:
            # Convert single column to two columns: [P(class=0), P(class=1)]
            probs = np.column_stack([1 - probs[:, 0], probs[:, 0]])
        
        return probs

    def predict(self, X):
        """Add predict method to support LIME explainer"""
        X_arr = np.asarray(X)
        if self.uses_scaled and self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)
        if hasattr(self.base_model, 'predict'):
            return self.base_model.predict(X_arr, verbose=0)
        else:
            # Derive predict from predict_proba
            probs = self.predict_proba(X_arr)
            return np.argmax(probs, axis=1)


def _select_sample_indices(y_array, available_n, desired_n, stratified=True):
    """
    Choose sample indices with optional stratified sampling.
    """
    desired = min(desired_n, available_n)
    if desired <= 0:
        return np.array([], dtype=int)
    indices = np.arange(available_n)
    try:
        if stratified and y_array is not None:
            from sklearn.model_selection import StratifiedShuffleSplit
            y_array = np.asarray(y_array)[:available_n]
            if len(np.unique(y_array)) > 1 and available_n >= desired and desired >= len(np.unique(y_array)):
                sss = StratifiedShuffleSplit(n_splits=1, test_size=desired, random_state=42)
                for _, test_idx in sss.split(indices, y_array):
                    return np.sort(test_idx)
    except Exception:
        pass
    # Fallback: uniform random
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(indices, size=desired, replace=False))


def _evaluate_and_save(model_name,
                       model,
                       X_test_df,
                       y_test,
                       shap_values_for_plot,
                       scaler=None,
                       uses_scaled=False,
                       class_names=None,
                       out_dir=None,
                       explanations=None,
                       method_name: str = "shap_local",
                       task_type: str = "binary",
                       explainer=None,
                       global_importance=None):
    """
    Evaluate model using simplified evaluation framework and save results
    Supports both local and global explanation evaluation
    """
    # Determine output directory
    if out_dir is None:
        model_dir = create_model_result_dir(task_type, model_name)
        out_dir = os.path.join(model_dir, "evaluate")
    
    # Method-specific directory
    model_eval_dir = os.path.join(out_dir, method_name)
    os.makedirs(model_eval_dir, exist_ok=True)

    # Use simplified evaluator
    evaluator = SimplifiedXAIEvaluator()

    # Compatible with models without predict_proba
    model_for_eval = model if hasattr(model, 'predict_proba') else _PredictProbaAdapter(model, scaler=scaler, uses_scaled=uses_scaled)

    # Check if this is global explanation evaluation
    is_global = 'global' in method_name.lower()
    
    if is_global and global_importance is not None:
        # ===== Global Explanation Evaluation =====
        print(f"[GLOBAL] Starting GLOBAL explanation evaluation {model_name} - {method_name}")
        
        # Prepare samples for evaluation
        eval_sample_size = min(200, len(X_test_df))
        X_eval = X_test_df.iloc[:eval_sample_size].values if hasattr(X_test_df, 'iloc') else np.asarray(X_test_df)[:eval_sample_size]
        y_eval = np.array(y_test)[:eval_sample_size] if y_test is not None else None
        
        # Determine explainer type
        explainer_type = 'shap' if 'shap' in method_name else 'lime'
        
        # Ensure X_eval is numeric
        try:
            X_eval_numeric = np.asarray(X_eval)
            if X_eval_numeric.dtype == object:
                X_eval_numeric = X_eval_numeric.astype('float64', copy=False)
        except Exception:
            X_eval_numeric = X_eval
        
        print(f"  📊 Evaluation samples: {len(X_eval_numeric)}")
        print(f"  📊 Global importance features: {len(global_importance)}")
        
        # Evaluate using global explanation methods
        results = evaluator.evaluate_global_explanation(
            model=model_for_eval,
            X_samples=X_eval_numeric,
            explainer=explainer,
            global_importance=global_importance,
            y_samples=y_eval,
            sample_size=eval_sample_size,
            explainer_type=explainer_type
        )
        
    else:
        # ===== Local Explanation Evaluation (original logic) =====
        # Build explanations and evaluation data
        if explanations is None:
            if shap_values_for_plot is None:
                return
            available_n = min(getattr(shap_values_for_plot, 'shape', [0])[0], len(X_test_df))
            sel_idx = _select_sample_indices(y_test, available_n, EVAL_SAMPLE_SIZE, stratified=EVAL_STRATIFIED)
            if sel_idx.size == 0:
                return
            shap_subset = shap_values_for_plot[sel_idx]
            X_eval = X_test_df.iloc[sel_idx].values if hasattr(X_test_df, 'iloc') else np.asarray(X_test_df)[sel_idx]
            y_eval = np.array(y_test)[sel_idx]
            explanations = _build_explanations_from_shap(shap_subset, top_k=10, max_samples=len(sel_idx))
        else:
            available_n = min(len(explanations), len(X_test_df))
            sel_idx = _select_sample_indices(y_test, available_n, EVAL_SAMPLE_SIZE, stratified=EVAL_STRATIFIED)
            if sel_idx.size == 0:
                return
            explanations = [explanations[i] for i in sel_idx]
            X_eval = X_test_df.iloc[sel_idx].values if hasattr(X_test_df, 'iloc') else np.asarray(X_test_df)[sel_idx]
            y_eval = np.array(y_test)[sel_idx]

        # Execute evaluation
        print(f"[LOCAL] Starting LOCAL explanation evaluation {model_name} - {method_name}")
        print(f"  📊 Evaluation samples: {len(X_eval)}, explanations: {len(explanations)}")
        
        evaluator.evaluate_faithfulness(model_for_eval, explanations, X_eval, y_eval)
        
        # Determine explainer type
        explainer_type = 'shap' if 'shap' in method_name else 'lime'
        # Ensure X_eval passed to evaluation is pure numeric to avoid LIME object dtype errors
        try:
            X_eval_numeric = np.asarray(X_eval)
            if X_eval_numeric.dtype == object:
                X_eval_numeric = X_eval_numeric.astype('float64', copy=False)
        except Exception:
            X_eval_numeric = X_eval
        evaluator.evaluate_robustness(model_for_eval, explanations, X_eval_numeric, explainer, explainer_type)
        
        # Now explanations contain all features, no need to pass total_features parameter
        evaluator.evaluate_complexity(explanations)

    # Save reports and radar chart
    print("  💾 Generating evaluation report...")
    report_text = evaluator.generate_evaluation_report()
    report_path = os.path.join(model_eval_dir, f'{model_name}_{method_name}_evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"    ✅ Report saved: {report_path}")
    
    print("  📊 Generating radar chart...")
    radar_path = os.path.join(model_eval_dir, f'{model_name}_{method_name}_evaluation_radar.png')
    evaluator.visualize_results(save_path=radar_path)
    print(f"    ✅ Radar chart saved: {radar_path}")
    
    print(f"✅ {model_name} - {method_name} evaluation completed")


def _evaluate_specific_samples(model_name, model, X_test_df, y_test, preds, 
                              selected_samples, method_name, task_type, 
                              shap_explainer=None, lime_explainer=None, 
                              scaler=None, uses_scaled=False, class_names=None):
    """
    Evaluate specific samples for individual metrics generation
    
    Args:
        model_name: Name of the model
        model: Trained model
        X_test_df: Test data DataFrame
        y_test: Test labels
        preds: Model predictions
        selected_samples: List of selected sample indices
        method_name: Method name (shap_local or lime_local)
        task_type: Task type (binary or multiclass)
        shap_explainer: SHAP explainer (for shap_local)
        lime_explainer: LIME explainer (for lime_local)
        scaler: Data scaler (for neural networks)
        uses_scaled: Whether model uses scaled data
        class_names: Class names for multiclass
    """
    if len(selected_samples) == 0:
        print(f"No samples selected for {method_name} evaluation")
        return
    
    # Create output directory
    model_dir = create_model_result_dir(task_type, model_name)
    eval_dir = os.path.join(model_dir, "evaluate", f"{method_name}_individual")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Convert to numpy array for easier indexing
    if hasattr(X_test_df, 'iloc'):
        X_array = X_test_df.values
    else:
        X_array = np.asarray(X_test_df)
    
    # Calculate global means for masking baseline
    global_means = np.mean(X_array, axis=0) if isinstance(X_array, np.ndarray) else None
    
    # Create evaluator
    evaluator = SimplifiedXAIEvaluator()
    
    # Compatible with models without predict_proba
    model_for_eval = model if hasattr(model, 'predict_proba') else _PredictProbaAdapter(model, scaler=scaler, uses_scaled=uses_scaled)
    
    print(f"🔍 Starting individual sample evaluation {model_name} - {method_name}")
    print(f"  📊 Selected samples: {selected_samples}")
    
    individual_results = []
    
    for i, sample_idx in enumerate(selected_samples):
        print(f"  📊 Processing sample {i+1}/{len(selected_samples)} (index: {sample_idx})")
        
        # Get sample data
        x_sample = X_array[sample_idx]
        y_sample = y_test[sample_idx]
        pred_sample = preds[sample_idx]
        
        # Generate explanation based on method
        if method_name == "shap_local" and shap_explainer is not None:
            try:
                # Generate SHAP explanation for this sample
                if hasattr(X_test_df, 'iloc'):
                    instance_df = X_test_df.iloc[[sample_idx]]
                else:
                    instance_df = pd.DataFrame([x_sample])
                
                # Handle different SHAP explainers
                if model_name in ['MLP', 'DNN']:
                    # For neural networks, use numpy array
                    if model_name == 'MLP':
                        # GradientExplainer expects float32
                        instance_data = instance_df.values.astype('float32')
                    else:
                        # DeepExplainer - ensure correct data format
                        instance_data = instance_df.values.astype('float32')
                    shap_values_instance = shap_explainer.shap_values(instance_data)
                else:
                    # For tree-based models
                    shap_values_instance = shap_explainer.shap_values(instance_df)
                
                # Handle different SHAP output formats
                if isinstance(shap_values_instance, list):
                    # For binary classification, use the positive class (index 1)
                    if task_type == 'binary':
                        shap_values_instance = shap_values_instance[1] if len(shap_values_instance) > 1 else shap_values_instance[0]
                    else:
                        # For multiclass, use the predicted class
                        shap_values_instance = shap_values_instance[pred_sample] if pred_sample < len(shap_values_instance) else shap_values_instance[0]
                elif shap_values_instance.ndim > 1:
                    # For multi-dimensional output, take the first sample
                    if shap_values_instance.ndim == 3:
                        # 3D array: (samples, features, classes) -> take first sample and appropriate class
                        if task_type == 'binary':
                            # For binary classification, use positive class (index 1) if available
                            shap_values_instance = shap_values_instance[0, :, 1] if shap_values_instance.shape[2] > 1 else shap_values_instance[0, :, 0]
                        else:
                            # For multiclass, use the predicted class
                            shap_values_instance = shap_values_instance[0, :, pred_sample] if pred_sample < shap_values_instance.shape[2] else shap_values_instance[0, :, 0]
                    elif shap_values_instance.ndim == 2:
                        # 2D array: (samples, features) -> take first sample
                        shap_values_instance = shap_values_instance[0]
                
                # Ensure shap_values_instance is 1D array
                if shap_values_instance.ndim > 1:
                    shap_values_instance = shap_values_instance.flatten()
                
                # Convert to explanation format
                top_indices = np.argsort(np.abs(shap_values_instance))[-10:]
                feature_importance = {}
                for idx in top_indices:
                    try:
                        value = shap_values_instance[idx]
                        if hasattr(value, 'item'):
                            value = value.item()  # Convert numpy scalar to Python scalar
                        feature_importance[int(idx)] = float(value)
                    except (ValueError, TypeError):
                        feature_importance[int(idx)] = 0.0
                explanation = {
                    'feature_importance': feature_importance,
                    'text_explanation': f'SHAP explanation for sample {sample_idx}',
                    'rules': []
                }
                
            except Exception as e:
                print(f"    ❌ Failed to generate SHAP explanation for sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        elif method_name == "lime_local" and lime_explainer is not None:
            try:
                # Generate LIME explanation for this sample
                instance = x_sample
                
                if hasattr(model, 'predict_proba'):
                    def predict_fn_wrapper(x):
                        if uses_scaled and scaler is not None:
                            x_scaled = scaler.transform(x)
                            proba = model.predict_proba(x_scaled)
                        else:
                            proba = model.predict_proba(x)
                        
                        # ensure correct probability array shape, especially for MLP/DNN models
                        if proba.ndim == 1:
                            # if 1D array, convert to 2D
                            proba = proba.reshape(1, -1)
                        return proba
                    predict_fn = predict_fn_wrapper
                else:
                    predict_fn = lambda x: np.hstack([(1 - model.predict(x, verbose=0)), model.predict(x, verbose=0)])
                
                # modified: generate explanations for all features, not just top-10
                exp = lime_explainer.explain_instance(instance, predict_fn, num_features=len(instance))
                weights = exp.as_list()
                
                # Convert to explanation format
                feature_importance = {}
                for feat, w in weights:
                    try:
                        # Try to extract feature index from feature name
                        feat_str = str(feat)
                        if 'feature_' in feat_str:
                            # Handle different LIME feature name formats
                            if ' <= ' in feat_str or ' > ' in feat_str or ' < ' in feat_str:
                                # Format: "feature_X <= value" or "feature_X > value"
                                parts = feat_str.split('_')
                                if len(parts) >= 2:
                                    idx_str = parts[1].split()[0]  # Get the number before any operator
                                    idx = int(idx_str)
                                else:
                                    continue
                            else:
                                # Simple format: "feature_X"
                                idx = int(feat_str.split('_')[1])
                        else:
                            # Fallback: use position in weights list
                            idx = len(feature_importance)
                        feature_importance[idx] = float(w)
                    except Exception:
                        continue
                
                # assign 0 importance to missing features, ensure all features are included
                for idx in range(len(instance)):
                    if idx not in feature_importance:
                        feature_importance[idx] = 0.0
                
                explanation = {
                    'feature_importance': feature_importance,
                    'text_explanation': f'LIME explanation for sample {sample_idx}',
                    'rules': []
                }
                
            except Exception as e:
                print(f"    ❌ Failed to generate LIME explanation for sample {sample_idx}: {e}")
                continue
        else:
            print(f"    ❌ No explainer available for {method_name}")
            continue
        
        # Evaluate this sample
        try:
            # determine explainer type and total number of features
            explainer_type = 'shap' if 'shap' in method_name else 'lime'
            total_features = len(x_sample)
            
            # get corresponding explainer
            current_explainer = None
            if method_name == "shap_local" and shap_explainer is not None:
                current_explainer = shap_explainer
            elif method_name == "lime_local" and lime_explainer is not None:
                current_explainer = lime_explainer
            
            sample_results = evaluator.evaluate_single_sample_complete(
                model_for_eval, explanation, x_sample, global_means,
                explainer=current_explainer, explainer_type=explainer_type,
                total_features=total_features
            )
            
            # Add sample metadata
            sample_results['sample_index'] = int(sample_idx)
            sample_results['true_label'] = int(y_sample)
            sample_results['predicted_label'] = int(pred_sample)
            sample_results['prediction_correct'] = bool(y_sample == pred_sample)
            
            individual_results.append(sample_results)
            
            print(f"    ✅ Sample {sample_idx} evaluated - Overall score: {sample_results['overall_score']:.3f}")
            
        except Exception as e:
            print(f"    ❌ Failed to evaluate sample {sample_idx}: {e}")
            continue
    
    if individual_results:
        # Save individual results
        import json
        results_file = os.path.join(eval_dir, f'{model_name}_{method_name}_individual_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(individual_results, f, ensure_ascii=False, indent=2)
        
        # Generate summary report
        report = f"# Individual Sample Evaluation Report - {model_name} - {method_name}\n\n"
        report += f"**Task Type**: {task_type}\n"
        report += f"**Method**: {method_name}\n"
        report += f"**Total Samples Evaluated**: {len(individual_results)}\n\n"
        
        # Calculate average scores
        avg_overall = np.mean([r['overall_score'] for r in individual_results])
        avg_faithfulness = np.mean([r['faithfulness']['overall_score'] for r in individual_results])
        avg_robustness = np.mean([r['robustness']['overall_score'] for r in individual_results])
        avg_complexity = np.mean([r['complexity']['overall_score'] for r in individual_results])
        
        report += f"## Average Scores\n\n"
        report += f"- **Overall Score**: {avg_overall:.3f}\n"
        report += f"- **Faithfulness**: {avg_faithfulness:.3f}\n"
        report += f"- **Robustness**: {avg_robustness:.3f}\n"
        report += f"- **Complexity**: {avg_complexity:.3f}\n\n"
        
        report += f"## Individual Sample Results\n\n"
        for i, result in enumerate(individual_results):
            sample_idx = result['sample_index']
            correct = "✅" if result['prediction_correct'] else "❌"
            report += f"### Sample {i+1} (Index: {sample_idx}) {correct}\n\n"
            report += f"- **True Label**: {result['true_label']}\n"
            report += f"- **Predicted Label**: {result['predicted_label']}\n"
            report += f"- **Overall Score**: {result['overall_score']:.3f}\n"
            report += f"- **Faithfulness**: {result['faithfulness']['overall_score']:.3f}\n"
            report += f"- **Robustness**: {result['robustness']['overall_score']:.3f}\n"
            report += f"- **Complexity**: {result['complexity']['overall_score']:.3f}\n\n"
        
        # Save report
        report_file = os.path.join(eval_dir, f'{model_name}_{method_name}_individual_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Generate individual radar charts for each sample
        print("  📊 Generating individual radar charts...")
        for i, result in enumerate(individual_results):
            try:
                # Create evaluator for this sample
                sample_evaluator = SimplifiedXAIEvaluator()
                
                # Set evaluation results for this sample
                sample_evaluator.evaluation_results = {
                    'faithfulness': {'overall_score': result['faithfulness']['overall_score']},
                    'robustness': {'overall_score': result['robustness']['overall_score']},
                    'complexity': {'overall_score': result['complexity']['overall_score']}
                }
                
                # Generate radar chart for this sample
                sample_idx = result['sample_index']
                correct = "correct" if result['prediction_correct'] else "incorrect"
                radar_path = os.path.join(eval_dir, f'sample_{i+1}_idx_{sample_idx}_{correct}_radar.png')
                sample_evaluator.visualize_results(save_path=radar_path)
                
                print(f"    ✅ Sample {i+1} radar chart saved: {radar_path}")
                
            except Exception as e:
                print(f"    ❌ Failed to generate radar chart for sample {i+1}: {e}")
                continue
        
        # Generate average radar chart
        try:
            avg_evaluator = SimplifiedXAIEvaluator()
            avg_evaluator.evaluation_results = {
                'faithfulness': {'overall_score': avg_faithfulness},
                'robustness': {'overall_score': avg_robustness},
                'complexity': {'overall_score': avg_complexity}
            }
            
            avg_radar_path = os.path.join(eval_dir, f'{model_name}_{method_name}_average_radar.png')
            avg_evaluator.visualize_results(save_path=avg_radar_path)
            print(f"    ✅ Average radar chart saved: {avg_radar_path}")
            
        except Exception as e:
            print(f"    ❌ Failed to generate average radar chart: {e}")
        
        print(f"    ✅ Individual results saved to: {results_file}")
        print(f"    ✅ Individual report saved to: {report_file}")
        print(f"    ✅ Average overall score: {avg_overall:.3f}")
    
    print(f"✅ {model_name} - {method_name} individual evaluation completed")


# Initialize SHAP for notebook visualization
shap.initjs()

# Enable TF GPU memory growth (on all visible GPUs) to avoid OOM spikes
try:
    gpus = tf.config.list_physical_devices('GPU')
    for _g in gpus:
        tf.config.experimental.set_memory_growth(_g, True)
except Exception:
    pass

def make_xgb_classifier(task: str):
    """Create an XGBClassifier preferring GPU; fallback to CPU if unsupported."""
    base_params = dict(
        random_state=42,
        n_estimators=200,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
    )
    if task == 'binary':
        base_params.update(objective='binary:logistic')
    else:
        base_params.update(objective='multi:softprob')

    gpu_param_variants = [
        {'device': 'cuda'},  # xgboost >= 1.7
        {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'},  # older style
    ]

    for gpu_params in gpu_param_variants:
        try:
            model = xgb.XGBClassifier(**base_params, **gpu_params)
            # Parameter validation
            model.get_params()
            return model
        except Exception:
            continue
    # CPU fallback
    return xgb.XGBClassifier(**base_params)

def create_dnn_model_bi(input_shape):
    """Helper function to create the DNN model."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_mlp_model_bi(input_shape):
    """Helper function to create the DNN model."""
    model = Sequential([
        Dense(100, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_explain_binary_models(X, y):
    """
    Trains, evaluates, and explains multiple models with enhanced performance
    metrics and individual prediction analysis using SHAP and LIME.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    feature_names = X.columns.astype(str).tolist()

    # y = np.array(y)
    # # X_train, X_test, y_train, y_test = train_test_split(
    # #     X, y, test_size=0.2, random_state=42, stratify=y
    # # )
    # split_idx = int(0.8 * len(X))

    # X_train = X.iloc[:split_idx]
    # X_test  = X.iloc[split_idx:]

    # y_train = y[:split_idx]
    # y_test  = y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    model_configs = {
        'DecisionTree': {'model': DecisionTreeClassifier(random_state=42, class_weight='balanced')},
        'RandomForest': {'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')},
        'XGBoost': {'model': make_xgb_classifier('binary')},
        'MLP': {'model': 'create_mlp_model_bi'},
        'DNN': {'model': 'create_dnn_model_bi'}
    }

    models = {}
    llm_bundle = {
        'task': 'binary',
        'label_mapping': {0: 'Benign', 1: 'Malicious'},
        'feature_names': feature_names,
        'models': {}
    }
    top_features_indices = {}

    # compute class/sample weights for imbalance handling (binary)
    classes_unique = np.unique(y_train)
    class_weights_array = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)
    class_weight_dict = dict(zip(classes_unique, class_weights_array))
    sample_weight = np.array([class_weight_dict[c] for c in y_train])

    for name, config in model_configs.items():
        print(f"\n{'='*30}\n--- Processing Model: {name} ---\n{'='*30}")

        is_neural_net = name in ['MLP', 'DNN']
        current_X_train = X_train_scaled_df if is_neural_net else X_train
        current_X_test = X_test_scaled_df if is_neural_net else X_test

        if name in ['DNN','MLP']:
            if name == 'MLP':
                model = create_mlp_model_bi(current_X_train.shape[1])
            else:
                model = create_dnn_model_bi(current_X_train.shape[1])
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            # pass class_weight for imbalance
            model.fit(current_X_train, y_train, epochs=100, batch_size=512, validation_split=0.1, callbacks=[early_stop], verbose=0, class_weight=class_weight_dict)
            preds = (model.predict(current_X_test, verbose=0) > 0.5).astype('int32').flatten()
        else:
            model = config['model']
            if name == 'XGBoost':
                # set scale_pos_weight and use sample_weight
                class_counts = np.bincount(y_train)
                if len(class_counts) == 2 and class_counts[1] > 0:
                    spw = class_counts[0] / class_counts[1]
                    try:
                        model.set_params(scale_pos_weight=spw)
                    except Exception:
                        pass
                model.fit(current_X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(current_X_train, y_train)
            preds = model.predict(current_X_test)

        models[name] = model
        # Create organized directory for this model
        model_dir = create_model_result_dir('binary', name)
        
        # Save minimal performance and SHAP/LIME paths for LLM bundle later
        llm_bundle['models'][name] = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'report_image': os.path.join(model_dir, "classification_report.png"),
            'confusion_image': os.path.join(model_dir, "confusion_matrix.png"),
            'shap_summary_image': os.path.join(model_dir, "shap_summary_plot.png"),
            'lime_json': os.path.join(model_dir, f"{name}_lime_local_top10.json")
        }

        print("\n--- Model Performance ---")
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, zero_division=0))
        # Save classification report as markdown text for LLM parsing
        try:
            report_txt = classification_report(y_test, preds, target_names=['Benign', 'Malicious'], zero_division=0)
            with open(os.path.join(model_dir, "classification_report.md"), 'w', encoding='utf-8') as f:
                f.write("# Classification Report\n\n")
                f.write("````\n")
                f.write(report_txt)
                f.write("\n````\n")
        except Exception as _e:
            print(f"Failed to write classification_report.md: {_e}")

        # Save classification report as image
        save_classification_report_image(
            y_true=y_test,
            y_pred=preds,
            title=f'Classification Report ({name} Binary)',
            save_path=os.path.join(model_dir, "classification_report.png"),
            target_names=['Benign', 'Malicious']
        )

        # === Precision-Recall (Binary) with AUPRC and Bootstrap CI ===
        try:
            # scores for positive class
            if name in ['DNN','MLP']:
                y_score = model.predict(current_X_test, verbose=0).flatten()
            else:
                y_score = model.predict_proba(current_X_test)[:, 1] if hasattr(model, 'predict_proba') else preds

            precisions, recalls, _ = precision_recall_curve(y_test, y_score)
            ap = float(average_precision_score(y_test, y_score))

            # bootstrap AP CI
            rng = np.random.default_rng(42)
            boot_aps = []
            B = 200
            n = len(y_test)
            for _ in range(B):
                idx = rng.integers(0, n, size=n)
                boot_aps.append(average_precision_score(y_test[idx], y_score[idx]))
            low, high = np.percentile(boot_aps, [2.5, 97.5])

            # save curve plot
            plt.figure(figsize=(5,4))
            plt.plot(recalls, precisions, label=f'AUPRC={ap:.3f} [{low:.3f}, {high:.3f}]')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve (Binary)')
            plt.legend()
            plt.tight_layout()
            pr_path = os.path.join(model_dir, 'pr_curve_binary.png')
            plt.savefig(pr_path, dpi=300, bbox_inches='tight')
            plt.close()

            # save metrics json
            with open(os.path.join(model_dir, 'pr_metrics_binary.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'auprc': ap,
                    'auprc_ci_95': [float(low), float(high)]
                }, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"Failed PR/AUPRC (binary) for {name}: {_e}")

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix for {name}')
        plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        # Save confusion matrix values for LLM
        try:
            cm_json_path = os.path.join(model_dir, "confusion_matrix.json")
            if cm.shape == (2, 2):
                tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
                with open(cm_json_path, 'w', encoding='utf-8') as f:
                    json.dump({"tn": tn, "fp": fp, "fn": fn, "tp": tp}, f, ensure_ascii=False, indent=2)
            else:
                with open(cm_json_path, 'w', encoding='utf-8') as f:
                    json.dump({"matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"Failed to write confusion_matrix.json: {_e}")

        # --- Create Explainers (SHAP & LIME) ---
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=current_X_train.values.astype('float64'),
            feature_names=feature_names,
            class_names=['Benign', 'Malicious'],
            mode='classification'
        )

        shap_explainer = None
        X_test_sample = current_X_test.head(1000)
        
        # Dynamic background data size based on dataset size
        train_size = len(current_X_train)
        if train_size <= 1000:
            background_size = min(100, train_size)  # Use up to 100 samples for small datasets
        elif train_size <= 10000:
            background_size = min(500, train_size)  # Use up to 500 samples for medium datasets
        else:
            background_size = min(1000, train_size)  # Use up to 1000 samples for large datasets
        
        background_data = shap.sample(current_X_train, background_size).values.astype('float32')
        print(f"Using {background_size} samples as background data (dataset size: {train_size})")

        if name in ['DecisionTree', 'RandomForest', 'XGBoost']:
            # Prefer GPU TreeShap when available (XGBoost + cuda), otherwise fallback
            try:
                shap_explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
                shap_values = shap_explainer.shap_values(X_test_sample)
            except Exception:
                shap_explainer = shap.TreeExplainer(model)
                shap_values = shap_explainer.shap_values(X_test_sample)
        elif name == 'MLP':
            # ensure background_data format is correct
            shap_explainer = shap.GradientExplainer(model, background_data)
            # ensure test data format is correct
            test_data = X_test_sample.values.astype('float32')
            shap_values = shap_explainer.shap_values(test_data)
        elif name == 'DNN':
            # ensure background_data format is correct
            shap_explainer = shap.DeepExplainer(model, background_data)
            # ensure test data format is correct
            test_data = X_test_sample.values.astype('float32')
            shap_values = shap_explainer.shap_values(test_data)

        if isinstance(shap_values, list):
            shap_values_for_plot = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_for_plot = shap_values[:, :, 0]
        else:
            shap_values_for_plot = shap_values

        print("\n--- SHAP Summary Plot ---")
        plt.figure()
        shap.summary_plot(shap_values_for_plot, X_test_sample, show=False)
        plt.title(f'SHAP Summary Plot for {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


        mean_abs_shap = np.mean(np.abs(shap_values_for_plot), axis=0)
        top_features_indices[name] = np.argsort(mean_abs_shap)[-10:][::-1]
        # Save SHAP top features (feature -> importance)
        try:
            top_idx = np.argsort(mean_abs_shap)[-50:][::-1]
            shap_top = [
                {"feature": str(feature_names[i]) if i < len(feature_names) else str(i), "importance": float(mean_abs_shap[i])}
                for i in top_idx
            ]
            with open(os.path.join(model_dir, "shap_top_features.json"), 'w', encoding='utf-8') as f:
                json.dump(shap_top, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"Failed to write shap_top_features.json: {_e}")

        # Save aggregated LIME top features
        try:
            # Define a prediction function that returns probability for class index when possible
            def _predict_proba_fn(X_np):
                try:
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(X_np)
                    # Keras models: use predict; ensure 2D probs
                    probs = model.predict(X_np, verbose=0)
                    probs = np.asarray(probs)
                    if probs.ndim == 1:
                        probs = np.vstack([1 - probs, probs]).T
                    return probs
                except Exception:
                    # Fallback to decision function or predict
                    if hasattr(model, 'decision_function'):
                        df = model.decision_function(X_np)
                        # convert to pseudo-probabilities
                        from scipy.special import expit
                        if df.ndim == 1:
                            p1 = expit(df)
                            return np.vstack([1 - p1, p1]).T
                        return df
                    preds_bin = model.predict(X_np)
                    return np.vstack([1 - preds_bin, preds_bin]).T

            agg_importance = {}
            sample_count = min(1000, len(current_X_test))
            X_lime = current_X_test.head(sample_count).values
            class_idx = 1  # focus on positive class for aggregation
            for i in range(X_lime.shape[0]):
                exp = lime_explainer.explain_instance(
                    X_lime[i],
                    lambda x: _predict_proba_fn(np.asarray(x)),
                    num_features=10,
                    top_labels=1
                )
                # Get explanation for target class
                try:
                    label = class_idx if class_idx in exp.as_map() else exp.top_labels[0]
                    for feat_idx, weight in exp.as_map()[label]:
                        feat_name = str(feature_names[feat_idx]) if feat_idx < len(feature_names) else str(feat_idx)
                        agg_importance[feat_name] = agg_importance.get(feat_name, 0.0) + float(abs(weight))
                except Exception:
                    continue

            if agg_importance:
                # Normalize and sort
                total = sum(agg_importance.values()) or 1.0
                lime_top = [
                    {"feature": k, "importance": v / total}
                    for k, v in sorted(agg_importance.items(), key=lambda kv: kv[1], reverse=True)[:50]
                ]
                with open(os.path.join(model_dir, "lime_top_features.json"), 'w', encoding='utf-8') as f:
                    json.dump(lime_top, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"Failed to write lime_top_features.json: {_e}")


        # --- XAI Quality Assessment (SHAP-Global, using new global evaluation method) ---
        try:
            if shap_values_for_plot is not None and getattr(shap_values_for_plot, 'shape', None) is not None:
                # Global importance: take absolute value then mean by sample
                global_importance = np.mean(np.abs(shap_values_for_plot), axis=0)
                fi_global = {int(idx): float(val) for idx, val in enumerate(global_importance)}

                _evaluate_and_save(
                    model_name=name,
                    model=model,
                    X_test_df=X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test),
                    y_test=y_test,
                    shap_values_for_plot=None,
                    scaler=scaler,
                    uses_scaled=is_neural_net,
                    class_names=['Benign','Malicious'],
                    explanations=None,
                    method_name="shap_global",
                    task_type="binary",
                    explainer=shap_explainer,
                    global_importance=fi_global
                )
        except Exception as _e:
            print(f"XAI evaluation failed (SHAP Global) {name}: {_e}")

        # --- XAI Quality Assessment (LIME-Local) ---
        try:
            lime_explanations = []
            # select samples (supports stratification)
            available_n = X_test_sample.shape[0]
            sel_idx = _select_sample_indices(y_test, available_n, EVAL_SAMPLE_SIZE, stratified=EVAL_STRATIFIED)
            for i in sel_idx:
                instance = X_test_sample.iloc[i].values if hasattr(X_test_sample, 'iloc') else X_test_sample[i]
                if hasattr(model, 'predict_proba'):
                    def predict_fn_wrapper(x):
                        instance_df_lime = pd.DataFrame(x, columns=feature_names)
                        proba = model.predict_proba(instance_df_lime)
                        # Ensure correct probability array shape
                        if proba.ndim == 1:
                            # If 1D array, reshape to 2D
                            proba = proba.reshape(1, -1)
                        return proba
                    predict_fn = predict_fn_wrapper
                else:
                    predict_fn = lambda x: np.hstack([(1 - model.predict(x, verbose=0)), model.predict(x, verbose=0)])
                # modified: generate explanations for all features, not just top-10
                exp = lime_explainer.explain_instance(instance, predict_fn, num_features=len(instance))
                # convert to structure needed by evaluator, using integer indices
                # fix: check model type and returned probability array shape to avoid index out of bounds
                if hasattr(model, 'predict_proba') and not (name in ['MLP', 'DNN']):
                    # check model returned probability array shape
                    try:
                        test_proba = model.predict_proba(instance.reshape(1, -1))
                        if test_proba.shape[1] >= 2:
                            weights = exp.as_list(label=1)
                        else:
                            weights = exp.as_list()
                    except Exception:
                        weights = exp.as_list()
                else:
                    # for MLP/DNN models, use as_list() directly, do not use label parameter
                    # because MLP/DNN may return 1D probability array in binary classification, using label=1 causes index out of bounds
                    weights = exp.as_list()
                # weights: list of (feature_name, weight)
                fi = {}
                for feat, w in weights:
                    # skip when parsing index from feature name fails, or match by column name
                    try:
                        idx = int(str(feat).split()[0].replace('feature_', '').replace('f', ''))
                    except Exception:
                        # fallback: locate by column name
                        if hasattr(X_test_sample, 'columns') and str(feat) in X_test_sample.columns:
                            idx = int(list(X_test_sample.columns).index(str(feat)))
                        else:
                            continue
                    fi[int(idx)] = float(w)
                
                # assign 0 importance to missing features, ensure all features are included
                for idx in range(len(instance)):
                    if idx not in fi:
                        fi[idx] = 0.0
                lime_explanations.append({'feature_importance': fi, 'text_explanation': 'LIME local explanation', 'rules': []})

        except Exception as _e:
            print(f"XAI evaluation failed (LIME) {name}: {_e}")
        # Save per-sample LIME JSON for LLM (top-10 only) - now in results directory
        try:
            if 'lime_explanations' in locals() and len(lime_explanations) > 0:
                lime_json = []
                for i, ex in enumerate(lime_explanations):
                    sorted_items = sorted(ex['feature_importance'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
                    lime_json.append({
                        'sample_index': int(i),
                        'top_features': [{'index': int(k), 'weight': float(v)} for k, v in sorted_items]
                    })
                # Save in results directory instead of separate LLM directory
                lime_json_path = os.path.join(RESULT_DIR, 'binary', name, f'{name}_lime_local_top10.json')
                with open(lime_json_path, 'w', encoding='utf-8') as f:
                    import json as _json
                    _json.dump(lime_json, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # note: LIME does not provide strict global explanations, no LIME-Global evaluation here

        # --- Individual Prediction Analysis ---
        print("\n--- Individual Prediction Analysis ---")
        correct_indices = np.where(preds == y_test)[0]
        incorrect_indices = np.where(preds != y_test)[0]
        # Set random seed for reproducible sample selection
        np.random.seed(42)
        samples_correct = np.random.choice(correct_indices, min(3, len(correct_indices)), replace=False) if len(correct_indices) > 0 else np.array([], dtype=int)
        samples_incorrect = np.random.choice(incorrect_indices, min(3, len(incorrect_indices)), replace=False) if len(incorrect_indices) > 0 else np.array([], dtype=int)
        
        # Combine selected samples for evaluation
        selected_samples = np.concatenate([samples_correct, samples_incorrect])
        selected_samples = np.sort(selected_samples).astype(int)
        
        print(f"Selected samples for visualization and evaluation: {selected_samples}")
        
        analysis_samples = {"Correctly Predicted": samples_correct, "Incorrectly Predicted": samples_incorrect}

        for category, indices in analysis_samples.items():
            if not indices.any():
                print(f"\nNo samples found for '{category}' category.")
                continue

            print(f"\n--- Analyzing {len(indices)} '{category}' Samples ---")
            for i, idx in enumerate(indices):
                print(f"\n--- Sample {i+1} (Test Set Index: {idx}) ---")
                instance_df = current_X_test.iloc[[idx]]
                print(instance_df)

                # SHAP Force Plot
                try:
                    print(f"\nSHAP Force Plot for Index {idx}:")
                    if name=='MLP':
                      # Use the same background data size as defined earlier
                      background_sample = shap.sample(X_train_scaled_df, background_size)
                      predictions = model.predict(background_sample.values.astype('float32'))
                      base_value = predictions.mean(axis=0)
                      if len(base_value) > 1:
                          base_value = base_value[0]
                    else:
                      base_value = shap_explainer.expected_value
                    shap_values_instance = shap_explainer.shap_values(instance_df if name not in ['MLP','DNN'] else instance_df.values.astype('float32'))

                    if name in ['MLP','DNN']:
                      base_value= np.array(base_value)[0]
                      shap_values_instance = shap_values_instance.flatten()
                    else:
                      shap_values_instance = shap_values_instance[0] if isinstance(shap_values_instance, list) else shap_values_instance
                      shap_values_instance = shap_values_instance[0]
                      base_value = base_value[0] if isinstance(base_value, np.ndarray) else base_value


                    # --- FIX ---
                    # If shap_values are 2D (e.g., from KernelExplainer), select the values for the positive class (column 1).
                    if shap_values_instance.ndim == 2:
                    # If it has multiple columns (like from MLP's KernelExplainer), select the positive class.
                        if shap_values_instance.shape[1] > 1:
                            shap_values_instance = shap_values_instance[:, 1]
                        # If it only has one column (like from DNN's DeepExplainer), just flatten it to 1D.
                    # Pass the feature values as a Series to ensure it's treated as a single sample.
                    plt.figure(figsize=(16, 8))
                    plots.force(base_value, shap_values_instance, instance_df.iloc[0], matplotlib=True, show=False)
                    plt.title(f'SHAP Force Plot for Index {idx}')
                    plt.tight_layout(pad=2.0)
                    plt.savefig(os.path.join(model_dir, f'shap_force_plot_{idx}.png'), dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
                    
                    # Save SHAP values as JSON for LLM
                    try:
                        shap_data = {
                            "sample_index": int(idx),
                            "base_value": float(base_value),
                            "feature_values": instance_df.iloc[0].to_dict(),
                            "shap_values": shap_values_instance.tolist(),
                            "feature_names": feature_names,
                            "prediction": float(preds[idx]),
                            "true_label": float(y_test[idx])
                        }
                        
                        # Create feature importance list
                        feature_importance = []
                        for i, (feature_name, shap_value) in enumerate(zip(feature_names, shap_values_instance)):
                            feature_importance.append({
                                "feature": feature_name,
                                "shap_value": float(shap_value),
                                "feature_value": float(instance_df.iloc[0][feature_name])
                            })
                        
                        # Sort by absolute SHAP value
                        feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                        shap_data["feature_importance"] = feature_importance
                        
                        # Save to JSON file
                        shap_json_path = os.path.join(model_dir, f'shap_individual_{idx}.json')
                        with open(shap_json_path, 'w', encoding='utf-8') as f:
                            import json as _json
                            _json.dump(shap_data, f, ensure_ascii=False, indent=2)
                        print(f"    ✅ SHAP JSON saved: {shap_json_path}")
                        
                    except Exception as json_e:
                        print(f"    ⚠️  Failed to save SHAP JSON for index {idx}: {json_e}")

                except Exception as e:
                    print(f"Could not generate SHAP plot for index {idx}: {e}")

                # LIME Plot
                try:
                    print(f"\nLIME Plot for Index {idx}:")
                    if hasattr(model, 'predict_proba'):
                        def predict_fn_wrapper(x):
                            instance_df_lime = pd.DataFrame(x, columns=feature_names)
                            proba = model.predict_proba(instance_df_lime)
                            # ensure correct probability array shape
                            if proba.ndim == 1:
                                # if 1D array, convert to 2D
                                proba = proba.reshape(1, -1)
                            return proba
                        predict_fn = predict_fn_wrapper
                    else:
                        predict_fn = lambda x: np.hstack([(1 - model.predict(x, verbose=0)), model.predict(x, verbose=0)])

                    explanation = lime_explainer.explain_instance(
                    X_test.iloc[idx].values,
                    predict_fn,
                    num_features=10)
                    explanation.save_to_file(os.path.join(model_dir, f'lime_explanation_{idx}.html'))
                    
                    # Save LIME values as text for LLM
                    try:
                        # Get explanation data
                        lime_weights = explanation.as_list()
                        # Get instance data for feature values
                        instance_data = current_X_test.iloc[idx]
                        
                        # Create LIME explanation text
                        lime_text_content = f"""LIME Explanation for Sample {idx}
Prediction: {preds[idx]}
True Label: {y_test[idx]}

Feature Values:
{instance_data.to_string()}

LIME Feature Importance (Top 10):
"""
                        
                        # Sort by absolute LIME weight and add to text
                        sorted_weights = sorted(lime_weights, key=lambda x: abs(x[1]), reverse=True)
                        for i, (feature_name, weight) in enumerate(sorted_weights[:10]):
                            lime_text_content += f"{i+1:2d}. {feature_name}: {weight:.4f}\n"
                        
                        # Save to text file
                        lime_text_path = os.path.join(model_dir, f'lime_individual_{idx}.txt')
                        with open(lime_text_path, 'w', encoding='utf-8') as f:
                            f.write(lime_text_content)
                        print(f"    ✅ LIME text saved: {lime_text_path}")
                        
                    except Exception as text_e:
                        print(f"    ⚠️  Failed to save LIME text for index {idx}: {text_e}")
                    
                    # explanation = lime_explainer.explain_instance(instance_df.iloc[0].values, predict_fn, num_features=10)
                    # fig = explanation.as_pyplot_figure()
                    # plt.title(f'LIME Explanation for Index {idx}')
                    # plt.tight_layout()
                    # plt.show()
                except Exception as e:
                    print(f"Could not generate LIME plot for index {idx}: {e}")

        # --- Individual Sample Evaluation for Selected Samples ---
        print(f"\n--- Individual Sample Evaluation for {name} ---")
        
        # Evaluate SHAP-Local for selected samples
        try:
            _evaluate_specific_samples(
                model_name=name,
                model=model,
                X_test_df=current_X_test,
                y_test=y_test,
                preds=preds,
                selected_samples=selected_samples,
                method_name="shap_local",
                task_type="binary",
                shap_explainer=shap_explainer,
                scaler=scaler,
                uses_scaled=is_neural_net,
                class_names=['Benign', 'Malicious']
            )
        except Exception as e:
            print(f"❌ Individual SHAP evaluation failed for {name}: {e}")
        
        # Evaluate LIME-Local for selected samples
        try:
            _evaluate_specific_samples(
                model_name=name,
                model=model,
                X_test_df=current_X_test,
                y_test=y_test,
                preds=preds,
                selected_samples=selected_samples,
                method_name="lime_local",
                task_type="binary",
                lime_explainer=lime_explainer,
                scaler=scaler,
                uses_scaled=is_neural_net,
                class_names=['Benign', 'Malicious']
            )
        except Exception as e:
            print(f"❌ Individual LIME evaluation failed for {name}: {e}")

    top_features_names = {name: [feature_names[i] for i in indices] for name, indices in top_features_indices.items()}
    top_features_df = pd.DataFrame({k: pd.Series(v) for k, v in top_features_names.items()})

    if not top_features_df.empty:
      top_features_df.index = [f"Top {i+1}" for i in range(len(top_features_df))]

    print("\n--- Top 10 Features Per Model ---")
    print(top_features_df)

    valid_feature_lists = [v for v in top_features_names.values() if v]
    common_features = set.intersection(*(set(features) for features in valid_feature_lists)) if valid_feature_lists else set()

    print(f"\n--- Common Top 10 Features Across All Models ---\n{common_features if common_features else 'None'}")

    # Save to text files in results directory (binary)
    try:
        binary_top_path = os.path.join(RESULT_DIR, 'binary_top_features_per_model.txt')
        with open(binary_top_path, 'w', encoding='utf-8') as f:
            f.write("Top 10 Features Per Model (Binary)\n")
            if not top_features_df.empty:
                f.write(top_features_df.to_string())
            else:
                f.write("<empty>\n")
        binary_common_path = os.path.join(RESULT_DIR, 'binary_common_top_features.txt')
        with open(binary_common_path, 'w', encoding='utf-8') as f:
            f.write("Common Top 10 Features Across All Models (Binary)\n")
            if common_features:
                for feat in sorted(list(common_features)):
                    f.write(f"{feat}\n")
            else:
                f.write("None\n")
    except Exception:
        pass

    # Save global info for LLM - now in results directory
    try:
        import json as _json
        # Save bundle in results directory
        with open(os.path.join(RESULT_DIR, 'binary', 'llm_bundle.json'), 'w', encoding='utf-8') as f:
            _json.dump(llm_bundle, f, ensure_ascii=False, indent=2)
        # Save top-features file paths
        with open(os.path.join(RESULT_DIR, 'binary', 'top_features_files.json'), 'w', encoding='utf-8') as f:
            _json.dump({
                'per_model': f"{RESULT_DIR}/binary_top_features_per_model.txt",
                'common': f"{RESULT_DIR}/binary_common_top_features.txt"
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return models


## NOTE: binary and multiclass pipeline calls moved to __main__

def train_lightgbm_multiclass(X, y, class_names):
    """
    Trains a LightGBM model for multi-class classification using sample weights
    to handle class imbalance.
    """
    num_classes = len(np.unique(y))
    print(f"Starting multi-class training for {num_classes} classes.")

    # 1. Divide the dataset
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )
    split_idx = int(0.8 * len(X))

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]

    y_train = y[:split_idx]
    y_test  = y[split_idx:]

    # 2. Calculate class weights
    class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights_array))

    # --- START OF FIX ---
    # Create a weights array for each sample in the training data
    # This maps the class weight to each sample's class label
    sample_weights = np.array([class_weight_dict[i] for i in y_train])
    # --- END OF FIX ---


    print(f"Class distribution in training set: {np.bincount(y_train)}")
    print(f"Computed class_weight dictionary: {class_weight_dict}")

    # 3. Build LightGBM dataset, now with sample weights
    # --- START OF FIX ---
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    # --- END OF FIX ---
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 4. Set parameters for multi-class classification
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',
        'num_leaves': 30,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    # 5. Train the model
    # Try GPU first, fallback to CPU if not available
    try:
        params_gpu = dict(params)
        params_gpu.update(tree_learner='serial', device_type='gpu')
        model = lgb.train(
            params_gpu,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )
    except Exception:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )

    # 6. Make predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 7. Evaluate the model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Create organized directory for LightGBM multiclass results
    lightgbm_multi_dir = create_model_result_dir('multiclass', 'LightGBM_Multi')
    
    # Save classification report as image
    save_classification_report_image(
        y_true=y_test,
        y_pred=y_pred,
        title='Classification Report (LightGBM Multi-class)',
        save_path=os.path.join(lightgbm_multi_dir, "classification_report.png"),
        target_names=class_names
    )

    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix')
    plt.savefig(os.path.join(lightgbm_multi_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"\nROC-AUC Score (One-vs-Rest): {roc_auc:.4f}")

    return model, X_train, X_test, y_train, y_test


def shap_plots_multiclass(model, X_train, class_names, top_n=30):
    """
    Generates SHAP plots, robustly handling list, 2D, or 3D SHAP value arrays.
    """
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    n_features = len(feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # --- START OF FIX ---
    # Robustly calculate mean_abs_shap to ensure it's always 1D
    if isinstance(shap_values, list):
        # Handles standard multi-class case (list of 2D arrays)
        mean_abs_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Handles modern multi-class case (single 3D array)
        # Average across samples (axis 0) and classes (axis 2)
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        # Handles binary classification or regression (single 2D array)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # --- END OF FIX ---

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_abs_shap})
    # Sort full importance
    importance_sorted = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    # Save all features global importance
    shap_multi_dir = create_model_result_dir('multiclass', 'LightGBM_Multi')
    try:
        with open(os.path.join(shap_multi_dir, "shap_top_features.json"), 'w', encoding='utf-8') as f:
            json.dump([
                {"feature": str(row.Feature), "importance": float(row.Importance)}
                for _, row in importance_sorted.iterrows()
            ], f, ensure_ascii=False, indent=2)
    except Exception as _e:
        print(f"Failed to write shap_top_features.json (multiclass): {_e}")

    # Compute cumulative importance threshold at 90%
    importance_vals = importance_sorted['Importance'].values
    total = importance_vals.sum() or 1.0
    cum = np.cumsum(importance_vals) / total
    cutoff_idx = int(np.searchsorted(cum, 0.99) + 1)
    print(f"Cumulative importance threshold at 99%: {cutoff_idx}")
    top_n_features = max(cutoff_idx, int(0.8 * n_features+1))
    top_n_features_df = importance_sorted.head(top_n_features)
    # Also save cum90 selection
    try:
        with open(os.path.join(shap_multi_dir, "shap_top_features_cum99.json"), 'w', encoding='utf-8') as f:
            json.dump([
                {"feature": str(row.Feature), "importance": float(row.Importance)}
                for _, row in top_n_features_df.iterrows()
            ], f, ensure_ascii=False, indent=2)
    except Exception as _e:
        print(f"Failed to write shap_top_features_cum99.json (multiclass): {_e}")

    # Derive display top-N from sorted data (for plots)
    available_features = len(importance_sorted)
    actual_top_n = min(top_n, available_features)
    top_n_df = importance_sorted.head(actual_top_n)

    print("="*80)
    print(f"Top {top_n} Features by Mean Absolute SHAP Importance")
    print("="*80)
    print(top_n_df.to_string(index=False))
    print("\n")

    # Create organized directory for multiclass SHAP results
    # shap_multi_dir already created above
    
    print("Generating SHAP summary plots...")
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar",
                      class_names=class_names, feature_names=feature_names,
                      max_display=top_n, show=False)
    plt.title("Overall SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_multi_dir, "shap_importance_bar.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Also make the per-class plotting robust
    if isinstance(shap_values, list):
        print("Generating per-class SHAP summary plots...")
        for i, class_name in enumerate(class_names):
            plt.figure()
            shap.summary_plot(shap_values[i], X_train, feature_names=feature_names, show=False, max_display=top_n)
            plt.title(f"SHAP Value Distribution for Class: {class_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_multi_dir, f'{class_name}_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
         print("Generating per-class SHAP summary plots...")
         for i, class_name in enumerate(class_names):
            # For 3D arrays, we slice along the class axis
            plt.figure()
            shap.summary_plot(shap_values[:, :, i], X_train, feature_names=feature_names, show=False, max_display=top_n)
            plt.title(f"SHAP Value Distribution for Class: {class_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_multi_dir, f'{class_name}_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
    else:
         print("Generating SHAP summary plot...")
         plt.figure()
         shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False, max_display=top_n)
         plt.title("SHAP Value Distribution")
         plt.tight_layout()
         plt.savefig(os.path.join(shap_multi_dir, f'{class_name}_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
         plt.show()
         plt.close()

    return explainer, shap_values, top_n_df, top_n_features_df


def create_dnn_model_multi(X_train_scaled, num_classes):
    """Helper function to create the DNN model for multi-class classification."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_mlp_model_multi(input_shape, num_classes):
    """Helper function to create the MLP model for multi-class classification."""
    model = Sequential([
        Dense(100, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def lime_analysis_for_samples(model, model_name, X_train, X_test, y_test, preds, class_names, feature_names, scaler=None, num_samples=3, selected_samples=None):
    """
    Performs LIME analysis for a few good and bad predictions.

    Args:
        model: The trained model.
        model_name (str): The name of the model ('MLP', 'DNN', etc.).
        X_train (pd.DataFrame): The original training data (unscaled).
        X_test (pd.DataFrame): The original test data (unscaled).
        y_test (np.array): True labels for the test set.
        preds (np.array): Predicted labels for the test set.
        class_names (list): List of class names.
        feature_names (list): List of feature names.
        scaler (StandardScaler, optional): The scaler used for models that need it.
        num_samples (int): The number of good/bad predictions to explain (only used when selected_samples is None).
        selected_samples (array-like, optional): Explicit indices to explain. If provided, LIME will use exactly these samples.
    """
    print(f"\n--- LIME Analysis for Individual Samples ({model_name}) ---")
    
    # Create organized directory for this model
    model_dir = create_model_result_dir('multiclass', model_name)

    # Determine if the model was trained on scaled data
    uses_scaled_data = model_name in ['MLP', 'DNN']

    # Define a prediction function that handles scaling if necessary
    if uses_scaled_data:
        # LIME generates perturbations around the original unscaled data.
        # This function scales them before feeding them to the model.
        def predict_fn(x):
            try:
                # ensure input is numpy array
                x = np.asarray(x)
                
                # if 1D array, convert to 2D
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                
                # standardize input
                x_scaled = scaler.transform(x)
                
                # get probability prediction
                proba = model.predict(x_scaled, verbose=0)
                
                # ensure correct shape is returned
                if proba.ndim == 1:
                    # if 1D, convert to 2D
                    proba = proba.reshape(1, -1)
                
                # ensure probability array column count equals number of classes
                if proba.shape[1] != len(class_names):
                    # if column count does not match, pad or truncate
                    if proba.shape[1] < len(class_names):
                        # pad with zero columns
                        padding = np.zeros((proba.shape[0], len(class_names) - proba.shape[1]))
                        proba = np.column_stack([proba, padding])
                    else:
                        # truncate excess columns
                        proba = proba[:, :len(class_names)]
                
                return proba
                
            except Exception as e:
                print(f"Predict function error: {e}")
                # return default probability distribution
                return np.ones((x.shape[0], len(class_names))) / len(class_names)
        # LIME explainer needs training data in the model's format (scaled)
        X_train_for_lime = scaler.transform(X_train)
    else:
        # For tree models, use predict_proba directly on original data
        predict_fn = model.predict_proba
        X_train_for_lime = X_train.values

    # Create the LIME Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_for_lime.astype('float64'),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    # Find indices for good and bad predictions
    correct_indices = np.where(preds == y_test)[0]
    incorrect_indices = np.where(preds != y_test)[0]

    if selected_samples is not None and len(selected_samples) > 0:
        # Use provided selected samples to ensure consistency with SHAP
        selected_samples = np.array(sorted(set(int(i) for i in selected_samples)), dtype=int)
        print(f"Using provided selected_samples for LIME: {selected_samples}")
    else:
        if len(correct_indices) == 0 and len(incorrect_indices) == 0:
            print("No samples found to analyze.")
            return
        if len(correct_indices) == 0:
            print("No correct predictions found to analyze.")
        if len(incorrect_indices) == 0:
            print("No incorrect predictions found in the test set.")
        # Select random samples to explain with consistent random seed
        np.random.seed(42)  # Set random seed for reproducibility
        samples_correct = np.random.choice(correct_indices, min(num_samples, len(correct_indices)), replace=False) if len(correct_indices) > 0 else np.array([], dtype=int)
        samples_incorrect = np.random.choice(incorrect_indices, min(num_samples, len(incorrect_indices)), replace=False) if len(incorrect_indices) > 0 else np.array([], dtype=int)
        # Combine and sort selected samples for consistency with SHAP
        selected_samples = np.concatenate([samples_correct, samples_incorrect])
        selected_samples = np.sort(selected_samples).astype(int)
        print(f"Selected samples for LIME (random due to no external selection): {selected_samples}")

    # Derive good/bad subsets from selected samples
    good_samples_indices = np.array([i for i in selected_samples if i in set(correct_indices)], dtype=int)
    bad_samples_indices = np.array([i for i in selected_samples if i in set(incorrect_indices)], dtype=int)


    # --- Explain Good Predictions ---
    print(f"\nAnalyzing {len(good_samples_indices)} good predictions... 🎯")
    for i in good_samples_indices:
        print(f"\nSample #{i}: Predicted '{class_names[preds[i]]}', Actual '{class_names[y_test[i]]}' (Correct)")
        # Explain the instance from the original, unscaled X_test
        instance = X_test.iloc[i].values
        explanation = explainer.explain_instance(instance, predict_fn, num_features=10, top_labels=2)
        explanation.save_to_file(os.path.join(model_dir, f'lime_explanation_{i}.html'))
        
        # Save LIME values as text for LLM
        try:
            # Get explanation data
            predicted_label = preds[i]
            lime_weights = explanation.as_list(label=predicted_label)
            
            # Create LIME explanation text
            # safely get feature values
            try:
                if hasattr(X_test, 'iloc'):
                    feature_values = X_test.iloc[i].to_string()
                else:
                    # if numpy array, format manually
                    feature_values = "\n".join([f"{feature_names[j]}: {X_test[i][j]:.4f}" for j in range(len(X_test[i]))])
            except Exception:
                feature_values = f"Feature values: {X_test[i]}"
            
            lime_text_content = f"""LIME Explanation for Sample {i}
Prediction: {class_names[preds[i]]} (Class {preds[i]})
True Label: {class_names[y_test[i]]} (Class {y_test[i]})

Feature Values:
{feature_values}

LIME Feature Importance (Top 10):
"""
            
            # Sort by absolute LIME weight and add to text
            sorted_weights = sorted(lime_weights, key=lambda x: abs(x[1]), reverse=True)
            for j, (feature_name, weight) in enumerate(sorted_weights[:10]):
                # safely handle weight, may be array or scalar
                try:
                    if hasattr(weight, 'item'):
                        weight_value = weight.item()
                    elif hasattr(weight, '__len__') and len(weight) > 0:
                        weight_value = weight[0]
                    else:
                        weight_value = float(weight)
                    lime_text_content += f"{j+1:2d}. {feature_name}: {weight_value:.4f}\n"
                except Exception:
                    lime_text_content += f"{j+1:2d}. {feature_name}: {weight}\n"
            
            # Save to text file
            lime_text_path = os.path.join(model_dir, f'lime_individual_{i}.txt')
            with open(lime_text_path, 'w', encoding='utf-8') as f:
                f.write(lime_text_content)
            print(f"    ✅ LIME text saved: {lime_text_path}")
            
        except Exception as text_e:
            print(f"    ⚠️  Failed to save LIME text for sample {i}: {text_e}")
        
        # Show explanation for the predicted class
        explanation.show_in_notebook(show_table=True)

    # --- Explain Bad Predictions ---
    if len(bad_samples_indices) > 0:
        print(f"\nAnalyzing {len(bad_samples_indices)} bad predictions... ❌")
        for i in bad_samples_indices:
            print(f"\nSample #{i}: Predicted '{class_names[preds[i]]}', Actual '{class_names[y_test[i]]}' (Incorrect)")
            instance = X_test.iloc[i].values
            explanation = explainer.explain_instance(instance, predict_fn, num_features=10, top_labels=2)
            explanation.save_to_file(os.path.join(model_dir, f'lime_explanation_{i}.html'))
            
            # Save LIME values as text for LLM
            try:
                # Get explanation data
                predicted_label = preds[i]
                lime_weights = explanation.as_list(label=predicted_label)
                
                # Create LIME explanation text
                # safely get feature values
                try:
                    if hasattr(X_test, 'iloc'):
                        feature_values = X_test.iloc[i].to_string()
                    else:
                        # if numpy array, format manually
                        feature_values = "\n".join([f"{feature_names[j]}: {X_test[i][j]:.4f}" for j in range(len(X_test[i]))])
                except Exception:
                    feature_values = f"Feature values: {X_test[i]}"
                
                lime_text_content = f"""LIME Explanation for Sample {i}
Prediction: {class_names[preds[i]]} (Class {preds[i]})
True Label: {class_names[y_test[i]]} (Class {y_test[i]})

Feature Values:
{feature_values}

LIME Feature Importance (Top 10):
"""
                
                # Sort by absolute LIME weight and add to text
                sorted_weights = sorted(lime_weights, key=lambda x: abs(x[1]), reverse=True)
                for j, (feature_name, weight) in enumerate(sorted_weights[:10]):
                    # safely handle weight, may be array or scalar
                    try:
                        if hasattr(weight, 'item'):
                            weight_value = weight.item()
                        elif hasattr(weight, '__len__') and len(weight) > 0:
                            weight_value = weight[0]
                        else:
                            weight_value = float(weight)
                        lime_text_content += f"{j+1:2d}. {feature_name}: {weight_value:.4f}\n"
                    except Exception:
                        lime_text_content += f"{j+1:2d}. {feature_name}: {weight}\n"
                
                # Save to text file
                lime_text_path = os.path.join(model_dir, f'lime_individual_{i}.txt')
                with open(lime_text_path, 'w', encoding='utf-8') as f:
                    f.write(lime_text_content)
                print(f"    ✅ LIME text saved: {lime_text_path}")
                
            except Exception as text_e:
                print(f"    ⚠️  Failed to save LIME text for sample {i}: {text_e}")
    

def train_and_explain_multi_models(X, y, class_names,top_n=10):
    """
    Trains, evaluates, and explains multiple models for multi-class classification,
    generating a per-class SHAP beeswarm plot for each model.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    feature_names = X.columns.tolist()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    split_idx = int(0.8 * len(X))

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]

    y_train = y[:split_idx]
    y_test  = y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use DataFrames for SHAP plotting to get proper feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    X_train_df= pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)


    model_configs = {
        'DecisionTree': {'model': DecisionTreeClassifier(random_state=42, class_weight='balanced')},
        'RandomForest': {'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')},
        'XGBoost': {'model': make_xgb_classifier('multi')},
        'MLP': {'model': 'create_mlp_model_multi'},
        'DNN': {'model': 'create_dnn_model_multi'}
    }

    models, results, top_features_indices = {}, {}, {}
    llm_bundle = {
        'task': 'multiclass',
        'label_mapping': {int(i): str(c) for i, c in enumerate(class_names)},
        'feature_names': feature_names,
        'models': {}
    }

    # compute class/sample weights for imbalance handling (multi)
    classes_unique = np.unique(y_train)
    class_weights_array = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)
    class_weight_dict = dict(zip(classes_unique, class_weights_array))
    sample_weight = np.array([class_weight_dict[c] for c in y_train])

    for name, config in model_configs.items():
        print(f"--- Processing {name} ---")
        model = None
        # Train models
        if name in ['MLP','DNN']:
            if name == 'MLP':
                model = create_mlp_model_multi(X_train_scaled.shape[1], len(class_names))
            else:
                model = create_dnn_model_multi(X_train_scaled, len(class_names))
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            # pass class_weight for imbalance in multiclass
            model.fit(X_train_scaled, y_train, epochs=100, batch_size=512, validation_split=0.1, callbacks=[early_stop], verbose=0, class_weight=class_weight_dict)
            preds_proba = model.predict(X_test_scaled, verbose=0)
            preds = np.argmax(preds_proba, axis=1)
        else:
            model = config['model']
            if name == 'XGBoost':
                # use sample_weight for multiclass imbalance
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train) # Tree models work better with original data
            preds = model.predict(X_test)

        models[name] = model
        accuracy = accuracy_score(y_test, preds)
        results[name] = accuracy
        # Create organized directory for this model
        model_dir = create_model_result_dir('multiclass', name)
        
        llm_bundle['models'][name] = {
            'accuracy': float(accuracy),
            'report_image': os.path.join(model_dir, "classification_report.png"),
            'confusion_image': os.path.join(model_dir, "confusion_matrix.png"),
            'lime_json': os.path.join(model_dir, f"{name}_lime_local_top10.json")
        }
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=class_names))
        
        # Save classification report as markdown text for LLM parsing
        try:
            report_txt = classification_report(y_test, preds, target_names=class_names, zero_division=0)
            with open(os.path.join(model_dir, "classification_report.md"), 'w', encoding='utf-8') as f:
                f.write("# Classification Report\n\n")
                f.write("````\n")
                f.write(report_txt)
                f.write("\n````\n")
        except Exception as _e:
            print(f"Failed to write classification_report.md: {_e}")

        # Save classification report as image
        save_classification_report_image(
            y_true=y_test,
            y_pred=preds,
            title=f'Classification Report ({name} Multi-class)',
            save_path=os.path.join(model_dir, "classification_report.png"),
            target_names=class_names
        )

        cm=confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix')
        plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Save confusion matrix values for LLM
        try:
            cm_json_path = os.path.join(model_dir, "confusion_matrix.json")
            with open(cm_json_path, 'w', encoding='utf-8') as f:
                json.dump({"matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"Failed to write confusion_matrix.json: {_e}")

        # === Precision-Recall (Multiclass) class-wise + macro/micro with Bootstrap CIs ===
        try:
            # obtain probabilities per class
            if name in ['MLP','DNN']:
                y_proba = model.predict(X_test_scaled, verbose=0)
            else:
                y_proba = model.predict_proba(X_test)
                if isinstance(y_proba, list):
                    # some classifiers return list of per-class arrays
                    y_proba = np.column_stack([p[:, 1] if p.ndim == 2 else p for p in y_proba])

            n_classes = len(class_names)
            Y_true_bin = label_binarize(y_test, classes=np.arange(n_classes))

            # per-class PR curves and AP with CI
            pr_dir = os.path.join(model_dir, 'pr_curves')
            os.makedirs(pr_dir, exist_ok=True)
            metrics_summary = {'per_class': {}, 'macro': {}, 'micro': {}}

            rng = np.random.default_rng(42)
            B = 200
            n = len(y_test)

            for k in range(n_classes):
                y_true_k = Y_true_bin[:, k]
                y_score_k = y_proba[:, k]
                P, R, _ = precision_recall_curve(y_true_k, y_score_k)
                ap_k = float(average_precision_score(y_true_k, y_score_k))

                boot_aps = []
                for _ in range(B):
                    idx = rng.integers(0, n, size=n)
                    boot_aps.append(average_precision_score(y_true_k[idx], y_score_k[idx]))
                low, high = np.percentile(boot_aps, [2.5, 97.5])

                plt.figure(figsize=(5,4))
                plt.plot(R, P, label=f'AUPRC={ap_k:.3f} [{low:.3f}, {high:.3f}]')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'PR Curve - Class: {class_names[k]}')
                plt.legend()
                plt.tight_layout()
                path_k = os.path.join(pr_dir, f'pr_curve_class_{k}.png')
                plt.savefig(path_k, dpi=300, bbox_inches='tight')
                plt.close()

                metrics_summary['per_class'][class_names[k]] = {
                    'auprc': ap_k,
                    'auprc_ci_95': [float(low), float(high)]
                }

            # macro-average (mean of per-class AP)
            macro_ap = float(np.mean([v['auprc'] for v in metrics_summary['per_class'].values()]))
            # bootstrap macro by resampling classes' APs may be misleading; instead resample samples and recompute avg
            macro_boot = []
            for _ in range(B):
                idx = rng.integers(0, n, size=n)
                ap_list = []
                for k in range(n_classes):
                    ap_list.append(average_precision_score(Y_true_bin[idx, k], y_proba[idx, k]))
                macro_boot.append(np.mean(ap_list))
            macro_low, macro_high = np.percentile(macro_boot, [2.5, 97.5])
            metrics_summary['macro'] = {
                'auprc': macro_ap,
                'auprc_ci_95': [float(macro_low), float(macro_high)]
            }

            # micro-average (flattened one-vs-rest)
            micro_ap = float(average_precision_score(Y_true_bin.ravel(), y_proba.ravel()))
            micro_boot = []
            for _ in range(B):
                idx = rng.integers(0, n, size=n)
                micro_boot.append(average_precision_score(Y_true_bin[idx].ravel(), y_proba[idx].ravel()))
            micro_low, micro_high = np.percentile(micro_boot, [2.5, 97.5])
            metrics_summary['micro'] = {
                'auprc': micro_ap,
                'auprc_ci_95': [float(micro_low), float(micro_high)]
            }

            with open(os.path.join(model_dir, 'pr_metrics_multiclass.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"Failed PR/AUPRC (multiclass) for {name}: {_e}")



        # --- SHAP Value Calculation ---
        explainer, shap_values = None, None
        # We use a smaller sample for explainers to keep it fast
        X_test_sample = X_test_df.head(1000)
        X_test_scaled_sample = X_test_scaled_df.head(1000)
        
        # Dynamic background data size based on dataset size
        train_size = len(X_train_scaled_df)
        if train_size <= 1000:
            background_size = min(100, train_size)  # Use up to 100 samples for small datasets
        elif train_size <= 10000:
            background_size = min(500, train_size)  # Use up to 500 samples for medium datasets
        else:
            background_size = min(1000, train_size)  # Use up to 1000 samples for large datasets
        
        background_data = shap.sample(X_train_scaled_df, background_size).values.astype('float32')
        print(f"Using {background_size} samples as background data (dataset size: {train_size})")
        
        # Update LLM bundle with SHAP-related paths now that X_test_sample is defined
        # We'll update this after we know the actual selected samples
        llm_bundle['models'][name].update({
            'shap_importance_bar': os.path.join(model_dir, 'shap_importance_bar.png')
        })

        if name in ['DecisionTree', 'RandomForest', 'XGBoost']:
            try:
                explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
                shap_values = explainer.shap_values(X_test_sample)
            except Exception:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
            # Set shap_explainer for individual evaluation
            shap_explainer = explainer
        elif name == 'MLP':
            # ensure background_data format is correct
            shap_explainer = shap.GradientExplainer(model, background_data)
            # ensure test data format is correct
            test_data = X_test_scaled_sample.values.astype('float32')
            shap_values = shap_explainer.shap_values(test_data)
            # shap_explainer is already set above
        elif name == 'DNN':
            # ensure background_data format is correct
            shap_explainer = shap.DeepExplainer(model, background_data)
            # ensure test data format is correct
            test_data = X_test_scaled_sample.values.astype('float32')
            shap_values = shap_explainer.shap_values(test_data)
            # shap_explainer is already set above

        # --- Create LIME Explainer for Individual Evaluation ---
        # Determine training data for LIME based on model type
        if name in ['MLP', 'DNN']:
            # For neural networks, use scaled data - ensure numeric type
            X_train_for_lime = X_train_scaled_df.values.astype('float64')
        else:
            # For tree models, use original data - ensure numeric type
            X_train_for_lime = X_train_df.values.astype('float64')
        
        # Create LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_for_lime,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

        # --- SHAP Importance Bar Plot (Overall) ---
        print(f"Generating SHAP importance bar plot for {name}...")
        if shap_values is not None:
            # Calculate overall feature importance across all classes
            if isinstance(shap_values, list):
                # For list-based SHAP values (most common case)
                mean_abs_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # For 3D array SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                # For 2D array SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Create importance bar plot
            plt.figure(figsize=(10, 6))
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_abs_shap
            })
            # Handle case where features < top_n
            available_features = len(feature_importance_df)
            actual_top_n = min(top_n, available_features)
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True).tail(actual_top_n)
            
            plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
            plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
            plt.xlabel('Mean Absolute SHAP Value')
            plt.title(f'SHAP Feature Importance - {name} (Multi-class)')
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'shap_importance_bar.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            
            # Update LLM bundle with SHAP importance bar plot
            llm_bundle['models'][name]['shap_importance_bar'] = os.path.join(model_dir, 'shap_importance_bar.png')

        # --- Per-Class SHAP Beeswarm Plotting ---
        print(f"Generating per-class SHAP beeswarm plots for {name}...")
        if shap_values is not None:
            # This block handles both list-based and 3D array outputs from SHAP explainers
            if isinstance(shap_values, list): # For models like RF, XGB, MLP, DNN
                 for i, class_name in enumerate(class_names):
                     # The plot_type='dot' is the beeswarm plot
                     if name in ['DecisionTree', 'RandomForest', 'XGBoost']:
                          plt.figure()
                          shap.summary_plot(shap_values[i], X_test_sample, feature_names=feature_names, show=False, max_display=top_n, plot_type='dot')
                     else:
                          plt.figure()
                          shap.summary_plot(shap_values[i], X_test_scaled_sample, feature_names=feature_names, show=False, max_display=top_n, plot_type='dot')
                     plt.title(f"SHAP Feature Importance for Class: '{class_name}' ({name})")
                     plt.tight_layout()
                     plt.savefig(os.path.join(model_dir, f'{class_name}_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
                     plt.show()
                     plt.close()
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3: # Also possible for some explainers
                  for i, class_name in enumerate(class_names):
                      if name in ['DecisionTree', 'RandomForest', 'XGBoost']:
                           plt.figure()
                           shap.summary_plot(shap_values[:, :, i], X_test_sample, feature_names=feature_names, show=False, max_display=top_n, plot_type='dot')
                      else:
                           plt.figure()
                           shap.summary_plot(shap_values[:, :, i], X_test_scaled_sample, feature_names=feature_names, show=False, max_display=top_n, plot_type='dot')
                      plt.title(f"SHAP Feature Importance for Class: '{class_name}' ({name})")
                      plt.tight_layout()
                      plt.savefig(os.path.join(model_dir, f'{class_name}_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
                      plt.show()
                      plt.close()
                      # Save path for LLM bundle
                      llm_bundle['models'][name][f'shap_beeswarm_{class_name}'] = os.path.join(model_dir, f'{class_name}_shap_beeswarm.png')
            else: # Fallback for binary or single-output cases
                  plt.figure()
                  shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False, max_display=top_n, plot_type='dot')
                  plt.title(f"SHAP Feature Importance ({name})")
                  plt.tight_layout()
                  plt.show()
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0) 
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
            else:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            # abs_shap_values = [np.abs(sv) for sv in shap_values]
            # mean_abs_shap_across_classes = np.mean(abs_shap_values, axis=0)
            # mean_abs_shap = np.mean(mean_abs_shap_across_classes, axis=0)
            # Handle case where features < top_n
            available_features = len(mean_abs_shap)
            print(f"Available features: {available_features}")
            actual_top_n = min(top_n, available_features)
            top_features_indices[name] = np.argsort(mean_abs_shap)[-actual_top_n:][::-1]
            
            # Save SHAP top features (feature -> importance)
            try:
                # Handle case where features < 50
                available_features = len(mean_abs_shap)
                actual_top_50 = min(50, available_features)
                top_idx = np.argsort(mean_abs_shap)[-actual_top_50:][::-1]
                shap_top = [
                    {"feature": str(feature_names[i]) if i < len(feature_names) else str(i), "importance": float(mean_abs_shap[i])}
                    for i in top_idx
                ]
                with open(os.path.join(model_dir, "shap_top_features.json"), 'w', encoding='utf-8') as f:
                    json.dump(shap_top, f, ensure_ascii=False, indent=2)
            except Exception as _e:
                print(f"Failed to write shap_top_features.json: {_e}")

        # --- XAI Quality Assessment (Multiclass SHAP-Local) ---
        try:
            # convert multiclass SHAP to sample×feature 2D array
            shap_for_eval = None
            if shap_values is None:
                shap_for_eval = None
            elif isinstance(shap_values, list):
                # average absolute values across classes
                shap_for_eval = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_for_eval = np.abs(shap_values).mean(axis=2)
            else:
                shap_for_eval = shap_values

        except Exception as _e:
            print(f"XAI evaluation failed (multiclass) {name}: {_e}")

        # --- XAI Quality Assessment (Multiclass SHAP-Global, via sample/class aggregation) ---
        try:
            if shap_values is not None:
                if isinstance(shap_values, list):
                    shap_for_eval = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_for_eval = np.abs(shap_values).mean(axis=2)
                else:
                    shap_for_eval = np.abs(shap_values)

                global_importance = np.mean(np.abs(shap_for_eval), axis=0)
                fi_global = {int(idx): float(val) for idx, val in enumerate(global_importance)}

                _evaluate_and_save(
                    model_name=name,
                    model=model,
                    X_test_df=X_test,
                    y_test=y_test,
                    shap_values_for_plot=None,
                    scaler=scaler,
                    uses_scaled=(name in ['MLP','DNN']),
                    class_names=class_names,
                    explanations=None,
                    method_name="shap_global",
                    task_type="multiclass",
                    explainer=shap_explainer,
                    global_importance=fi_global
                )
        except Exception as _e:
            print(f"XAI evaluation failed (multiclass SHAP Global) {name}: {_e}")

        # --- XAI Quality Assessment (Multiclass LIME-Local) ---
        try:
            lime_explanations = []
            available_n = X_test_sample.shape[0]
            sel_idx = _select_sample_indices(y_test, available_n, EVAL_SAMPLE_SIZE, stratified=EVAL_STRATIFIED)
            
            # prepare appropriate predict_fn for MLP/DNN
            if name in ['MLP', 'DNN']:
                # neural network models need scaled data
                def predict_fn_nn(x):
                    x = np.asarray(x)
                    if x.ndim == 1:
                        x = x.reshape(1, -1)
                    # use scaler for transformation
                    x_scaled = scaler.transform(x)
                    proba = model.predict(x_scaled, verbose=0)
                    if proba.ndim == 1:
                        proba = proba.reshape(1, -1)
                    # ensure column count matches number of classes
                    if proba.shape[1] != len(class_names):
                        if proba.shape[1] < len(class_names):
                            padding = np.zeros((proba.shape[0], len(class_names) - proba.shape[1]))
                            proba = np.column_stack([proba, padding])
                        else:
                            proba = proba[:, :len(class_names)]
                    return proba
                predict_fn_to_use = predict_fn_nn
                # for neural networks, LIME uses original unscaled training data
                X_train_for_lime = X_train_df.values.astype('float64')
            else:
                # tree models use predict_proba directly
                def predict_fn_tree(x):
                    x = np.asarray(x)
                    if x.ndim == 1:
                        x = x.reshape(1, -1)
                    proba = model.predict_proba(x)
                    if proba.ndim == 1:
                        proba = proba.reshape(1, -1)
                    return proba
                predict_fn_to_use = predict_fn_tree
                X_train_for_lime = X_train_df.values.astype('float64')
            
            for i in sel_idx:
                instance = X_test_sample.iloc[i].values if hasattr(X_test_sample, 'iloc') else X_test_sample[i]
                
                # modified: generate explanations for all features, not just top-10
                exp = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_train_for_lime,
                    feature_names=feature_names,
                    class_names=class_names,
                    mode='classification'
                ).explain_instance(instance, predict_fn_to_use, num_features=len(instance))
                weights = exp.as_list()
                fi = {}
                for feat, w in weights:
                    try:
                        idx = int(str(feat).split()[0].replace('feature_', '').replace('f', ''))
                    except Exception:
                        if hasattr(X_test_sample, 'columns') and str(feat) in X_test_sample.columns:
                            idx = int(list(X_test_sample.columns).index(str(feat)))
                        else:
                            continue
                    fi[int(idx)] = float(w)
                
                # assign 0 importance to missing features, ensure all features are included
                for idx in range(len(instance)):
                    if idx not in fi:
                        fi[idx] = 0.0
                lime_explanations.append({'feature_importance': fi, 'text_explanation': 'LIME local explanation', 'rules': []})


        except Exception as _e:
            print(f"XAI evaluation failed (multiclass LIME) {name}: {_e}")

                # --- Individual Sample Evaluation for Selected Samples (Multiclass) ---
        print(f"\n--- Individual Sample Evaluation for {name} (Multiclass) ---")
        
        # Select samples for individual evaluation
        correct_indices = np.where(preds == y_test)[0]
        incorrect_indices = np.where(preds != y_test)[0]
        np.random.seed(42)  # For reproducibility
        samples_correct = np.random.choice(correct_indices, min(3, len(correct_indices)), replace=False) if len(correct_indices) > 0 else np.array([], dtype=int)
        samples_incorrect = np.random.choice(incorrect_indices, min(3, len(incorrect_indices)), replace=False) if len(incorrect_indices) > 0 else np.array([], dtype=int)
        selected_samples= np.concatenate([samples_correct, samples_incorrect])
        selected_samples = np.sort(selected_samples).astype(int)
        
        print(f"  📊 Selected samples: {selected_samples}")
        
        # --- Individual Sample SHAP Waterfall Plots and JSON ---
        print(f"Generating individual SHAP waterfall plots and JSON for {name}...")
        if shap_explainer is not None:
            try:
                # Use the same selected samples as LIME for consistency
                if 'selected_samples' in locals() and len(selected_samples) > 0:
                    selected_indices = np.asarray(selected_samples, dtype=int)
                    print(f"Using LIME selected samples for SHAP: {selected_indices}")
                else:
                    # Fallback: Select samples consistent with LIME: 3 correct + 3 incorrect predictions
                    correct_indices = np.where(preds == y_test)[0]
                    incorrect_indices = np.where(preds != y_test)[0]
                    np.random.seed(42)
                    samples_correct = np.random.choice(correct_indices, min(3, len(correct_indices)), replace=False) if len(correct_indices) > 0 else np.array([], dtype=int)
                    samples_incorrect = np.random.choice(incorrect_indices, min(3, len(incorrect_indices)), replace=False) if len(incorrect_indices) > 0 else np.array([], dtype=int)
                    selected_indices = np.sort(np.concatenate([samples_correct, samples_incorrect])).astype(int)
                    print(f"Using fallback sample selection for SHAP: {selected_indices}")
                
                # Calculate correct/incorrect counts for display
                correct_count = len([idx for idx in selected_indices if preds[idx] == y_test[idx]])
                incorrect_count = len(selected_indices) - correct_count
                print(f"Selected samples for SHAP waterfall: {correct_count} correct, {incorrect_count} incorrect")
                print(f"Sample indices: {selected_indices}")
                
                # Determine the correct test data based on model type (align with LIME)
                if name in ['MLP','DNN']:
                    _test_data_for_waterfall = X_test_scaled_df
                else:
                    _test_data_for_waterfall = X_test_df
                
                # Guard against out-of-range indices
                selected_indices = np.asarray([int(i) for i in selected_indices if int(i) < len(_test_data_for_waterfall)], dtype=int)

                for idx in selected_indices:
                    try:
                        # Get the sample data (align with LIME-selected dataset)
                        instance = _test_data_for_waterfall.iloc[idx:idx+1]
                        
                        # Compute SHAP values on-demand for this instance to avoid index mismatch
                        predicted_class = preds[idx]
                        base_value = 0
                        try:
                            if name in ['MLP', 'DNN']:
                                # For neural networks, use numpy arrays with correct dtype
                                instance_data = instance.values.astype('float32')
                                shap_values_instance_all = shap_explainer.shap_values(instance_data)
                                
                                # Handle DNN special case: convert tensor to numpy if needed
                                if hasattr(shap_values_instance_all, 'numpy'):
                                    shap_values_instance_all = shap_values_instance_all.numpy()
                                
                                # Handle scalar tensor case for DNN
                                if hasattr(shap_values_instance_all, 'shape') and shap_values_instance_all.shape == ():
                                    print(f"    ⚠️  DNN returned scalar tensor for sample {idx}, skipping...")
                                    continue
                                    
                            else:
                                # For tree-based models, pass DataFrame row
                                shap_values_instance_all = shap_explainer.shap_values(instance)
                            
                            # Handle multi-class outputs
                            if isinstance(shap_values_instance_all, list):
                                if predicted_class < len(shap_values_instance_all):
                                    # Handle tensor conversion for list elements
                                    shap_vals = shap_values_instance_all[predicted_class]
                                    if hasattr(shap_vals, 'numpy'):
                                        shap_vals = shap_vals.numpy()
                                    if hasattr(shap_vals, 'shape') and len(shap_vals.shape) > 0:
                                        shap_values_instance = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                                    else:
                                        print(f"    ⚠️  Invalid SHAP shape for sample {idx}, skipping...")
                                        continue
                                else:
                                    print(f"    ⚠️  Predicted class out of range for sample {idx}, skipping...")
                                    continue
                                if hasattr(shap_explainer, 'expected_value'):
                                    ev = shap_explainer.expected_value
                                    if hasattr(ev, 'numpy'):
                                        ev = ev.numpy()
                                    if isinstance(ev, (list, np.ndarray)) and predicted_class < len(ev):
                                        base_value = ev[predicted_class]
                                    else:
                                        base_value = ev if not isinstance(ev, (list, np.ndarray)) else 0
                            elif isinstance(shap_values_instance_all, np.ndarray) and shap_values_instance_all.ndim == 3:
                                # shape: (n_samples, n_features, n_classes)
                                shap_values_instance = shap_values_instance_all[0, :, predicted_class]
                                if hasattr(shap_explainer, 'expected_value'):
                                    ev = shap_explainer.expected_value
                                    if hasattr(ev, 'numpy'):
                                        ev = ev.numpy()
                                    if isinstance(ev, (list, np.ndarray)) and predicted_class < len(ev):
                                        base_value = ev[predicted_class]
                                    else:
                                        base_value = ev if not isinstance(ev, (list, np.ndarray)) else 0
                            else:
                                # binary/regression 2D output
                                if hasattr(shap_values_instance_all, 'numpy'):
                                    shap_values_instance_all = shap_values_instance_all.numpy()
                                if hasattr(shap_values_instance_all, 'shape') and len(shap_values_instance_all.shape) > 0:
                                    shap_values_instance = shap_values_instance_all[0]
                                else:
                                    print(f"    ⚠️  Invalid SHAP shape for sample {idx}, skipping...")
                                    continue
                                if hasattr(shap_explainer, 'expected_value') and not isinstance(shap_explainer.expected_value, (list, np.ndarray)):
                                    ev = shap_explainer.expected_value
                                    if hasattr(ev, 'numpy'):
                                        ev = ev.numpy()
                                    base_value = ev
                        except Exception as _sv_e:
                            print(f"    ⚠️  Failed to compute SHAP for sample {idx}: {_sv_e}")
                            continue
                        
                        # Create Explanation object for waterfall plot
                        try:
                            # Get feature values for this instance (use same data source as instance)
                            feature_values = _test_data_for_waterfall.iloc[idx].values
                            
                            # Create Explanation object
                            explanation_obj = shap.Explanation(
                                values=shap_values_instance,
                                base_values=base_value,
                                data=feature_values,
                                feature_names=feature_names
                            )
                            
                            # Create waterfall plot
                            plt.figure(figsize=(12, 8))
                            shap.waterfall_plot(explanation_obj, show=False)
                            
                        except Exception as waterfall_e:
                            print(f"    ⚠️  Waterfall plot failed for sample {idx}: {waterfall_e}")
                            plt.close()
                            continue
                        
                        # Determine if prediction is correct
                        is_correct = preds[idx] == y_test[idx]
                        prediction_status = "Correct" if is_correct else "Incorrect"
                        
                        plt.title(f'SHAP Waterfall Plot - Sample {idx} ({name} Multi-class) - {prediction_status}')
                        plt.tight_layout()
                        waterfall_path = os.path.join(model_dir, f'shap_waterfall_{idx}.png')
                        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                        plt.show()
                        plt.close()
                        
                        # Save SHAP values as JSON for LLM
                        try:
                            
                            shap_data = {
                                "sample_index": int(idx),
                                "base_value": float(base_value),
                                "feature_values": instance.iloc[0].to_dict() if hasattr(instance, 'iloc') else instance[0].tolist(),
                                "shap_values": shap_values_instance.tolist(),
                                "feature_names": feature_names,
                                "prediction": int(preds[idx]),
                                "true_label": int(y_test[idx]),
                                "predicted_class_name": class_names[preds[idx]],
                                "true_class_name": class_names[y_test[idx]],
                                "prediction_correct": bool(is_correct),
                                "prediction_status": "Correct" if is_correct else "Incorrect"
                            }
                            
                            # Create feature importance list
                            feature_importance = []
                            for i, (feature_name, shap_value) in enumerate(zip(feature_names, shap_values_instance)):
                                feature_importance.append({
                                    "feature": feature_name,
                                    "shap_value": float(shap_value),
                                    "feature_value": float(instance.iloc[0][feature_name]) if hasattr(instance, 'iloc') else float(instance[0][i])
                                })
                            
                            # Sort by absolute SHAP value
                            feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                            shap_data["feature_importance"] = feature_importance
                            
                            # Save to JSON file
                            shap_json_path = os.path.join(model_dir, f'shap_individual_{idx}.json')
                            with open(shap_json_path, 'w', encoding='utf-8') as f:
                                import json as _json
                                _json.dump(shap_data, f, ensure_ascii=False, indent=2)
                            print(f"    ✅ SHAP JSON saved: {shap_json_path}")
                            
                        except Exception as json_e:
                            print(f"    ⚠️  Failed to save SHAP JSON for index {idx}: {json_e}")
                    
                    except Exception as e:
                        print(f"    ⚠️  Failed to generate waterfall plot for sample {idx}: {e}")
                        continue
                        
            except Exception as e:
                print(f"    ⚠️  Failed to generate individual SHAP plots: {e}")
            
            # Update LLM bundle with actual generated files
            if 'selected_indices' in locals() and len(selected_indices) > 0:
                llm_bundle['models'][name].update({
                    'shap_individual_json': [os.path.join(model_dir, f'shap_individual_{idx}.json') for idx in selected_indices],
                    'shap_waterfall_plots': [os.path.join(model_dir, f'shap_waterfall_{idx}.png') for idx in selected_indices]
                })
        
        # Save aggregated LIME top features for multiclass
        try:
            # Define a prediction function that returns probability for class index when possible
            def _predict_proba_fn(X_np):
                try:
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(X_np)
                    # Keras models: use predict; ensure 2D probs
                    probs = model.predict(X_np, verbose=0)
                    probs = np.asarray(probs)
                    if probs.ndim == 1:
                        probs = np.vstack([1 - probs, probs]).T
                    return probs
                except Exception:
                    # Fallback to decision function or predict
                    if hasattr(model, 'decision_function'):
                        df = model.decision_function(X_np)
                        # convert to pseudo-probabilities
                        from scipy.special import expit
                        if df.ndim == 1:
                            p1 = expit(df)
                            return np.vstack([1 - p1, p1]).T
                        return df
                    preds_bin = model.predict(X_np)
                    return np.vstack([1 - preds_bin, preds_bin]).T

            agg_importance = {}
            sample_count = min(1000, len(X_test_df))
            X_lime = X_test_df.head(sample_count).values
            # For multiclass, focus on the most important class (highest prediction probability)
            for i in range(X_lime.shape[0]):
                exp = lime_explainer.explain_instance(
                    X_lime[i],
                    lambda x: _predict_proba_fn(np.asarray(x)),
                    num_features=10,
                    top_labels=1
                )
                # Get explanation for the top predicted class
                try:
                    label = exp.top_labels[0]
                    for feat_idx, weight in exp.as_map()[label]:
                        feat_name = str(feature_names[feat_idx]) if feat_idx < len(feature_names) else str(feat_idx)
                        agg_importance[feat_name] = agg_importance.get(feat_name, 0.0) + float(abs(weight))
                except Exception:
                    continue

            if agg_importance:
                # Normalize and sort
                total = sum(agg_importance.values()) or 1.0
                lime_top = [
                    {"feature": k, "importance": v / total}
                    for k, v in sorted(agg_importance.items(), key=lambda kv: kv[1], reverse=True)[:50]
                ]
                with open(os.path.join(model_dir, "lime_top_features.json"), 'w', encoding='utf-8') as f:
                    json.dump(lime_top, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"Failed to write lime_top_features.json: {_e}")
        

        
        # Evaluate SHAP-Local for selected samples
        try:
            # Determine the correct test data based on model type
            if name in ['MLP','DNN']:
                test_data = X_test_scaled_df
            else:
                test_data = X_test_df
            _evaluate_specific_samples(
                model_name=name,
                model=model,
                X_test_df=test_data,
                y_test=y_test,
                preds=preds,
                selected_samples=selected_samples,
                method_name="shap_local",
                task_type="multiclass",
                shap_explainer=shap_explainer,
                scaler=scaler,
                uses_scaled=(name in ['MLP','DNN']),
                class_names=class_names
            )
        except Exception as e:
            print(f"❌ Individual SHAP evaluation failed for {name}: {e}")
        
        # Evaluate LIME-Local for selected samples
        try:
            # Determine the correct test data based on model type
            if name in ['MLP','DNN']:
                test_data = X_test_scaled_df
            else:
                test_data = X_test_df
            print(f"  📊 Using same selected samples for LIME: {selected_samples}")
            _evaluate_specific_samples(
                model_name=name,
                model=model,
                X_test_df=test_data,
                y_test=y_test,
                preds=preds,
                selected_samples=selected_samples,
                method_name="lime_local",
                task_type="multiclass",
                lime_explainer=lime_explainer,
                scaler=scaler,
                uses_scaled=(name in ['MLP','DNN']),
                class_names=class_names
            )
        except Exception as e:
            print(f"❌ Individual LIME evaluation failed for {name}: {e}")
        
        # Save LIME local top10 JSON for LLM - now in results directory
        try:
            if 'lime_explanations' in locals() and len(lime_explanations) > 0:
                lime_json = []
                for i, ex in enumerate(lime_explanations):
                    sorted_items = sorted(ex['feature_importance'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
                    lime_json.append({
                        'sample_index': int(i),
                        'top_features': [{'index': int(k), 'weight': float(v)} for k, v in sorted_items]
                    })
                # Save in results directory instead of separate LLM directory
                lime_json_path = os.path.join(RESULT_DIR, 'multiclass', name, f'{name}_lime_local_top10.json')
                with open(lime_json_path, 'w', encoding='utf-8') as f:
                    import json as _json
                    _json.dump(lime_json, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # note: LIME does not provide strict global explanations, no LIME-Global evaluation here

        # --- LIME ANALYSIS CALL ---
        # Pass the same selected_samples to LIME for consistency
        lime_analysis_for_samples(
            model=model,
            model_name=name,
            X_train=X_train_df,      # Pass original unscaled training data
            X_test=X_test_df,        # Pass original unscaled test data
            y_test=y_test,
            preds=preds,             # Use predictions on the full test set
            class_names=class_names,
            feature_names=feature_names,
            scaler=scaler,           # Pass the scaler for models that need it
            num_samples=3,
            selected_samples=selected_samples
        )
        print(f"\n{'='*20} Finished {name} {'='*20}")



    # --- Final Results ---
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
    print("\n--- 📊 Model Performance ---")
    print(results_df.sort_values(by='Accuracy', ascending=False))

    top_features_names = {name: [feature_names[i] for i in indices] for name, indices in top_features_indices.items()}
    top_features_df = pd.DataFrame({k: pd.Series(v) for k, v in top_features_names.items()})
    if not top_features_df.empty:
        top_features_df.index = [f"Top {i+1}" for i in range(len(top_features_df))]

    print("\n--- ✨ Top 10 Features Per Model (Aggregated) ---")
    print(top_features_df)

    valid_feature_lists = [v for v in top_features_names.values() if v]
    common_features = set.intersection(*(set(features) for features in valid_feature_lists)) if valid_feature_lists else set()
    print(f"\n--- 🤝 Common Top Features Across All Models ---\n{common_features if common_features else 'None'}")

    # Save to text files in results directory (multiclass)
    try:
        multi_top_path = os.path.join(RESULT_DIR, 'multiclass_top_features_per_model.txt')
        with open(multi_top_path, 'w', encoding='utf-8') as f:
            f.write("Top 10 Features Per Model (Multiclass)\n")
            if not top_features_df.empty:
                f.write(top_features_df.to_string())
            else:
                f.write("<empty>\n")
        multi_common_path = os.path.join(RESULT_DIR, 'multiclass_common_top_features.txt')
        with open(multi_common_path, 'w', encoding='utf-8') as f:
            f.write("Common Top 10 Features Across All Models (Multiclass)\n")
            if common_features:
                for feat in sorted(list(common_features)):
                    f.write(f"{feat}\n")
            else:
                f.write("None\n")
    except Exception:
        pass

    # Save global info for LLM - now in results directory
    try:
        import json as _json
        # Save bundle in results directory
        with open(os.path.join(RESULT_DIR, 'multiclass', 'llm_bundle.json'), 'w', encoding='utf-8') as f:
            _json.dump(llm_bundle, f, ensure_ascii=False, indent=2)
        # Save top-features file paths
        with open(os.path.join(RESULT_DIR, 'multiclass', 'top_features_files.json'), 'w', encoding='utf-8') as f:
            _json.dump({
                'per_model': f"{RESULT_DIR}/multiclass_top_features_per_model.txt",
                'common': f"{RESULT_DIR}/multiclass_common_top_features.txt"
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return models, results_df, top_features_df, common_features



# ========================
# LLM Explanation Generation
# ========================

def generate_llm_explanations(api_key: str = None, enable_llm: bool = True):
    """
    Generate LLM explanations for all XAI results using Deepseek API.
    Saves explanations to each model's LLM folder for better organization.
    
    Args:
        api_key: Deepseek API key (if None, will try to get from environment)
        enable_llm: Whether to enable LLM explanation generation
    """
    if not enable_llm:
        print("LLM explanation generation is disabled.")
        return
    
    # Get API key
    if api_key is None:
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            print("Warning: DEEPSEEK_API_KEY not found. LLM explanations will be skipped.")
            print("To enable LLM explanations, set the DEEPSEEK_API_KEY environment variable.")
            return
    
    print("\n" + "="*60)
    print("🤖 Starting LLM Explanation Generation")
    print("="*60)
    
    try:
        # Initialize the LLM explainer
        explainer = DeepseekLLMExplainer(api_key)
        
        # Process binary classification results
        binary_results_dir = os.path.join(RESULT_DIR, 'binary')
        if os.path.exists(binary_results_dir):
            print("\n📊 Processing Binary Classification Results...")
            _process_task_type_llm(explainer, binary_results_dir, 'binary')
        else:
            print("⚠️  Binary results directory not found, skipping...")
        
        # Process multiclass classification results
        multiclass_results_dir = os.path.join(RESULT_DIR, 'multiclass')
        if os.path.exists(multiclass_results_dir):
            print("\n📊 Processing Multiclass Classification Results...")
            _process_task_type_llm(explainer, multiclass_results_dir, 'multiclass')
        else:
            print("⚠️  Multiclass results directory not found, skipping...")
        
        # Display API usage statistics
        stats = explainer.get_api_statistics()
        print(f"\n📊 API Usage Statistics:")
        print(f"  • Total API calls: {stats['total_calls']}")
        print(f"  • Successful calls: {stats['successful_calls']}")
        print(f"  • Failed calls: {stats['failed_calls']}")
        print(f"  • Success rate: {stats['success_rate']:.2%}")
        print(f"  • Total tokens used: {stats['total_tokens']}")
        print(f"  • Average tokens per call: {stats['average_tokens_per_call']:.1f}")
        
        print("\n🎉 LLM Explanation Generation Completed Successfully!")
        print("\nGenerated explanations include:")
        print("  • Model performance analysis")
        print("  • Feature importance interpretation")
        print("  • Feature importance explanations (SHAP, LIME)")
        print("  • Overall summaries and recommendations")
        print("  • Technical insights for cybersecurity professionals")
        print("  • Individual sample explanations")
        
    except Exception as e:
        print(f"❌ Error during LLM explanation generation: {e}")
        print("Please check your API key and network connection.")
        import traceback
        traceback.print_exc()

def _process_task_type_llm(explainer, results_dir: str, task_type: str):
    """
    Process LLM explanations for a specific task type and save to each model's LLM folder
    
    Args:
        explainer: LLM explainer instance
        results_dir: Results directory path
        task_type: Task type ('binary' or 'multiclass')
    """
    try:
        # Get all model directories
        model_dirs = [d for d in os.listdir(results_dir) 
                     if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith('.')]
        
        for model_name in model_dirs:
            model_dir = os.path.join(results_dir, model_name)
            print(f"\n  🔍 Processing model: {model_name}")
            
            # Create LLM directory for this model
            llm_dir = os.path.join(model_dir, 'LLM')
            os.makedirs(llm_dir, exist_ok=True)
            
            # Generate global explanation
            print(f"    📝 Generating global explanation...")
            global_explanation = explainer.generate_global_explanation(task_type, model_name, model_dir)
            
            if global_explanation:
                # Save global explanation
                global_file = os.path.join(llm_dir, 'global_explanation.md')
                with open(global_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {model_name} Global Explanation\n\n")
                    f.write(f"**Task Type**: {task_type}\n")
                    f.write(f"**Generated**: {global_explanation['timestamp']}\n\n")
                    f.write("## Analysis\n\n")
                    f.write(global_explanation['content'])
                
                print(f"    ✅ Global explanation saved to: {global_file}")
            else:
                print(f"    ❌ Failed to generate global explanation for {model_name}")
            
            # Generate individual explanations
            print(f"    📝 Generating individual explanations...")
            sample_data_list = explainer._get_sample_data(model_dir, task_type, num_samples=6)
            
            individual_explanations = []
            for sample_data in sample_data_list:
                individual_explanation = explainer.generate_individual_explanation(sample_data, model_name, task_type)
                if individual_explanation:
                    individual_explanations.append(individual_explanation)
                    print(f"    ✅ Individual explanation generated for sample {sample_data['sample_id']}")
                else:
                    print(f"    ❌ Failed to generate individual explanation for sample {sample_data['sample_id']}")
            
            if individual_explanations:
                # Save individual explanations - one file per sample
                samples_dir = os.path.join(llm_dir, 'individual_samples')
                os.makedirs(samples_dir, exist_ok=True)
                
                for explanation in individual_explanations:
                    # Create individual sample file
                    sample_file = os.path.join(samples_dir, f"sample_{explanation['sample_id']}.md")
                    with open(sample_file, 'w', encoding='utf-8') as f:
                        f.write(f"# Sample {explanation['sample_id']} - {model_name} Analysis\n\n")
                        f.write(f"**Task Type**: {task_type}\n")
                        f.write(f"**Model**: {model_name}\n")
                        f.write(f"**Generated**: {explanation['timestamp']}\n\n")
                        
                        # Add sample metadata if available
                        sample_data = explanation.get('sample_data', {})
                        metadata = sample_data.get('sample_metadata', {})
                        if metadata:
                            f.write("## Sample Information\n\n")
                            if metadata.get('prediction_correct') is not None:
                                f.write(f"- **Prediction Correct**: {metadata['prediction_correct']}\n")
                            if metadata.get('true_label') is not None:
                                f.write(f"- **True Label**: {metadata['true_label']}\n")
                            if metadata.get('predicted_label') is not None:
                                f.write(f"- **Predicted Label**: {metadata['predicted_label']}\n")
                            # XAI quality scores removed - not needed for LLM input
                            f.write("\n")
                        
                        f.write("## Analysis\n\n")
                        f.write(explanation['content'])
                    print(f"    ✅ Individual sample explanation saved to: {sample_file}")
                
                # Also create a summary file with all samples for reference
                summary_file = os.path.join(llm_dir, 'individual_explanations_summary.md')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {model_name} Individual Explanations Summary\n\n")
                    f.write(f"**Task Type**: {task_type}\n")
                    f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**Total Samples**: {len(individual_explanations)}\n\n")
                    f.write("## Sample Files\n\n")
                    f.write("Individual sample explanations are saved in separate files:\n\n")
                    for explanation in individual_explanations:
                        f.write(f"- [Sample {explanation['sample_id']}](individual_samples/sample_{explanation['sample_id']}.md)\n")
                    f.write("\n## Quick Overview\n\n")
                    for explanation in individual_explanations:
                        f.write(f"### Sample {explanation['sample_id']}\n")
                        f.write(f"**Generated**: {explanation['timestamp']}\n\n")
                        # Add first few lines of content as preview
                        content_preview = explanation['content'][:200] + "..." if len(explanation['content']) > 200 else explanation['content']
                        f.write(f"{content_preview}\n\n")
                        f.write("---\n\n")
                print(f"    ✅ Individual explanations summary saved to: {summary_file}")
            
            # Save complete JSON data for this model
            complete_data = {
                "model_name": model_name,
                "task_type": task_type,
                "global_explanation": global_explanation,
                "individual_explanations": individual_explanations,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            json_file = os.path.join(llm_dir, 'llm_explanations.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, ensure_ascii=False, indent=2)
            
            print(f"    ✅ Complete data saved to: {json_file}")
    
    except Exception as e:
        print(f"❌ Error processing {task_type} results: {e}")
        import traceback
        traceback.print_exc()


def data_preprocess(data_name):
    if data_name=='5G-NIDD':
        df = pd.read_csv(f'{cwd}/{data_name}.csv', encoding='utf-8')
        MIN_NON_NULL = 1215000
        df_cleaned = df.dropna(axis=1, thresh=MIN_NON_NULL)

        df_cleaned['Proto'] = np.where(df_cleaned['Proto'].isin(['lldp','llc','arp','ipv6-icmp']), 'other', df_cleaned['Proto'])
        df_cleaned['State'] = np.where(df_cleaned['State'].isin(['ECO','ACC','URP','RSP','TST','NRS']), 'other', df_cleaned['State'])

        proto_dummies = pd.get_dummies(df_cleaned['Proto'], prefix='Proto')
        cause_dummies = pd.get_dummies(df_cleaned['Cause'], prefix='Cause')
        state_dummies = pd.get_dummies(df_cleaned['State'], prefix='State')
        df_new = pd.concat([df_cleaned, proto_dummies, cause_dummies, state_dummies], axis=1)

        for c in ['Seq', 'Offset', 'Proto', 'Cause', 'State', 'sDSb']:
            if c in df_new.columns:
                df_new.drop(columns=[c], inplace=True)

        df_new = df_new.dropna(axis=0)

        features = [c for c in df_new.columns if c not in ['Label', 'Attack Type', 'Attack Tool', 'Unnamed: 0']]
        print(features)

    elif data_name=='5GC_PFCP':
        df_new = pd.read_csv(f'{cwd}/5GC_PFCP.csv', encoding='utf-8')
        df_new['Attack Type'] = df_new['Label'].copy()
        df_new['Label'] = np.where(df_new['Label']=='Normal', 'Benign', 'Malicious')
        features = [c for c in df_new.columns if c not in ['Label','Attack Type']]
        print(features)

    elif data_name=='5GAD':
        df = pd.read_csv(f'{cwd}/5GAD.csv', encoding='utf-8')
        df_new = df.copy()  # keep IPs for session grouping
        df_new = df_new.rename(columns={'attack_type':'Attack Type'})

        # Convert tcp_flags to numeric
        if 'tcp_flags' in df_new.columns:
            tcp_flags_mapping = {'0':0,'A':1,'PA':2,'FPA':3}
            df_new['tcp_flags'] = df_new['tcp_flags'].map(tcp_flags_mapping).fillna(0)

        features = [c for c in df_new.columns if c not in ['label','Attack Type']]

        # Binary label
        df_new['Label'] = df_new['label'].map({'normal':'Benign','attack':'Malicious'})

    else:
        print(f"Data name {data_name} not found")
        return

    X = df_new[features]
    le = LabelEncoder()
    y = df_new['Label']
    y = le.fit_transform(y)
    print("Encoded Labels:", y)
    return X, y, le, df_new


if __name__ == "__main__":

    import os
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    # === Config ===
    dataset = '5GAD'  # change if needed
    cwd = os.getcwd()
    RESULT_DIR = f"{cwd}/results/{dataset}"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # === Data preprocessing ===
    X, y, le, df_new = data_preprocess(dataset)

    # show label encoding
    for label, encoded in zip(le.classes_, range(len(le.classes_))):
        print(f"Original label '{label}' is encoded as {encoded}")

    # ============================================================
    # SESSION-AWARE SPLIT (FIXES DATA LEAKAGE)
    # ============================================================

    if dataset == '5GAD':

        session_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port']

        for c in session_cols:
            df_new[c] = df_new[c].fillna(0).astype(str)

        df_new['session_id'] = df_new[session_cols].agg('-'.join, axis=1)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        train_idx, test_val_idx = next(gss.split(df_new, groups=df_new['session_id']))

        X_train = X.iloc[train_idx].copy()
        y_train = y[train_idx]

        X_temp = X.iloc[test_val_idx].copy()
        y_temp = y[test_val_idx]

        n_temp = len(X_temp)
        val_end = int(0.5 * n_temp)

        X_val = X_temp.iloc[:val_end].copy()
        y_val = y_temp[:val_end]

        X_test = X_temp.iloc[val_end:].copy()
        y_test = y_temp[val_end:]

        # remove identifiers before training
        X_train = X_train.drop(session_cols, axis=1)
        X_val   = X_val.drop(session_cols, axis=1)
        X_test  = X_test.drop(session_cols, axis=1)

    else:
        from sklearn.model_selection import train_test_split

        # 60% train
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.4,
            stratify=y,
            random_state=42
        )

        # 20% validation / 20% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=42
        )
        
        # # simple split for other datasets
        # n = len(X)

        # train_end = int(0.6 * n)
        # val_end   = int(0.8 * n)

        # X_train = X.iloc[:train_end]
        # X_val   = X.iloc[train_end:val_end]
        # X_test  = X.iloc[val_end:]

        # y_train = y[:train_end]
        # y_val   = y[train_end:val_end]
        # y_test  = y[val_end:]

    # ============================================================
    # BINARY LIGHTGBM
    # ============================================================

    model = train_lightgbm_with_scale_pos_weight(
        X_train, y_train,
        X_val, y_val
    )

    from sklearn.metrics import f1_score

    # Get probabilities for validation set
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    test_probs = model.predict(X_test, num_iteration=model.best_iteration)

    # Test a range of thresholds
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_val, (val_probs >= t).astype(int)) for t in thresholds]

    # Find the best one
    best_threshold = thresholds[np.argmax(f1_scores)]

    # Apply to Test Set
    test_preds = (test_probs >= best_threshold).astype(int)
    print(f"Optimal Threshold: {best_threshold}")
    print("Best threshold:", best_threshold)

    pred_default = (test_probs >= 0.5).astype(int)
    pred_best = (test_probs >= best_threshold).astype(int)

    print("Pred positives (0.5):", pred_default.sum())
    print("Pred positives (best):", pred_best.sum())


    print("Validation AUC:", roc_auc_score(y_val, val_probs))
    print("Test AUC:", roc_auc_score(y_test, test_probs))

    print("\n=== Class Ratio Check ===")
    print("Train malicious ratio:", y_train.mean())
    print("Test malicious ratio :", y_test.mean())
    print("Train size:", len(y_train))
    print("Test size :", len(y_test))

    # ============================================================
    # SHAP EXPLANATIONS
    # ============================================================

    try:

        explainer, shap_values, top_30_df, cum99_df = shap_plots_Tree(
            model,
            X_train,
            feature_names=X_train.columns.tolist()
        )

        selected_features = cum99_df['Feature'].tolist()

        shap_values_for_summary = shap_values[1] if isinstance(shap_values, list) else shap_values
        global_importance = np.mean(np.abs(shap_values_for_summary), axis=0)

        fi_global = {int(idx): float(val) for idx, val in enumerate(global_importance)}

        _evaluate_and_save(
            model_name="LightGBM",
            model=model,
            X_test_df=X_test,
            y_test=y_test,
            shap_values_for_plot=None,
            scaler=None,
            uses_scaled=False,
            class_names=['Benign','Malicious'],
            explanations=None,
            method_name="shap_global",
            task_type="binary",
            explainer=explainer,
            global_importance=fi_global
        )

    except Exception as e:
        print(f"SHAP evaluation failed: {e}")

    # ============================================================
    # BINARY MULTI-MODEL STAGE
    # ============================================================

    try:
        X_binary = X_train
        y_binary = y_train

        # X_binary = pd.concat([X_train, X_test])
        # y_binary = np.concatenate([y_train, y_test])

        models_binary = train_and_explain_binary_models(X_binary, y_binary)

    except Exception as e:
        print(f"Binary multi-model stage failed: {e}")

    # ============================================================
    # MULTICLASS STAGE (ATTACK TYPE)
    # ============================================================

    try:

        train_df = df_new.iloc[train_idx]
        test_df  = df_new.iloc[test_val_idx]

        train_malicious = train_df[train_df['Label'] == 'Malicious']
        test_malicious  = test_df[test_df['Label'] == 'Malicious']

        features = train_malicious.columns.tolist()
        exclude_cols = ['Label', 'Attack Type', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'session_id']

        for col in ['Label','Attack Type']:
            if col in features:
                features.remove(col)

        X_multi_train = train_malicious[features]
        X_multi_test  = test_malicious[features]

        le_multi = LabelEncoder()

        y_multi_train = le_multi.fit_transform(train_malicious['Attack Type'])
        y_multi_test  = le_multi.transform(test_malicious['Attack Type'])

        class_names = le_multi.classes_.tolist()

        model_multi, X_train_m, X_test_m, y_train_m, y_test_m = train_lightgbm_multiclass(
            X_multi_train,
            y_multi_train,
            class_names
        )

        train_and_explain_multi_models(
            X_multi_train,
            y_multi_train,
            class_names,
            top_n=10
        )

        train_df = X_train.copy()
        train_df['Label'] = y_train

        test_df = X_test.copy()
        test_df['Label'] = y_test

        # Convert numeric labels back to strings if needed
        if train_df['Label'].dtype != object:
            train_df['Label'] = train_df['Label'].map({0: 'Benign', 1: 'Malicious'})
            test_df['Label']  = test_df['Label'].map({0: 'Benign', 1: 'Malicious'})

        # Keep only malicious samples
        train_malicious = train_df[train_df['Label'] == 'Malicious']
        test_malicious  = test_df[test_df['Label'] == 'Malicious']

        # Ensure Attack Type exists
        if 'Attack Type' not in df_new.columns:
            raise ValueError("Attack Type column not available for multiclass classification")

        # Get attack types for those rows
        y_multi_train_raw = df_new.loc[train_malicious.index, 'Attack Type']
        y_multi_test_raw  = df_new.loc[test_malicious.index, 'Attack Type']

        # Feature selection
        features = train_malicious.columns.tolist()
        exclude_cols = ['Label', 'Attack Type', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'session_id']

        features = [f for f in features if f not in exclude_cols]

        X_multi_train = train_malicious[features]
        X_multi_test  = test_malicious[features]

        # Encode attack types
        le_multi = LabelEncoder()

        y_multi_train = le_multi.fit_transform(y_multi_train_raw)
        y_multi_test  = le_multi.transform(y_multi_test_raw)

        class_names = le_multi.classes_.tolist()

        model_multi, X_train_m, X_test_m, y_train_m, y_test_m = train_lightgbm_multiclass(
            X_multi_train,
            y_multi_train,
            class_names
        )

        train_and_explain_multi_models(
            X_multi_train,
            y_multi_train,
            class_names,
            top_n=10
        )

    except Exception as e:
        print(f"Multiclass stage failed: {e}")

    # ============================================================
    # LLM EXPLANATIONS
    # ============================================================

    GENERATE_LLM_EXPLANATIONS = os.environ.get(
        'GENERATE_LLM_EXPLANATIONS', '1'
    ) == '1'

    if GENERATE_LLM_EXPLANATIONS:

        print("\n" + "="*60)
        print("LLM Explanation Generation")
        print("="*60)

        generate_llm_explanations()

    else:

        print("\nLLM explanation generation disabled.")