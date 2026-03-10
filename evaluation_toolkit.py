"""
Simplified XAI Quality Assessment Toolkit
Simplified evaluation framework based on three core dimensions:
1. Faithfulness: deletion/insertion AUC and feature ablation correlation
2. Robustness: stability of top-k features under small perturbations (Jaccard/variance)
3. Complexity: sparsity/effective complexity (Effective Complexity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import pearsonr, spearmanr
import time
import warnings
warnings.filterwarnings('ignore')

class SimplifiedXAIEvaluator:
    """Simplified XAI quality evaluator"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize evaluator
        
        Args:
            config: Evaluation configuration parameters
        """
        self.config = config or self._default_config()
        self.evaluation_results = {}
        self.dimension_weights = {
            'faithfulness': 0.4,    # Increased weight as it's the most important dimension
            'robustness': 0.35,     # Stability is also important
            'complexity': 0.25      # Simplicity has relatively lower weight
        }
    
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'sample_size': 10,
            'top_k_features': 10,           # Number of top-k features for robustness evaluation
            'perturbation_levels': [0.05, 0.1, 0.2],  # Perturbation levels
            'bootstrap_samples': 10,        # Bootstrap sampling count
            'deletion_steps': 10,           # Steps for deletion/insertion tests
            'complexity_threshold': 0.01,  # Effective complexity threshold
        }
    
    # ============ 1. Faithfulness Evaluation ============
    def evaluate_faithfulness_single_sample(self, model, explanation: Dict, x_sample: np.ndarray, 
                                           global_means: Optional[np.ndarray]) -> Dict:
        """
        Evaluate faithfulness for a single sample
        
        Args:
            model: Model to be explained
            explanation: Single explanation result
            x_sample: Single test sample
            global_means: Global means for masking baseline
            
        Returns:
            Single sample faithfulness evaluation results
        """
        # 1. Deletion/Insertion AUC
        del_auc, ins_auc = self._deletion_insertion_auc(
            model, x_sample, explanation, global_means
        )
        
        # 2. Feature Ablation Correlation
        ablation_corr = self._feature_ablation_correlation(
            model, x_sample, explanation, global_means
        )
        
        # Calculate single sample faithfulness score
        faithfulness_score = (
            del_auc * 0.4 +
            ins_auc * 0.4 +
            ablation_corr * 0.2
        )
        
        return {
            'overall_score': faithfulness_score,
            'deletion_auc': del_auc,
            'insertion_auc': ins_auc,
            'ablation_correlation': ablation_corr
        }
    
    def evaluate_faithfulness(self, model, explanations: List, X_test: np.ndarray, 
                            y_test: np.ndarray) -> Dict:
        """
        Evaluate explanation faithfulness
        
        Includes two key metrics:
        1. Deletion/Insertion AUC: AUC calculation based on target class probability
        2. Feature Ablation Correlation: Correlation between feature ablation decline and importance
        
        Args:
            model: Model to be explained
            explanations: List of explanation results
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Faithfulness evaluation results
        """
        print("🔍 Evaluating Faithfulness...")
        
        deletion_auc_list = []
        insertion_auc_list = []
        ablation_correlations = []
        
        # Convert to numpy array for indexing
        if hasattr(X_test, 'iloc'):
            X_array = X_test.values
        else:
            X_array = np.asarray(X_test)
        
        # Compute column means as masking baseline
        global_means = np.mean(X_array, axis=0) if isinstance(X_array, np.ndarray) else None
        
        for i, explanation in enumerate(explanations[:self.config['sample_size']]):
            # 1. Deletion/Insertion AUC
            del_auc, ins_auc = self._deletion_insertion_auc(
                model, X_array[i], explanation, global_means
            )
            deletion_auc_list.append(del_auc)
            insertion_auc_list.append(ins_auc)
            
            # 2. Feature Ablation Correlation
            ablation_corr = self._feature_ablation_correlation(
                model, X_array[i], explanation, global_means
            )
            ablation_correlations.append(ablation_corr)
        
        # Compute overall faithfulness score
        faithfulness_score = (
            np.mean(deletion_auc_list) * 0.4 +
            np.mean(insertion_auc_list) * 0.4 +
            np.mean(ablation_correlations) * 0.2
        )
        
        results = {
            'overall_score': faithfulness_score,
            'deletion_auc': np.mean(deletion_auc_list),
            'insertion_auc': np.mean(insertion_auc_list),
            'ablation_correlation': np.mean(ablation_correlations),
            'evaluation_count': len(deletion_auc_list)
        }
        
        self.evaluation_results['faithfulness'] = results
        return results
    
    def _deletion_insertion_auc(self, model, x: np.ndarray, explanation: Dict, 
                               global_means: Optional[np.ndarray]) -> Tuple[float, float]:
        """
        Compute deletion/insertion AUC
        
        Principle:
        1. Deletion: start from original sample, remove features by importance, observe target prob decrease
        2. Insertion: start from baseline, add features by importance, observe target prob increase
        3. AUC: trapezoidal area under curve, normalized to [0,1]
        
        Args:
            model: model
            x: single sample
            explanation: explanation result
            global_means: global mean baseline
            
        Returns:
            (deletion_auc, insertion_auc)
        """
        try:
            probs = model.predict_proba([x])[0]
        except Exception:
            return 0.0, 0.0

        target_class = int(np.argmax(probs))
        fi = explanation.get('feature_importance', {})
        if not fi:
            return 0.0, 0.0

        num_features = len(x)
        
        # Explanations now include all features; sort by importance
        sorted_idx = [int(k) for k, _ in sorted(fi.items(), 
                                               key=lambda kv: abs(kv[1]), reverse=True)]
        
        if len(sorted_idx) == 0:
            return 0.0, 0.0

        # Step control
        steps = min(self.config['deletion_steps'], len(sorted_idx))
        k_list = np.linspace(1, len(sorted_idx), steps, dtype=int)

        baseline = global_means if global_means is not None else np.zeros_like(x)

        # Deletion test
        del_scores = []
        x_del = x.copy()
        for k in k_list:
            idx_k = sorted_idx[:k]
            x_del[idx_k] = baseline[idx_k]
            p = model.predict_proba([x_del])[0][target_class]
            del_scores.append(p)

        # Insertion test
        ins_scores = []
        x_ins = baseline.copy()
        for k in k_list:
            idx_k = sorted_idx[:k]
            x_ins[idx_k] = x[idx_k]
            p = model.predict_proba([x_ins])[0][target_class]
            ins_scores.append(p)

        # Compute AUC
        def _auc_normalized(scores: List[float]) -> float:
            if len(scores) < 2:
                return 0.0
            s = np.array(scores)
            if np.allclose(s.max(), s.min()):  # [MODIFIED] avoid meaningless normalization when all same
                return 0.5  # return neutral score instead of 0
            s = (s - s.min()) / (s.max() - s.min())
            x_axis = np.linspace(0, 1, len(s))
            return float(np.trapz(s, x_axis))

        return _auc_normalized(1 - np.array(del_scores)), _auc_normalized(np.array(ins_scores))

    
    def _feature_ablation_correlation(self, model, x: np.ndarray, explanation: Dict,
                                    global_means: Optional[np.ndarray]) -> float:
        """
        Compute correlation between ablation drop and importance
        
        Principle:
        1. For each feature, compute the drop in target probability after removing it
        2. Correlate the drop with feature importance
        3. Higher correlation indicates better faithfulness
        
        Args:
            model: model
            x: single sample
            explanation: explanation
            global_means: global mean baseline
            
        Returns:
            correlation coefficient
        """
        try:
            original_prob = model.predict_proba([x])[0]
            target_class = int(np.argmax(original_prob))
            original_score = original_prob[target_class]
        except Exception:
            return 0.0

        fi = explanation.get('feature_importance', {})
        if not fi:
            return 0.0

        baseline = global_means if global_means is not None else np.zeros_like(x)
        
        # Explanations include all features; use directly
        importance_values = []
        ablation_drops = []
        
        for feature_idx, importance in fi.items():
            # Compute probability drop after ablating feature
            x_ablated = x.copy()
            x_ablated[int(feature_idx)] = baseline[int(feature_idx)]
            
            try:
                ablated_prob = model.predict_proba([x_ablated])[0][target_class]
                drop = original_score - ablated_prob
                importance_values.append(abs(importance))
                ablation_drops.append(drop)
            except Exception:
                continue
        
        if len(importance_values) < 2:
            return 0.0
        
        # Compute correlation
        correlation, _ = pearsonr(importance_values, ablation_drops)
        if np.isnan(correlation):
            return 0.0
        return correlation
    
    # ============ 2. Robustness Evaluation ============
    def evaluate_robustness_single_sample(self, model, explanation: Dict, x_sample: np.ndarray, 
                                         explainer=None, explainer_type='shap') -> Dict:
        """
        Evaluate robustness for a single sample
        
        Args:
            model: Model to be explained
            explanation: Single explanation result
            x_sample: Single test sample
            explainer: XAI explainer (SHAP or LIME)
            explainer_type: explainer type ('shap' or 'lime')
            
        Returns:
            Single sample robustness evaluation results
        """
        # 1. Top-k feature stability
        jaccard_sim = self._top_k_stability(model, x_sample, explanation, explainer, explainer_type)
        
        # 2. Feature importance variance
        importance_var = self._importance_variance(model, x_sample, explanation, explainer, explainer_type)
        
        # Calculate single sample robustness score
        robustness_score = (
            jaccard_sim * 0.6 +
            (1 - importance_var) * 0.4  # lower variance is better
        )
        
        return {
            'overall_score': robustness_score,
            'jaccard_similarity': jaccard_sim,
            'importance_variance': importance_var
        }
    
    def evaluate_robustness(self, model, explanations: List, X_test: np.ndarray, 
                           explainer=None, explainer_type='shap') -> Dict:
        """
        Evaluate explanation robustness
        
        Includes two key metrics:
        1. Top-k Feature Stability: Jaccard similarity under small perturbations
        2. Feature Importance Variance: variance via bootstrap sampling
        
        Args:
            model: model
            explanations: list of explanations
            X_test: test data
            explainer: XAI explainer (SHAP or LIME)
            explainer_type: explainer type ('shap' or 'lime')
            
        Returns:
            robustness evaluation results
        """
        print("⚖️ Evaluating Robustness...")
        
        jaccard_similarities = []
        importance_variances = []
        
        for i, explanation in enumerate(explanations[:self.config['sample_size']]):
            # 1. Top-k feature stability
            jaccard_sim = self._top_k_stability(
                model, X_test[i], explanation, explainer, explainer_type
            )
            jaccard_similarities.append(jaccard_sim)
            
            # 2. Feature importance variance
            importance_var = self._importance_variance(
                model, X_test[i], explanation, explainer, explainer_type
            )
            importance_variances.append(importance_var)
        
        # Compute overall robustness score
        robustness_score = (
            np.mean(jaccard_similarities) * 0.6 +
            (1 - np.mean(importance_variances)) * 0.4  # lower variance is better
        )
        
        results = {
            'overall_score': robustness_score,
            'jaccard_similarity': np.mean(jaccard_similarities),
            'importance_variance': np.mean(importance_variances),
            'evaluation_count': len(jaccard_similarities)
        }
        
        self.evaluation_results['robustness'] = results
        return results


    def _top_k_stability(self, model, x: np.ndarray, explanation: Dict, 
                         explainer=None, explainer_type='shap') -> float:
        """
        Compute top-k feature stability under small perturbations (revised)
        """
        original_importance = explanation.get('feature_importance', {})
        if not original_importance or explainer is None:
            return 0.0  # [MODIFIED] no original importance => no score boost
        
        if explainer is None:
            return 0.0  # [MODIFIED] explainer=None counted as 0 instead of 1.0
        
        top_k = self.config['top_k_features']
        # Ensure keys are integers
        original_top_k = set(sorted([int(k) for k in original_importance.keys()], 
                                  key=lambda k: abs(original_importance[k]), 
                                  reverse=True)[:top_k])
        
        if not original_top_k:
            return 0.0

        jaccard_scores = []
        
        # Use unified bootstrap samples for stability
        num_samples = self.config.get('bootstrap_samples', 10)

        for _ in range(num_samples):
            noise_level = np.random.choice(self.config['perturbation_levels'])
            noise = np.random.normal(0, noise_level * np.std(x), x.shape) # use relative noise
            x_perturbed = x + noise
            
            # let explainer return all features, then select top-k here
            perturbed_importance = self._get_perturbed_explanation(
                model, x_perturbed, explainer, explainer_type, len(x)
            )
            
            if perturbed_importance:
                perturbed_top_k = set(sorted([int(k) for k in perturbed_importance.keys()],
                                           key=lambda k: abs(perturbed_importance[k]),
                                           reverse=True)[:top_k])
                
                intersection_size = len(original_top_k.intersection(perturbed_top_k))
                union_size = len(original_top_k.union(perturbed_top_k))
                
                score = intersection_size / union_size if union_size > 0 else 1.0
                jaccard_scores.append(score)
        
        return np.mean(jaccard_scores) if jaccard_scores else 0.0

    def _importance_variance(self, model, x: np.ndarray, explanation: Dict, 
                           explainer=None, explainer_type='shap') -> float:
        """
        calculate variance of feature importance (logic maintained, but dependent functions fixed)
        """
        original_importance = explanation.get('feature_importance', {})
        if not original_importance or explainer is None:
            return 0.0 # variance of 0 means perfect stability
        
        bootstrap_importances = []
        num_samples = self.config.get('bootstrap_samples', 10)
        
        for _ in range(num_samples):
            noise_level = np.random.choice(self.config['perturbation_levels'])
            noise = np.random.normal(0, noise_level * np.std(x), x.shape) # use relative noise
            x_bootstrap = x + noise
            
            boot_importance = self._get_perturbed_explanation(
                model, x_bootstrap, explainer, explainer_type, len(x)
            )
            
            if boot_importance:
                bootstrap_importances.append(boot_importance)
        
        if len(bootstrap_importances) < 2:
            return 0.0
        
        # unify keys of original importance to integers
        feature_keys = [int(k) for k in original_importance.keys()]
        feature_variances = []
        
        for feature_idx in feature_keys:
            values = [imp[feature_idx] for imp in bootstrap_importances if feature_idx in imp]
            if len(values) > 1:
                variance = np.var(values)
                feature_variances.append(variance)
        
        if not feature_variances:
            return 0.0
        
        # normalize using maximum absolute value of original importance, more stable
        max_abs_importance = max(abs(v) for v in original_importance.values())
        if max_abs_importance == 0:
            return 0.0 # avoid division by zero
        
        normalized_variance = np.mean(feature_variances) / (np.var(list(original_importance.values())) + 1e-9)  
        return min(1.0, normalized_variance)
    
    def _get_perturbed_explanation(self, model, x_perturbed: np.ndarray, 
                                 explainer, explainer_type: str, num_features: int) -> Dict:
        """
        get XAI explanation for perturbed sample (parameter names modified for clarity)
        """
        try:
            if explainer_type.lower() == 'shap':
                return self._get_shap_explanation(model, x_perturbed, explainer, num_features)
            elif explainer_type.lower() == 'lime':
                return self._get_lime_explanation(model, x_perturbed, explainer, num_features)
            else:
                print(f"Warning: Unknown explainer type: {explainer_type}")
                return {}
        except Exception as e:
            # catch broader exceptions
            print(f"Warning: Failed to get perturbed explanation for type {explainer_type}: {e}")
            return {}

    def _get_shap_explanation(self, model, x_perturbed: np.ndarray, 
                            shap_explainer, num_features: int) -> Dict:
        """
        get SHAP explanation (fixed version)
        """
        try:
            if x_perturbed.ndim == 1:
                x_perturbed = x_perturbed.reshape(1, -1)
            
            # ensure data format is correct, especially for DeepExplainer
            if hasattr(shap_explainer, '__class__') and 'DeepExplainer' in str(shap_explainer.__class__):
                x_perturbed = x_perturbed.astype('float32')
            
            shap_values = shap_explainer.shap_values(x_perturbed)
            
            # fix: safer way to get predicted class
            try:
                pred_probs = model.predict_proba(x_perturbed)
                # ensure pred_probs is 2D array
                if pred_probs.ndim == 1:
                    pred_probs = pred_probs.reshape(1, -1)
                # safely get predicted class
                predicted_class = int(np.argmax(pred_probs[0]))
            except Exception:
                # if prediction fails, use default class
                predicted_class = 0
            
            if isinstance(shap_values, list):
                # ensure predicted class is within range
                if predicted_class >= len(shap_values) or predicted_class < 0:
                    predicted_class = len(shap_values) - 1  # use last class
                class_shap_values = shap_values[predicted_class]
            else:
                class_shap_values = shap_values
            
            # ensure class_shap_values is 1D array
            if class_shap_values.ndim > 1:
                class_shap_values = class_shap_values[0]
            
            # safely convert SHAP values to dictionary
            feature_importance = {}
            for idx in range(len(class_shap_values)):
                try:
                    # ensure value is scalar
                    value = class_shap_values[idx]
                    if hasattr(value, 'item'):
                        value = value.item()  # convert numpy scalar to Python scalar
                    feature_importance[int(idx)] = float(value)
                except (ValueError, TypeError):
                    feature_importance[int(idx)] = 0.0
            
            return feature_importance
            
        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            return {}

    def _get_lime_explanation(self, model, x_perturbed: np.ndarray, 
                            lime_explainer, num_features: int) -> Dict:
        """
        get LIME explanation (fixed version)
        """
        try:
            # force convert input to pure numeric to avoid object dtype
            x_perturbed = np.asarray(x_perturbed)
            if x_perturbed.ndim == 2:
                x_perturbed = x_perturbed[0]
            x_perturbed = x_perturbed.astype('float64', copy=False)
            
            def predict_fn(x):
                x = np.asarray(x)
                # LIME will pass 2D array, ensure 2D here
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                # ensure input to model is numeric array
                if x.dtype == object:
                    x = x.astype('float64', copy=False)

                # predict probability
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(x)
                else:
                    # Keras/TensorFlow models usually return class probabilities through predict
                    proba = model.predict(x, verbose=0)

                # unify to 2D
                if proba.ndim == 1:
                    proba = proba.reshape(1, -1)

                # align with LIME class count (avoid dimension errors from column count mismatch)
                class_names = getattr(lime_explainer, 'class_names', None)
                if class_names is not None:
                    num_classes = len(class_names)
                    if proba.shape[1] < num_classes:
                        padding = np.zeros((proba.shape[0], num_classes - proba.shape[1]))
                        proba = np.column_stack([proba, padding])
                    elif proba.shape[1] > num_classes:
                        proba = proba[:, :num_classes]

                # special case: only 1 column, treat as binary classification sigmoid output
                if proba.shape[1] == 1:
                    proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])

                return proba
            
            # fix: no longer rely on fragile string parsing, use LIME's structured output
            exp = lime_explainer.explain_instance(x_perturbed, predict_fn, num_features=num_features)
            # use prediction function consistent with LIME to get predicted class, avoid class inconsistency
            try:
                proba_for_label = predict_fn(x_perturbed.reshape(1, -1))
                predicted_class = int(np.argmax(proba_for_label[0]))
            except Exception:
                predicted_class = 0
            
            if predicted_class in exp.local_exp:
                # local_exp value is list of (feature_index, weight)
                feature_importance = {}
                for idx, w in exp.local_exp[predicted_class]:
                    try:
                        # safely handle weight value, may be array or scalar
                        if hasattr(w, 'item'):
                            weight_value = float(w.item())
                        elif hasattr(w, '__len__') and len(w) > 0:
                            weight_value = float(w[0])
                        else:
                            weight_value = float(w)
                        feature_importance[int(idx)] = weight_value
                    except Exception:
                        feature_importance[int(idx)] = 0.0
                return feature_importance
            else:
                # if predicted class is not in explanation, try using first available class
                if exp.local_exp:
                    first_class = list(exp.local_exp.keys())[0]
                    feature_importance = {}
                    for idx, w in exp.local_exp[first_class]:
                        try:
                            # safely handle weight value, may be array or scalar
                            if hasattr(w, 'item'):
                                weight_value = float(w.item())
                            elif hasattr(w, '__len__') and len(w) > 0:
                                weight_value = float(w[0])
                            else:
                                weight_value = float(w)
                            feature_importance[int(idx)] = weight_value
                        except Exception:
                            feature_importance[int(idx)] = 0.0
                    return feature_importance
                return {}
                
        except Exception as e:
            print(f"Warning: LIME explanation failed: {e}")
            return {}   
    
    def _get_predicted_class_multiclass(self, model, x_sample: np.ndarray) -> int:
        """
        specifically handle multiclass model predicted class acquisition (based on probability argmax, more robust)
        """
        try:
            x = np.asarray(x_sample).reshape(1, -1)
            # prefer using predict_proba
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(x)
            else:
                proba = model.predict(x, verbose=0)

            if proba.ndim == 1:
                proba = proba.reshape(1, -1)
            if proba.shape[1] == 1:
                proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
            return int(np.argmax(proba, axis=1)[0])
        except Exception as e:
            print(f"Failed to get multiclass predicted class: {e}")
            return 0
    
    # ============ 3. Complexity Evaluation ============
    def evaluate_complexity_single_sample(self, explanation: Dict, total_features: int = None) -> Dict:
        """
        Evaluate complexity for a single sample - use all features for complexity evaluation
        
        Args:
            explanation: Single explanation result
            total_features: total number of features, if None use number of features in explanation
            
        Returns:
            Single sample complexity evaluation results
        """
        # 1. sparsity evaluation - use all features
        sparsity = self._calculate_sparsity(explanation, total_features)
        
        # 2. effective complexity evaluation (Effective Complexity)
        effective_complexity = self._calculate_effective_complexity(explanation)
        
        # Calculate single sample complexity score (higher is better - simplicity)
        # now both indicators are better when higher: higher sparsity means more concise, higher effective_complexity means more concise
        complexity_score = (
            sparsity * 0.6 +  # higher sparsity means more concise
            effective_complexity * 0.4  # higher effective complexity means more concise
        )
        
        return {
            'overall_score': complexity_score,  # use directly, higher is better
            'sparsity': sparsity,
            'effective_complexity': effective_complexity
        }
    
    def evaluate_single_sample_complete(self, model, explanation: Dict, x_sample: np.ndarray, 
                                      global_means: Optional[np.ndarray] = None,
                                      explainer=None, explainer_type='shap', 
                                      total_features: int = None) -> Dict:
        """
        Complete evaluation for a single sample across all three dimensions
        
        Args:
            model: Model to be explained
            explanation: Single explanation result
            x_sample: Single test sample
            global_means: Global means for masking baseline
            explainer: XAI explainer (SHAP or LIME)
            explainer_type: explainer type ('shap' or 'lime')
            
        Returns:
            Complete single sample evaluation results
        """
        # Calculate global means if not provided
        if global_means is None:
            global_means = np.zeros_like(x_sample)
        
        # Evaluate all three dimensions
        faithfulness_results = self.evaluate_faithfulness_single_sample(
            model, explanation, x_sample, global_means
        )
        
        robustness_results = self.evaluate_robustness_single_sample(
            model, explanation, x_sample, explainer, explainer_type
        )
        
        complexity_results = self.evaluate_complexity_single_sample(explanation, total_features)
        
        # Calculate overall score
        overall_score = (
            faithfulness_results['overall_score'] * self.dimension_weights['faithfulness'] +
            robustness_results['overall_score'] * self.dimension_weights['robustness'] +
            complexity_results['overall_score'] * self.dimension_weights['complexity']
        )
        
        return {
            'overall_score': overall_score,
            'faithfulness': faithfulness_results,
            'robustness': robustness_results,
            'complexity': complexity_results
        }
    
    def evaluate_complexity(self, explanations: List, total_features: int = None) -> Dict:
        """
        evaluate explanation complexity
        
        includes two key indicators:
        1. Sparsity: proportion of non-zero important features (higher means more concise)
        2. Effective Complexity: simplicity evaluation based on coefficient of variation (higher coefficient means more concise)
        
        Args:
            explanations: list of explanation results
            
        Returns:
            simplicity evaluation results (higher values mean more concise)
        """
        print(" 🔍 Evaluating complexity...")
        
        sparsity_scores = []
        effective_complexities = []
        
        for explanation in explanations[:self.config['sample_size']]:
            # 1. sparsity evaluation - use all features
            sparsity = self._calculate_sparsity(explanation, total_features)
            sparsity_scores.append(sparsity)
            
            # 2. effective complexity evaluation
            effective_complexity = self._calculate_effective_complexity(explanation)
            effective_complexities.append(effective_complexity)
        
        # calculate comprehensive complexity score (higher is better - simplicity)
        # now both indicators are better when higher: higher sparsity means more concise, higher effective_complexity means more concise
        complexity_score = (
            np.mean(sparsity_scores) * 0.6 +  # higher sparsity means more concise
            np.mean(effective_complexities) * 0.4   # higher effective complexity means more concise
        )
        
        results = {
            'overall_score': complexity_score,  # use directly, higher is better
            'sparsity': np.mean(sparsity_scores),
            'effective_complexity': np.mean(effective_complexities),
            'evaluation_count': len(sparsity_scores)
        }
        
        self.evaluation_results['complexity'] = results
        return results
    
    def _calculate_sparsity(self, explanation: Dict, total_features: int = None) -> float:
        """
        calculate explanation sparsity - use all features for complexity evaluation
        
        principle:
        1. count number of non-zero important features
        2. calculate proportion relative to total number of features
        3. high sparsity (few features have importance) indicates concise explanation
        
        Args:
            explanation: explanation result
            total_features: total number of features, if None use number of features in explanation
            
        Returns:
            sparsity score [0,1], higher means more sparse
        """
        feature_importance = explanation.get('feature_importance', {})
        if not feature_importance:
            return 0.0
        
        # now explanations contain all features, use directly
        if total_features is None:
            total_features = len(feature_importance)
        
        # calculate number of non-zero important features
        non_zero_features = sum(1 for v in feature_importance.values() if abs(v) > 0.02)
        
        if total_features == 0:
            return 0.0
        
        # sparsity = 1 - (non-zero features / total features)
        sparsity = 1 - (non_zero_features / total_features)
        return sparsity
    
    def _calculate_effective_complexity(self, explanation: Dict) -> float:
        """
        calculate effective complexity (Effective Complexity) - fixed version
        
        principle:
        1. since explanation data only contains top-k features, we use relative importance to evaluate complexity
        2. calculate concentration degree of importance value distribution, more concentrated means more concise
        3. use coefficient of variation of importance values to measure distribution concentration
        4. high coefficient of variation = importance concentrated on few features = more concise
        5. low coefficient of variation = importance distributed evenly = more complex
        6. higher values mean more concise (consistent with other dimension logic)
        
        Args:
            explanation: explanation result
            
        Returns:
            normalized simplicity score [0,1], higher means more concise
        """
        feature_importance = explanation.get('feature_importance', {})
        if not feature_importance:
            return 0.0
        
        # get all importance values
        importance_values = list(feature_importance.values())
        if not importance_values:
            return 0.0
        
        # method 1: based on relative distribution of importance values
        abs_values = [abs(v) for v in importance_values]
        if len(abs_values) < 2:
            return 1.0  # when only one feature, consider it most concise
        
        # calculate coefficient of variation of importance values (std/mean)
        mean_importance = np.mean(abs_values)
        std_importance = np.std(abs_values)
        
        if mean_importance == 0:
            return 1.0  # when all values are 0, consider it most concise
        
        # coefficient of variation = std / mean
        coefficient_of_variation = std_importance / mean_importance
        
        # normalize to [0,1], higher coefficient of variation means higher simplicity
        # use sigmoid function for normalization to make results smoother
        # adjust parameters so high coefficient of variation gets high score (high simplicity)
        simplicity_score = 1 / (1 + np.exp(-2 * (coefficient_of_variation - 1)))
        
        return min(1.0, max(0.0, simplicity_score))
    
    # ============ Comprehensive Evaluation ============
    def calculate_overall_score(self, custom_weights: Optional[Dict] = None) -> Dict:
        """calculate comprehensive quality score"""
        weights = custom_weights or self.dimension_weights
        
        overall_score = 0.0
        dimension_scores = {}
        
        for dimension, weight in weights.items():
            if dimension in self.evaluation_results:
                score = self.evaluation_results[dimension]['overall_score']
                dimension_scores[dimension] = score
                overall_score += score * weight
        
        return {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'weights_used': weights
        }
    
    def generate_evaluation_report(self) -> str:
        """Generate evaluation report"""
        report = "# Simplified XAI Quality Assessment Report\n\n"
        
        overall_results = self.calculate_overall_score()
        report += f"## Overall Score: {overall_results['overall_score']:.3f}\n\n"
        
        report += "## Detailed Dimension Scores\n\n"
        for dimension, results in self.evaluation_results.items():
            score = results['overall_score']
            report += f"### {dimension.title()}: {score:.3f}\n"
            
            # Add detailed metrics
            for key, value in results.items():
                if key != 'overall_score':
                    if isinstance(value, float):
                        report += f"- {key}: {value:.3f}\n"
                    else:
                        report += f"- {key}: {value}\n"
            report += "\n"
        
        return report
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Visualize evaluation results"""
        if not self.evaluation_results:
            print("No evaluation results available for visualization")
            return

        # Radar chart
        dimensions = list(self.evaluation_results.keys())
        scores = [self.evaluation_results[dim]['overall_score'] for dim in dimensions]

        # Set font for better readability
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False

        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, scores, 'o-', linewidth=3, label='XAI Assessment', color='#2E8B57', markersize=8)
        ax.fill(angles, scores, alpha=0.25, color='#2E8B57')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([dim.title() for dim in dimensions], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Simplified XAI Quality Assessment Radar Chart', size=18, pad=30, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add score labels with better positioning to avoid overlap
        for angle, score, dim in zip(angles[:-1], scores[:-1], dimensions):
            # Position labels outside the plot area to avoid overlap
            label_radius = score + 0.12
            ax.text(angle, label_radius, f'{score:.3f}',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Add grid lines for better readability
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Radar chart saved to: {save_path}")

        plt.show()

    # ============ Global Explanation Evaluation Methods ============
    def evaluate_global_faithfulness(self, model, X_samples: np.ndarray, 
                                    global_importance: Dict, y_samples=None,
                                    sample_size: int = 200) -> Dict:
        """
        Evaluate faithfulness of global explanation
        
        For global explanations, faithfulness measures how well the global feature
        importance ranking can explain the model's behavior across multiple samples.
        
        Args:
            model: Model to be explained
            X_samples: Test samples array
            global_importance: Global feature importance dict {feature_idx: importance}
            y_samples: True labels (optional)
            sample_size: Number of samples to evaluate
            
        Returns:
            Global faithfulness evaluation results
        """
        print("[INFO] Evaluating Global Faithfulness...")
        
        # Convert to numpy array
        if hasattr(X_samples, 'iloc'):
            X_array = X_samples.values
        else:
            X_array = np.asarray(X_samples)
        
        # Limit sample size
        n_samples = min(sample_size, len(X_array))
        X_eval = X_array[:n_samples]
        
        # Compute global means as baseline
        global_means = np.mean(X_array, axis=0)
        
        # Sort features by global importance
        sorted_features = sorted(global_importance.items(), 
                                key=lambda x: abs(x[1]), reverse=True)
        sorted_indices = [int(idx) for idx, _ in sorted_features]
        
        deletion_auc_list = []
        insertion_auc_list = []
        ablation_correlations = []
        
        # Evaluate on each sample using global importance ranking
        for i, x in enumerate(X_eval):
            # 1. Deletion/Insertion AUC with global ranking
            del_auc, ins_auc = self._global_deletion_insertion_auc(
                model, x, sorted_indices, global_means
            )
            deletion_auc_list.append(del_auc)
            insertion_auc_list.append(ins_auc)
            
            # 2. Correlation between global importance and ablation impact
            ablation_corr = self._global_ablation_correlation(
                model, x, global_importance, global_means
            )
            ablation_correlations.append(ablation_corr)
        
        # Compute overall faithfulness score
        faithfulness_score = (
            np.mean(deletion_auc_list) * 0.4 +
            np.mean(insertion_auc_list) * 0.4 +
            np.mean(ablation_correlations) * 0.2
        )
        
        results = {
            'overall_score': faithfulness_score,
            'average_deletion_auc': np.mean(deletion_auc_list),
            'average_insertion_auc': np.mean(insertion_auc_list),
            'average_ablation_correlation': np.mean(ablation_correlations),
            'evaluation_samples': n_samples
        }
        
        self.evaluation_results['faithfulness'] = results
        return results
    
    def _global_deletion_insertion_auc(self, model, x: np.ndarray, 
                                      sorted_indices: List[int],
                                      baseline: np.ndarray) -> Tuple[float, float]:
        """
        Compute deletion/insertion AUC using global feature ranking
        
        Args:
            model: Model
            x: Single sample
            sorted_indices: Feature indices sorted by global importance
            baseline: Baseline values for masking
            
        Returns:
            (deletion_auc, insertion_auc)
        """
        try:
            probs = model.predict_proba([x])[0]
            target_class = int(np.argmax(probs))
        except Exception:
            return 0.0, 0.0
        
        if len(sorted_indices) == 0:
            return 0.0, 0.0
        
        # Step control
        steps = min(self.config['deletion_steps'], len(sorted_indices))
        k_list = np.linspace(1, len(sorted_indices), steps, dtype=int)
        
        # Deletion test: remove features by global importance order
        del_scores = []
        x_del = x.copy()
        for k in k_list:
            idx_k = sorted_indices[:k]
            x_del[idx_k] = baseline[idx_k]
            p = model.predict_proba([x_del])[0][target_class]
            del_scores.append(p)
        
        # Insertion test: add features by global importance order
        ins_scores = []
        x_ins = baseline.copy()
        for k in k_list:
            idx_k = sorted_indices[:k]
            x_ins[idx_k] = x[idx_k]
            p = model.predict_proba([x_ins])[0][target_class]
            ins_scores.append(p)
        
        # Compute normalized AUC
        def _auc_normalized(scores: List[float]) -> float:
            if len(scores) < 2:
                return 0.0
            s = np.array(scores)
            if np.allclose(s.max(), s.min()):
                return 0.5
            s = (s - s.min()) / (s.max() - s.min())
            x_axis = np.linspace(0, 1, len(s))
            return float(np.trapz(s, x_axis))
        
        return _auc_normalized(1 - np.array(del_scores)), _auc_normalized(np.array(ins_scores))
    
    def _global_ablation_correlation(self, model, x: np.ndarray, 
                                    global_importance: Dict,
                                    baseline: np.ndarray) -> float:
        """
        Compute correlation between global importance and actual ablation impact
        
        Args:
            model: Model
            x: Single sample
            global_importance: Global feature importance
            baseline: Baseline values
            
        Returns:
            Correlation coefficient
        """
        try:
            original_prob = model.predict_proba([x])[0]
            target_class = int(np.argmax(original_prob))
            original_score = original_prob[target_class]
        except Exception:
            return 0.0
        
        importance_values = []
        ablation_drops = []
        
        for feature_idx, importance in global_importance.items():
            x_ablated = x.copy()
            x_ablated[int(feature_idx)] = baseline[int(feature_idx)]
            
            try:
                ablated_prob = model.predict_proba([x_ablated])[0][target_class]
                drop = original_score - ablated_prob
                importance_values.append(abs(importance))
                ablation_drops.append(drop)
            except Exception:
                continue
        
        if len(importance_values) < 2:
            return 0.0
        
        correlation, _ = pearsonr(importance_values, ablation_drops)
        if np.isnan(correlation):
            return 0.0
        return correlation
    
    def evaluate_global_robustness(self, model, X_samples: np.ndarray, 
                                  explainer, original_global_importance: Dict,
                                  n_subsets: int = 30, subset_ratio: float = 0.5,
                                  explainer_type: str = 'shap') -> Dict:
        """
        Evaluate robustness of global explanation through subset sampling
        
        For global explanations, robustness measures the stability of global
        feature importance when computed on different sample subsets.
        
        Args:
            model: Model to be explained
            X_samples: Test samples array
            explainer: XAI explainer (SHAP or LIME)
            original_global_importance: Original global feature importance
            n_subsets: Number of subsets to sample
            subset_ratio: Ratio of samples in each subset
            explainer_type: Type of explainer ('shap' or 'lime')
            
        Returns:
            Global robustness evaluation results
        """
        print("[INFO] Evaluating Global Robustness...")
        
        if explainer is None:
            print("Warning: No explainer provided, returning zero robustness")
            return {
                'overall_score': 0.0,
                'subset_jaccard_similarity': 0.0,
                'subset_importance_variance': 0.0,
                'n_subsets': 0
            }
        
        # Convert to numpy array
        if hasattr(X_samples, 'iloc'):
            X_array = X_samples.values
        else:
            X_array = np.asarray(X_samples)
        
        # Get original top-k features
        top_k = self.config['top_k_features']
        original_top_k = set(sorted(original_global_importance.keys(),
                                   key=lambda k: abs(original_global_importance[k]),
                                   reverse=True)[:top_k])
        
        jaccard_scores = []
        subset_importances = []  # Store importance vectors for variance calculation
        
        for i in range(n_subsets):
            # Random subset sampling
            subset_size = int(len(X_array) * subset_ratio)
            subset_indices = np.random.choice(len(X_array), size=subset_size, replace=False)
            X_subset = X_array[subset_indices]
            
            # Recompute global importance on subset
            subset_importance = self._compute_subset_global_importance(
                explainer, X_subset, explainer_type
            )
            
            if subset_importance is None or len(subset_importance) == 0:
                continue
            
            # Store for variance calculation
            subset_importances.append(subset_importance)
            
            # Compute Jaccard similarity of top-k features
            subset_top_k = set(sorted(subset_importance.keys(),
                                     key=lambda k: abs(subset_importance[k]),
                                     reverse=True)[:top_k])
            
            intersection = len(original_top_k & subset_top_k)
            union = len(original_top_k | subset_top_k)
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)
        
        # Compute variance of feature importance across subsets
        importance_variance = self._compute_subset_variance(
            subset_importances, original_global_importance
        )
        
        # Compute overall robustness score
        robustness_score = (
            np.mean(jaccard_scores) * 0.6 +
            (1 - importance_variance) * 0.4  # Lower variance is better
        )
        
        results = {
            'overall_score': robustness_score,
            'subset_jaccard_similarity': np.mean(jaccard_scores) if jaccard_scores else 0.0,
            'subset_importance_variance': importance_variance,
            'n_subsets': len(jaccard_scores),
            'subset_ratio': subset_ratio
        }
        
        self.evaluation_results['robustness'] = results
        return results
    
    def _compute_subset_global_importance(self, explainer, X_subset: np.ndarray,
                                         explainer_type: str) -> Optional[Dict]:
        """
        Compute global importance on a subset of samples
        
        Args:
            explainer: XAI explainer
            X_subset: Subset of samples
            explainer_type: Type of explainer
            
        Returns:
            Global importance dict or None if failed
        """
        try:
            if explainer_type.lower() == 'shap':
                # Ensure correct input format for SHAP
                if hasattr(explainer, '__class__') and 'DeepExplainer' in str(explainer.__class__):
                    X_subset = X_subset.astype('float32')
                
                shap_values = explainer.shap_values(X_subset)
                
                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    # Multi-class: average across classes
                    shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                elif shap_values.ndim == 3:
                    # 3D array: average across classes
                    shap_values = np.mean(np.abs(shap_values), axis=2)
                else:
                    shap_values = np.abs(shap_values)
                
                # Compute global importance: mean across samples
                if shap_values.ndim > 1:
                    global_importance = np.mean(shap_values, axis=0)
                else:
                    global_importance = shap_values
                
                return {int(idx): float(val) for idx, val in enumerate(global_importance)}
                
            elif explainer_type.lower() == 'lime':
                # For LIME, aggregate local explanations
                # Note: LIME is computationally expensive, limit subset size
                subset_size = min(50, len(X_subset))
                X_subset = X_subset[:subset_size]
                
                importance_sum = {}
                count = 0
                
                for x in X_subset:
                    try:
                        explanation = self._get_lime_explanation(
                            None, x, explainer, len(x)
                        )
                        if explanation:
                            for idx, val in explanation.items():
                                importance_sum[idx] = importance_sum.get(idx, 0.0) + abs(val)
                            count += 1
                    except Exception:
                        continue
                
                if count > 0:
                    return {idx: val / count for idx, val in importance_sum.items()}
                return {}
            else:
                return {}
                
        except Exception as e:
            print(f"Warning: Failed to compute subset global importance: {e}")
            return {}
    
    def _compute_subset_variance(self, subset_importances: List[Dict],
                                original_importance: Dict) -> float:
        """
        Compute variance of feature importance across subsets using coefficient of variation
        
        The coefficient of variation (CV = std/mean) is a better measure than variance/mean
        as it represents the relative standard deviation, which is more interpretable.
        
        Args:
            subset_importances: List of importance dicts from subsets
            original_importance: Original global importance
            
        Returns:
            Normalized variance score [0, 1], based on coefficient of variation
        """
        if len(subset_importances) < 2:
            return 0.0
        
        # Get all feature indices
        feature_indices = list(original_importance.keys())
        
        # Build importance matrix: (n_subsets, n_features)
        importance_matrix = []
        for subset_imp in subset_importances:
            importance_vector = [abs(subset_imp.get(idx, 0.0)) for idx in feature_indices]
            importance_matrix.append(importance_vector)
        
        importance_matrix = np.array(importance_matrix)
        
        # Compute standard deviation and mean for each feature
        feature_stds = np.std(importance_matrix, axis=0)
        mean_importances = np.mean(importance_matrix, axis=0)
        
        # Compute coefficient of variation (CV = std / mean) for each feature
        coefficients_of_variation = []
        for std, mean in zip(feature_stds, mean_importances):
            if mean > 1e-9:
                cv = std / mean
                coefficients_of_variation.append(cv)
        
        if not coefficients_of_variation:
            return 0.0
        
        # Average CV across features
        avg_cv = np.mean(coefficients_of_variation)
        
        # Normalize to [0, 1] range
        # CV of 0 = perfect stability (score 0)
        # CV of 0.5 or higher = high variability (score approaches 1)
        # Use a sigmoid-like transformation for smooth mapping
        normalized_variance = min(1.0, avg_cv / 0.5)
        
        return normalized_variance
    
    def evaluate_global_explanation(self, model, X_samples: np.ndarray,
                                   explainer, global_importance: Dict,
                                   y_samples=None, sample_size: int = 200,
                                   explainer_type: str = 'shap') -> Dict:
        """
        Complete evaluation for global explanation across all three dimensions
        
        Args:
            model: Model to be explained
            X_samples: Test samples array
            explainer: XAI explainer (SHAP or LIME)
            global_importance: Global feature importance dict
            y_samples: True labels (optional)
            sample_size: Number of samples for faithfulness evaluation
            explainer_type: Type of explainer ('shap' or 'lime')
            
        Returns:
            Complete global explanation evaluation results
        """
        print("[INFO] Starting Global Explanation Evaluation...")
        
        # 1. Faithfulness
        faithfulness_results = self.evaluate_global_faithfulness(
            model, X_samples, global_importance, y_samples, sample_size
        )
        
        # 2. Robustness
        robustness_results = self.evaluate_global_robustness(
            model, X_samples, explainer, global_importance, 
            n_subsets=30, subset_ratio=0.5, explainer_type=explainer_type
        )
        
        # 3. Complexity (same as local explanations)
        complexity_results = self.evaluate_complexity(
            [{'feature_importance': global_importance}]
        )
        
        # Calculate overall score
        overall_score = (
            faithfulness_results['overall_score'] * self.dimension_weights['faithfulness'] +
            robustness_results['overall_score'] * self.dimension_weights['robustness'] +
            complexity_results['overall_score'] * self.dimension_weights['complexity']
        )
        
        return {
            'overall_score': overall_score,
            'faithfulness': faithfulness_results,
            'robustness': robustness_results,
            'complexity': complexity_results,
            'evaluation_type': 'global'
        }


# # ============ Usage Example ============
# def example_usage():
#     """usage example"""
    
#     # create evaluator
#     evaluator = SimplifiedXAIEvaluator()
    
#     # simulate data
#     X_test = np.random.randn(100, 10)
#     y_test = np.random.randint(0, 2, 100)
    
#     # simulate explanation results
#     explanations = []
#     for i in range(100):
#         explanation = {
#             'feature_importance': {j: np.random.randn() for j in range(5)},
#             'text_explanation': f'This is explanation for sample {i}',
#             'rules': [f'if feature_{j} > 0.5 then class_1' for j in range(2)],
#         }
#         explanations.append(explanation)
    
#     # simulate model
#     class MockModel:
#         def predict_proba(self, X):
#             return np.random.rand(len(X), 2)
    
#     model = MockModel()
    
#     # execute evaluation
#     print("Starting simplified XAI quality assessment...")
    
#     # evaluate each dimension
#     evaluator.evaluate_faithfulness(model, explanations, X_test, y_test)
#     evaluator.evaluate_robustness(model, explanations, X_test)
#     evaluator.evaluate_complexity(explanations)
    
#     # generate report
#     print("\n" + "="*50)
#     print(evaluator.generate_evaluation_report())
    
#     # visualize results
#     evaluator.visualize_results()
    
#     return evaluator


# if __name__ == "__main__":
#     # run example
#     evaluator = example_usage()
