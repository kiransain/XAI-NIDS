#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Explanation Module for XAI Framework
Efficient Deepseek API integration for generating model explanations
"""

import os
import json
import time
import base64
import requests
import random
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepseekLLMExplainer:
    """
    Efficient LLM explainer using Deepseek API for XAI model explanations
    """
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", max_tokens: int = 4000, timeout_seconds: Optional[int] = None, max_retries: Optional[int] = None, backoff_base_seconds: Optional[float] = None):
        """
        Initialize the Deepseek LLM explainer
        
        Args:
            api_key: Deepseek API key
            model: Model name to use
            max_tokens: Maximum tokens per request
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Tunables (env overrides supported)
        self.timeout_seconds = int(os.getenv('DEEPSEEK_TIMEOUT', str(timeout_seconds or 60)))
        self.max_retries = int(os.getenv('DEEPSEEK_MAX_RETRIES', str(max_retries or 4)))
        self.backoff_base_seconds = float(os.getenv('DEEPSEEK_BACKOFF_BASE', str(backoff_base_seconds or 1.5)))
        self.max_samples = int(os.getenv('DEEPSEEK_MAX_SAMPLES', '3'))
        self.rate_limit_delay = float(os.getenv('DEEPSEEK_RATE_DELAY', '0.0'))
        
        # API usage tracking
        self.api_stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "successful_calls": 0,
            "failed_calls": 0
        }
        
        # Cache for explanations to avoid redundant API calls
        self.explanation_cache = {}
        # Persistent HTTP session
        self.session = requests.Session()
        
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.3) -> Optional[str]:
        """
        Make API call to Deepseek with error handling and retry logic
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            
        Returns:
            Response content or None if failed
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                self.api_stats["total_calls"] += 1
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)
                response = self.session.post(self.base_url, headers=self.headers, json=payload, timeout=self.timeout_seconds)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Track token usage
                    if "usage" in result:
                        self.api_stats["total_tokens"] += result["usage"]["total_tokens"]
                    
                    self.api_stats["successful_calls"] += 1
                    return content
                elif response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    wait_s = float(retry_after) if retry_after else self.backoff_base_seconds * (2 ** attempt)
                    logger.warning(f"Rate limited (429). Waiting {wait_s:.1f}s before retry...")
                    time.sleep(wait_s)
                    continue
                else:
                    logger.warning(f"API call failed (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    if attempt < self.max_retries - 1:
                        wait_s = self.backoff_base_seconds * (2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(wait_s)
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"API timeout (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    wait_s = self.backoff_base_seconds * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(wait_s)
            except Exception as e:
                logger.warning(f"API call error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    wait_s = self.backoff_base_seconds * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(wait_s)
        
        self.api_stats["failed_calls"] += 1
        return None
    
    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Encode image to base64 for API transmission
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string or None if failed
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to encode image {image_path}: {e}")
            return None
    
    def _get_model_performance_summary(self, model_dir: str) -> Dict[str, Any]:
        """
        Extract comprehensive model performance data from various files in the model directory
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            Dictionary with comprehensive performance data
        """
        performance = {
            "accuracy": None,
            "confusion_matrix": None,
            "classification_report": None,
            "feature_importance": None,
            "evaluation_reports": [],
            "radar_charts": [],
            "performance_metrics": {}
        }
        
        try:
            model_path = Path(model_dir)
            
            # Look for classification report (text files only)
            report_files = list(model_path.glob("*classification_report*.txt"))
            if not report_files:
                report_files = list(model_path.glob("*classification_report*.md"))
            if report_files:
                performance["classification_report"] = str(report_files[0])
                # Try to extract metrics from the report
                performance["performance_metrics"].update(
                    self._extract_metrics_from_report(report_files[0])
                )
            else:
                # If no text report found, just note the image file
                img_files = list(model_path.glob("*classification_report*.png"))
                if img_files:
                    performance["classification_report"] = str(img_files[0])
            
            # Look for confusion matrix
            cm_files = list(model_path.glob("*confusion_matrix*"))
            if cm_files:
                performance["confusion_matrix"] = str(cm_files[0])
            
            # Look for SHAP summary plots
            shap_files = list(model_path.glob("*shap_summary*"))
            if shap_files:
                performance["feature_importance"] = str(shap_files[0])
            
            # Skip XAI quality assessment files - not needed for LLM input

            # Additional numeric sources (prefer JSON when available)
            # 1) Confusion matrix values
            try:
                cm_json = model_path / "confusion_matrix.json"
                if cm_json.exists():
                    with open(cm_json, 'r', encoding='utf-8') as f:
                        performance["confusion_matrix_values"] = json.load(f)
            except Exception as _e:
                logger.warning(f"Failed reading confusion_matrix.json in {model_dir}: {_e}")

            # 2) SHAP top features
            try:
                shap_top_json = model_path / "shap_top_features.json"
                if shap_top_json.exists():
                    with open(shap_top_json, 'r', encoding='utf-8') as f:
                        performance["shap_top_features"] = json.load(f)
            except Exception as _e:
                logger.warning(f"Failed reading shap_top_features.json in {model_dir}: {_e}")

            # 3) LIME aggregated top features
            try:
                lime_top_json = model_path / "lime_top_features.json"
                if lime_top_json.exists():
                    with open(lime_top_json, 'r', encoding='utf-8') as f:
                        performance["lime_top_features"] = json.load(f)
            except Exception as _e:
                logger.warning(f"Failed reading lime_top_features.json in {model_dir}: {_e}")
                
        except Exception as e:
            logger.warning(f"Error extracting performance data from {model_dir}: {e}")
        
        return performance
    
    def _extract_metrics_from_report(self, report_file: Path) -> Dict:
        """Extract performance metrics from classification report"""
        metrics = {}
        try:
            if not report_file.exists():
                return metrics
                
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract accuracy if present
            import re
            accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', content, re.IGNORECASE)
            if accuracy_match:
                metrics['accuracy'] = float(accuracy_match.group(1))
            
            # Extract precision, recall, f1-score
            precision_match = re.search(r'precision\s+(\d+\.\d+)', content, re.IGNORECASE)
            if precision_match:
                metrics['precision'] = float(precision_match.group(1))
                
            recall_match = re.search(r'recall\s+(\d+\.\d+)', content, re.IGNORECASE)
            if recall_match:
                metrics['recall'] = float(recall_match.group(1))
                
            f1_match = re.search(r'f1-score\s+(\d+\.\d+)', content, re.IGNORECASE)
            if f1_match:
                metrics['f1_score'] = float(f1_match.group(1))
                
        except Exception as e:
            logger.warning(f"Error extracting metrics from {report_file}: {e}")
        
        return metrics
    
    def _extract_xai_metrics_from_reports(self, report_files: List[Path]) -> Dict:
        """Extract XAI evaluation metrics from evaluation reports"""
        metrics = {}
        try:
            for report_file in report_files:
                if not report_file.exists():
                    continue
                    
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract XAI scores
                import re
                faithfulness_match = re.search(r'Faithfulness.*?(\d+\.\d+)', content, re.IGNORECASE)
                if faithfulness_match:
                    metrics['faithfulness_score'] = float(faithfulness_match.group(1))
                
                robustness_match = re.search(r'Robustness.*?(\d+\.\d+)', content, re.IGNORECASE)
                if robustness_match:
                    metrics['robustness_score'] = float(robustness_match.group(1))
                
                complexity_match = re.search(r'Complexity.*?(\d+\.\d+)', content, re.IGNORECASE)
                if complexity_match:
                    metrics['complexity_score'] = float(complexity_match.group(1))
                
                overall_match = re.search(r'Overall.*?(\d+\.\d+)', content, re.IGNORECASE)
                if overall_match:
                    metrics['overall_xai_score'] = float(overall_match.group(1))
                    
        except Exception as e:
            logger.warning(f"Error extracting XAI metrics from reports: {e}")
        
        return metrics
    
    def _get_sample_data(self, model_dir: str, task_type: str, num_samples: int = 3) -> List[Dict]:
        """
        Extract comprehensive sample data for individual explanations
        
        Args:
            model_dir: Path to model directory
            task_type: 'binary' or 'multiclass'
            num_samples: Number of samples to extract
            
        Returns:
            List of sample data dictionaries with enhanced context
        """
        samples = []
        
        try:
            # Look for LIME individual text files first
            lime_text_files = list(Path(model_dir).glob("*lime_individual_*.txt"))
            logger.info(f"Found {len(lime_text_files)} LIME individual text files in {model_dir}")
            
            # Look for SHAP individual JSON files
            shap_json_files = list(Path(model_dir).glob("*shap_individual_*.json"))
            logger.info(f"Found {len(shap_json_files)} SHAP individual JSON files in {model_dir}")
            
            # Also look for individual evaluation results
            individual_eval_dir = Path(model_dir) / "evaluate"
            shap_individual_dir = individual_eval_dir / "shap_local_individual"
            lime_individual_dir = individual_eval_dir / "lime_local_individual"
            
            # Process LIME text files
            for i, lime_text_file in enumerate(lime_text_files[:num_samples]):
                sample_id = lime_text_file.stem.split('_')[-1]  # Extract sample ID
                
                # Look for corresponding SHAP JSON file
                shap_json_file = Path(model_dir) / f"shap_individual_{sample_id}.json"
                
                # Look for individual evaluation reports
                individual_report_files = []
                if shap_individual_dir.exists():
                    shap_reports = list(shap_individual_dir.glob(f"*sample*{sample_id}*.md"))
                    individual_report_files.extend(shap_reports)
                
                if lime_individual_dir.exists():
                    lime_reports = list(lime_individual_dir.glob(f"*sample*{sample_id}*.md"))
                    individual_report_files.extend(lime_reports)
                
                # Look for radar charts
                radar_charts = []
                if shap_individual_dir.exists():
                    radar_charts.extend(list(shap_individual_dir.glob(f"*sample*{sample_id}*radar.png")))
                if lime_individual_dir.exists():
                    radar_charts.extend(list(lime_individual_dir.glob(f"*sample*{sample_id}*radar.png")))
                
                # Try to extract sample metadata from individual evaluation reports
                sample_metadata = self._extract_sample_metadata(individual_report_files)
                
                # Load data for LIME (text) and SHAP (JSON)
                lime_data = None
                shap_data = None
                
                # Load LIME text data
                if lime_text_file.exists():
                    try:
                        with open(lime_text_file, 'r', encoding='utf-8') as f:
                            lime_data = f.read()
                        logger.info(f"Loaded LIME text data for sample {sample_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load LIME text for sample {sample_id}: {e}")
                
                # Load SHAP JSON data
                if shap_json_file.exists():
                    try:
                        with open(shap_json_file, 'r', encoding='utf-8') as f:
                            shap_data = json.load(f)
                        logger.info(f"Loaded SHAP data for sample {sample_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load SHAP JSON for sample {sample_id}: {e}")
                
                # Update metadata from SHAP data if available
                if shap_data:
                    sample_metadata.update({
                        'prediction_correct': shap_data.get('prediction') == shap_data.get('true_label'),
                        'true_label': shap_data.get('true_label'),
                        'predicted_label': shap_data.get('prediction')
                    })
                    if task_type == 'multiclass':
                        sample_metadata.update({
                            'predicted_class': shap_data.get('predicted_class'),
                            'true_class': shap_data.get('true_class')
                        })
                
                sample_data = {
                    "sample_id": sample_id,
                    "lime_data": lime_data,
                    "shap_data": shap_data,
                    "lime_explanation": str(lime_text_file),  # Keep for backward compatibility
                    "shap_force_plot": str(shap_json_file) if shap_json_file.exists() else None,  # Keep for backward compatibility
                    "individual_reports": [str(f) for f in individual_report_files],
                    "radar_charts": [str(f) for f in radar_charts],
                    "sample_metadata": sample_metadata,
                    "task_type": task_type
                }
                samples.append(sample_data)
                
        except Exception as e:
            logger.warning(f"Error extracting sample data from {model_dir}: {e}")
        
        return samples
    
    def _extract_sample_metadata(self, report_files: List[Path]) -> Dict:
        """
        Extract metadata from individual evaluation reports
        
        Args:
            report_files: List of report file paths
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "prediction_correct": None,
            "faithfulness_score": None,
            "robustness_score": None,
            "complexity_score": None,
            "overall_score": None
        }
        
        try:
            for report_file in report_files:
                if not report_file.exists():
                    continue
                    
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract prediction correctness
                if "prediction_correct" in content:
                    import re
                    match = re.search(r'"prediction_correct":\s*(true|false)', content)
                    if match:
                        metadata["prediction_correct"] = match.group(1) == "true"
                
                # Extract evaluation scores
                score_patterns = {
                    "faithfulness_score": r'"overall_score":\s*([\d.]+)',
                    "robustness_score": r'"overall_score":\s*([\d.]+)',
                    "complexity_score": r'"overall_score":\s*([\d.]+)',
                    "overall_score": r'"overall_score":\s*([\d.]+)'
                }
                
                for key, pattern in score_patterns.items():
                    if key not in metadata or metadata[key] is None:
                        match = re.search(pattern, content)
                        if match:
                            metadata[key] = float(match.group(1))
                            
        except Exception as e:
            logger.warning(f"Error extracting metadata from reports: {e}")
        
        return metadata
    
    def _extract_metadata_from_json(self, model_dir: str, sample_id: str) -> Dict:
        """
        Extract metadata from JSON evaluation files
        
        Args:
            model_dir: Path to model directory
            sample_id: Sample ID to look for
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        try:
            # Look for JSON files in individual evaluation directories
            individual_eval_dir = Path(model_dir) / "evaluate"
            shap_individual_dir = individual_eval_dir / "shap_local_individual"
            lime_individual_dir = individual_eval_dir / "lime_local_individual"
            
            # Check both SHAP and LIME individual directories for JSON files
            for eval_dir in [shap_individual_dir, lime_individual_dir]:
                if not eval_dir.exists():
                    continue
                    
                # Look for JSON files that might contain sample data
                json_files = list(eval_dir.glob("*results.json"))
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # If data is a list, look for the specific sample
                        if isinstance(data, list):
                            for item in data:
                                if str(item.get('sample_index')) == str(sample_id):
                                    metadata.update({
                                        'prediction_correct': item.get('prediction_correct'),
                                        'true_label': item.get('true_label'),
                                        'predicted_label': item.get('predicted_label'),
                                        'overall_score': item.get('overall_score'),
                                        'faithfulness_score': item.get('faithfulness', {}).get('overall_score'),
                                        'robustness_score': item.get('robustness', {}).get('overall_score'),
                                        'complexity_score': item.get('complexity', {}).get('overall_score')
                                    })
                                    break
                        
                        # If we found the metadata, break out of the loop
                        if metadata.get('prediction_correct') is not None:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error reading JSON file {json_file}: {e}")
                        continue
                
                # If we found the metadata, break out of the outer loop
                if metadata.get('prediction_correct') is not None:
                    break
                    
        except Exception as e:
            logger.warning(f"Error extracting metadata from JSON files: {e}")
        
        return metadata
    
    def generate_global_explanation(self, task_type: str, model_name: str, model_dir: str) -> Optional[Dict]:
        """
        Generate global explanation for a model
        
        Args:
            task_type: 'binary' or 'multiclass'
            model_name: Name of the model
            model_dir: Path to model directory
            
        Returns:
            Dictionary with global explanation or None if failed
        """
        # Check cache first
        cache_key = f"global_{task_type}_{model_name}"
        if cache_key in self.explanation_cache:
            logger.info(f"Using cached global explanation for {model_name}")
            return self.explanation_cache[cache_key]
        
        try:
            # Get model performance data
            performance = self._get_model_performance_summary(model_dir)
            
            # Prepare context for the API
            context_parts = [
                f"Task Type: {task_type} classification",
                f"Model: {model_name}",
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            
            # Add available files information
            available_files = []
            if performance["classification_report"]:
                available_files.append("Classification Report")
            if performance["confusion_matrix"]:
                available_files.append("Confusion Matrix")
            if performance["feature_importance"]:
                available_files.append("SHAP Feature Importance Plot")
            
            if available_files:
                context_parts.append(f"Available Analysis Files: {', '.join(available_files)}")
            
            # Create the prompt
            system_prompt = f"""You are an expert data scientist and cybersecurity analyst specializing in network intrusion detection. Your task is to provide comprehensive, data-driven global explanations for the {model_name} model.

**CRITICAL REQUIREMENTS:**
- Use ONLY the actual performance data provided - do not make assumptions about metrics
- Reference specific numbers, percentages, and values from the data
- Provide model-specific insights based on {model_name}'s architecture and characteristics
- Focus on actionable insights for cybersecurity operations

**ANALYSIS STRUCTURE:**
1. **Model Performance Analysis**: Quote exact accuracy, precision, recall, F1-scores from the data
2. **Feature Importance Deep Dive**: Analyze the most influential features with specific importance values
3. **Model-Specific Behavior**: Explain how {model_name} uniquely processes network traffic data
4. **Security Threat Coverage**: Which attack types this model excels at detecting vs. potential blind spots
5. **Operational Implications**: Specific recommendations for SOC teams and security operations
6. **Model Comparison Context**: How {model_name} compares to other ML approaches for this task

**WRITING STYLE:**
- Be specific and quantitative - use actual numbers from the data
- Avoid generic statements like "the model performs well"
- Provide concrete examples of attack patterns the model can/cannot detect
- Include specific deployment recommendations for cybersecurity teams
- Keep the analysis concise and well-structured with clear section headings
- Limit the total length to approximately 300 words"""

            # Build enhanced context with performance metrics
            enhanced_context = context_parts.copy()
            
            # Add performance metrics if available
            if performance.get("performance_metrics"):
                metrics = performance["performance_metrics"]
                enhanced_context.append("\n**Performance Metrics**:")
                if metrics.get('accuracy'):
                    enhanced_context.append(f"- Accuracy: {metrics['accuracy']:.3f}")
                if metrics.get('precision'):
                    enhanced_context.append(f"- Precision: {metrics['precision']:.3f}")
                if metrics.get('recall'):
                    enhanced_context.append(f"- Recall: {metrics['recall']:.3f}")
                if metrics.get('f1_score'):
                    enhanced_context.append(f"- F1-Score: {metrics['f1_score']:.3f}")
                # XAI quality scores removed - not needed for LLM input

            # Add confusion matrix numbers if available (binary or multiclass)
            if performance.get("confusion_matrix_values"):
                cm_vals = performance["confusion_matrix_values"]
                enhanced_context.append("\n**Confusion Matrix (values)**:")
                try:
                    if isinstance(cm_vals, dict) and all(k in cm_vals for k in ["tn","fp","fn","tp"]):
                        enhanced_context.append(f"- TN: {cm_vals['tn']}, FP: {cm_vals['fp']}, FN: {cm_vals['fn']}, TP: {cm_vals['tp']}")
                    else:
                        # Fallback for matrix format
                        enhanced_context.append(f"- Matrix: {cm_vals}")
                except Exception:
                    enhanced_context.append(f"- Matrix: {cm_vals}")

            # Add SHAP/LIME top features if available
            def _format_top_features(feat_obj, label):
                try:
                    if isinstance(feat_obj, list):
                        # list of {feature, importance}
                        parts = [f"{it.get('feature')}: {float(it.get('importance')):.4f}" for it in feat_obj[:10] if 'feature' in it and 'importance' in it]
                        if parts:
                            enhanced_context.append(f"\n**{label} (Top Features)**:")
                            enhanced_context.append("- " + "; ".join(parts))
                    elif isinstance(feat_obj, dict):
                        # mapping feature->importance
                        items = list(feat_obj.items())[:10]
                        parts = [f"{k}: {float(v):.4f}" for k, v in items]
                        if parts:
                            enhanced_context.append(f"\n**{label} (Top Features)**:")
                            enhanced_context.append("- " + "; ".join(parts))
                except Exception:
                    pass

            if performance.get("shap_top_features"):
                _format_top_features(performance["shap_top_features"], "SHAP Importance")
            if performance.get("lime_top_features"):
                _format_top_features(performance["lime_top_features"], "LIME Importance")
            
            # XAI quality assessment info removed - not needed for LLM input

            user_prompt = f"""Please analyze the following {task_type} classification model:

{chr(10).join(enhanced_context)}

**ANALYSIS REQUIREMENTS:**
Based on the comprehensive performance data and visualizations provided, generate a detailed global explanation that includes:

1. **Quantitative Performance Analysis**: Use the exact metrics provided to analyze model performance
2. **Feature Importance Deep Dive**: Analyze the most influential features with specific importance values from SHAP plots
3. **Model-Specific Behavior Analysis**: Explain how {model_name} uniquely processes network traffic data
4. **Security Threat Coverage**: Which attack types this model excels at detecting vs. potential blind spots
5. **Operational Deployment Recommendations**: Specific recommendations for SOC teams and security operations
6. **Model Comparison Context**: How {model_name} compares to other ML approaches for network intrusion detection

**CRITICAL**: Reference the specific performance metrics provided. Focus on practical deployment considerations for cybersecurity teams. Keep the response concise (around 300 words) and use clear section headings and bullet points where appropriate."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make API call
            response = self._make_api_call(messages)
            
            if response:
                explanation = {
                    "model_name": model_name,
                    "task_type": task_type,
                    "explanation_type": "global",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "performance_data": performance
                }
                
                # Cache the result
                self.explanation_cache[cache_key] = explanation
                return explanation
            
        except Exception as e:
            logger.error(f"Error generating global explanation for {model_name}: {e}")
        
        return None
    
    def generate_individual_explanation(self, sample_data: Dict, model_name: str, task_type: str) -> Optional[Dict]:
        """
        Generate individual explanation for a specific sample
        
        Args:
            sample_data: Dictionary with sample information
            model_name: Name of the model
            task_type: 'binary' or 'multiclass'
            
        Returns:
            Dictionary with individual explanation or None if failed
        """
        # Check cache first
        cache_key = f"individual_{task_type}_{model_name}_{sample_data['sample_id']}"
        if cache_key in self.explanation_cache:
            logger.info(f"Using cached individual explanation for {model_name} sample {sample_data['sample_id']}")
            return self.explanation_cache[cache_key]
        
        try:
            # Get prediction correctness from metadata
            metadata = sample_data.get('sample_metadata', {})
            prediction_correct = metadata.get('prediction_correct')
            
            # Create different prompts based on prediction correctness
            if prediction_correct is True:
                system_prompt = f"""You are an expert cybersecurity analyst and incident response specialist. Your task is to provide detailed explanations for a CORRECTLY PREDICTED network traffic sample analyzed by the {model_name} model.

**CRITICAL REQUIREMENTS:**
- This sample was CORRECTLY classified by the model
- Explain WHY the model's prediction was correct and what this means in real-world security context
- Analyze the ACTUAL feature values and SHAP/LIME weights that led to the correct classification
- Provide insights into the model's decision-making process for this successful prediction

**ANALYSIS STRUCTURE FOR CORRECT PREDICTIONS:**
1. **Traffic Pattern Analysis**: What specific network behavior this sample represents and why it was correctly identified
2. **Model Decision Validation**: Why the model's classification was accurate based on feature analysis
3. **Feature Impact Analysis**: Which exact features correctly indicated the threat/benign nature and their security significance
4. **Real-World Security Implications**: What this correct prediction means for network security operations
5. **Model Performance Insights**: How this successful prediction demonstrates the model's effectiveness
6. **Operational Value**: How this type of correct detection benefits security teams

**WRITING STYLE:**
- Emphasize the model's successful threat detection capabilities
- Explain the security value of correctly identifying this type of traffic
- Provide specific feature values that led to correct classification
- Highlight the practical benefits for security operations
- Keep the analysis concise and well-structured with clear section headings
- Limit the total length to approximately 300 words"""
            
            elif prediction_correct is False:
                system_prompt = f"""You are an expert cybersecurity analyst and incident response specialist. Your task is to provide detailed explanations for an INCORRECTLY PREDICTED network traffic sample analyzed by the {model_name} model.

**CRITICAL REQUIREMENTS:**
- This sample was INCORRECTLY classified by the model
- Explain WHY the model's prediction was wrong and what factors led to the misclassification
- Analyze the ACTUAL feature values and SHAP/LIME weights that caused the incorrect prediction
- Provide insights into model limitations and potential improvements

**ANALYSIS STRUCTURE FOR INCORRECT PREDICTIONS:**
1. **Traffic Pattern Analysis**: What the sample actually represents vs. what the model predicted
2. **Model Error Analysis**: Why the model made an incorrect classification based on feature analysis
3. **Feature Impact Analysis**: Which features misled the model and their actual security significance
4. **Root Cause Analysis**: What characteristics of this sample caused the model to fail
5. **Model Limitations**: What this error reveals about the model's blind spots or weaknesses
6. **Improvement Recommendations**: How the model could be improved to handle similar cases

**WRITING STYLE:**
- Focus on understanding why the model failed
- Explain the discrepancy between actual and predicted behavior
- Provide specific feature values that led to incorrect classification
- Suggest model improvements and additional training needs
- Keep the analysis concise and well-structured with clear section headings
- Limit the total length to approximately 300 words"""
            
            else:
                # Fallback for cases where prediction correctness is unknown
                system_prompt = f"""You are an expert cybersecurity analyst and incident response specialist. Your task is to provide detailed, actionable individual explanations for a network traffic sample analyzed by the {model_name} model.

**CRITICAL REQUIREMENTS:**
- Analyze the ACTUAL feature values and SHAP/LIME weights provided
- Provide specific, actionable recommendations for incident response
- Reference exact feature values and their security implications
- Focus on immediate threat assessment and response actions

**ANALYSIS STRUCTURE:**
1. **Traffic Pattern Analysis**: What specific network behavior this sample represents based on feature values
2. **Threat Classification**: Specific attack type or suspicious activity pattern identified
3. **Feature Impact Analysis**: Which exact features triggered the alert and their security significance
4. **Risk Severity Assessment**: Immediate vs. long-term security implications
5. **Incident Response Actions**: Specific steps for containment, investigation, and remediation
6. **False Positive Assessment**: Indicators that this might be legitimate traffic

**WRITING STYLE:**
- Be specific about feature values (e.g., "duration=0.05s, src_bytes=0, dst_bytes=1460")
- Provide concrete next steps for security teams
- Include specific indicators of compromise (IOCs) if applicable
- Reference standard incident response procedures and tools
- Keep the analysis concise and well-structured with clear section headings
- Limit the total length to approximately 300 words"""

            # Build enhanced context with available data
            context_parts = [
                f"**Sample ID**: {sample_data['sample_id']}",
                f"**Model**: {model_name}",
                f"**Task Type**: {task_type} classification"
            ]
            
            # Add sample metadata if available
            metadata = sample_data.get('sample_metadata', {})
            if metadata:
                context_parts.append("\n**Sample Evaluation Results**:")
                if metadata.get('prediction_correct') is not None:
                    context_parts.append(f"- Prediction Correct: {metadata['prediction_correct']}")
                if metadata.get('true_label') is not None:
                    context_parts.append(f"- True Label: {metadata['true_label']}")
                if metadata.get('predicted_label') is not None:
                    context_parts.append(f"- Predicted Label: {metadata['predicted_label']}")
                if task_type == 'multiclass':
                    if metadata.get('true_class'):
                        context_parts.append(f"- True Class: {metadata['true_class']}")
                    if metadata.get('predicted_class'):
                        context_parts.append(f"- Predicted Class: {metadata['predicted_class']}")
            
            # Add feature values and explanations from JSON data
            lime_data = sample_data.get('lime_data')
            shap_data = sample_data.get('shap_data')
            
            if lime_data:
                context_parts.append("\n**LIME Explanation Data**:")
                # LIME data is now text format, so we can include it directly
                if isinstance(lime_data, str):
                    context_parts.append(lime_data)
                else:
                    # Fallback for old JSON format
                    feature_values = lime_data.get('feature_values', {})
                    if feature_values:
                        context_parts.append("- Feature Values:")
                        for feature, value in list(feature_values.items())[:10]:  # Show first 10 features
                            context_parts.append(f"  • {feature}: {value}")
                    
                    feature_importance = lime_data.get('feature_importance', [])
                    if feature_importance:
                        context_parts.append("- LIME Feature Importance (Top 10):")
                        for item in feature_importance[:10]:
                            context_parts.append(f"  • {item['feature']}: {item['lime_weight']:.4f} (value: {item['feature_value']})")
            
            if shap_data:
                context_parts.append("\n**SHAP Explanation Data**:")
                feature_values = shap_data.get('feature_values', {})
                if feature_values:
                    context_parts.append("- Feature Values:")
                    for feature, value in list(feature_values.items())[:10]:  # Show first 10 features
                        context_parts.append(f"  • {feature}: {value}")
                
                feature_importance = shap_data.get('feature_importance', [])
                if feature_importance:
                    context_parts.append("- SHAP Feature Importance (Top 10):")
                    for item in feature_importance[:10]:
                        context_parts.append(f"  • {item['feature']}: {item['shap_value']:.4f} (value: {item['feature_value']})")
                
                base_value = shap_data.get('base_value')
                if base_value is not None:
                    context_parts.append(f"- SHAP Base Value: {base_value:.4f}")
            
            # Add fallback information if JSON data is not available
            if not lime_data and not shap_data:
                context_parts.append("\n**Available Analysis Files**:")
                context_parts.append(f"- LIME Explanation: {sample_data.get('lime_explanation', 'Not available')}")
                if sample_data.get('shap_force_plot'):
                    context_parts.append(f"- SHAP Force Plot: {sample_data['shap_force_plot']}")
                if sample_data.get('individual_reports'):
                    context_parts.append(f"- Individual Evaluation Reports: {len(sample_data['individual_reports'])} files")

            # Create different user prompts based on prediction correctness
            if prediction_correct is True:
                user_prompt = f"""Please analyze this CORRECTLY PREDICTED {task_type} classification sample:

{chr(10).join(context_parts)}

**ANALYSIS REQUIREMENTS FOR CORRECT PREDICTION:**
Based on the comprehensive explanation data for this correctly classified sample, provide a detailed analysis that includes:

1. **Traffic Pattern Analysis**: What specific network behavior this sample represents and why the model correctly identified it
2. **Model Success Validation**: Why the model's classification was accurate and what features led to the correct decision
3. **Feature Impact Analysis**: Which exact features correctly indicated the threat/benign nature and their security significance (reference specific values)
4. **Real-World Security Value**: What this correct prediction means for network security operations and threat detection
5. **Model Performance Insights**: How this successful prediction demonstrates the model's effectiveness and reliability
6. **Operational Benefits**: How this type of correct detection benefits security teams and incident response
7. **Threat Intelligence**: What this correct classification reveals about the attack pattern or benign behavior

**CRITICAL**: Emphasize the model's successful threat detection capabilities and the security value of correctly identifying this type of traffic.

Keep the response concise (around 300 words) and use clear section headings and bullet points where appropriate."""
            
            elif prediction_correct is False:
                user_prompt = f"""Please analyze this INCORRECTLY PREDICTED {task_type} classification sample:

{chr(10).join(context_parts)}

**ANALYSIS REQUIREMENTS FOR INCORRECT PREDICTION:**
Based on the comprehensive explanation data for this misclassified sample, provide a detailed analysis that includes:

1. **Traffic Pattern Analysis**: What the sample actually represents vs. what the model incorrectly predicted
2. **Model Error Analysis**: Why the model made an incorrect classification and what factors led to the misclassification
3. **Feature Impact Analysis**: Which features misled the model and their actual security significance (reference specific values)
4. **Root Cause Analysis**: What characteristics of this sample caused the model to fail and make the wrong decision
5. **Model Limitations**: What this error reveals about the model's blind spots, weaknesses, or training gaps
6. **Improvement Recommendations**: How the model could be improved to handle similar cases correctly
7. **False Positive/Negative Impact**: What the consequences of this misclassification are for security operations

**CRITICAL**: Focus on understanding why the model failed and what can be learned from this error.

Keep the response concise (around 300 words) and use clear section headings and bullet points where appropriate."""
            
            else:
                # Fallback for cases where prediction correctness is unknown
                user_prompt = f"""Please analyze this {task_type} classification sample:

{chr(10).join(context_parts)}

**ANALYSIS REQUIREMENTS:**
Based on the comprehensive explanation data for this specific sample, provide a detailed analysis that includes:

1. **Traffic Pattern Analysis**: What specific network behavior this sample represents based on the feature values and explanation data
2. **Threat Classification**: Specific attack type or suspicious activity pattern identified, with confidence level
3. **Feature Impact Analysis**: Which exact features triggered the alert and their security significance (reference specific values)
4. **Model Performance Assessment**: How well the model performed on this sample
5. **Risk Severity Assessment**: Immediate vs. long-term security implications with specific threat levels
6. **Incident Response Actions**: Specific, actionable steps for containment, investigation, and remediation
7. **False Positive Assessment**: Indicators that this might be legitimate traffic vs. confirmed threat

**CRITICAL**: Focus on practical threat assessment and incident response actions based on the model's prediction and feature analysis.

Keep the response concise (around 300 words) and use clear section headings and bullet points where appropriate."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make API call
            response = self._make_api_call(messages)
            
            if response:
                explanation = {
                    "model_name": model_name,
                    "task_type": task_type,
                    "sample_id": sample_data['sample_id'],
                    "explanation_type": "individual",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "sample_data": sample_data
                }
                
                # Cache the result
                self.explanation_cache[cache_key] = explanation
                return explanation
            
        except Exception as e:
            logger.error(f"Error generating individual explanation for {model_name} sample {sample_data['sample_id']}: {e}")
        
        return None
    
    def process_model_directory(self, model_dir: str, task_type: str, model_name: str) -> Dict[str, Any]:
        """
        Process a single model directory and generate all explanations
        
        Args:
            model_dir: Path to model directory
            task_type: 'binary' or 'multiclass'
            model_name: Name of the model
            
        Returns:
            Dictionary with all generated explanations
        """
        logger.info(f"Processing model: {model_name} ({task_type})")
        
        results = {
            "model_name": model_name,
            "task_type": task_type,
            "model_dir": model_dir,
            "global_explanation": None,
            "individual_explanations": [],
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Generate global explanation
        logger.info(f"Generating global explanation for {model_name}")
        global_explanation = self.generate_global_explanation(task_type, model_name, model_dir)
        if global_explanation:
            results["global_explanation"] = global_explanation
            logger.info(f"✅ Global explanation generated for {model_name}")
        else:
            logger.warning(f"❌ Failed to generate global explanation for {model_name}")
        
        # Generate individual explanations
        logger.info(f"Generating individual explanations for {model_name}")
        sample_data_list = self._get_sample_data(model_dir, task_type, num_samples=self.max_samples)
        
        for sample_data in sample_data_list:
            individual_explanation = self.generate_individual_explanation(sample_data, model_name, task_type)
            if individual_explanation:
                results["individual_explanations"].append(individual_explanation)
                logger.info(f"✅ Individual explanation generated for {model_name} sample {sample_data['sample_id']}")
            else:
                logger.warning(f"❌ Failed to generate individual explanation for {model_name} sample {sample_data['sample_id']}")
        
        return results
    
    def process_results_directory(self, results_dir: str, task_type: str) -> Dict[str, Any]:
        """
        Process all models in a results directory
        
        Args:
            results_dir: Path to results directory
            task_type: 'binary' or 'multiclass'
            
        Returns:
            Dictionary with all model explanations
        """
        logger.info(f"Processing {task_type} results directory: {results_dir}")
        
        all_results = {
            "task_type": task_type,
            "results_dir": results_dir,
            "models": {},
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Get all model directories
        results_path = Path(results_dir)
        if not results_path.exists():
            logger.error(f"Results directory does not exist: {results_dir}")
            return all_results
        
        model_dirs = [d for d in results_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            logger.info(f"Processing model directory: {model_name}")
            
            # Process the model
            model_results = self.process_model_directory(str(model_dir), task_type, model_name)
            all_results["models"][model_name] = model_results
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """
        Save explanation results to files
        
        Args:
            results: Results dictionary
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results as JSON
        json_file = output_path / f"{results['task_type']}_llm_explanations.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved complete results to: {json_file}")
        
        # Save individual model explanations
        for model_name, model_results in results["models"].items():
            model_output_dir = output_path / model_name
            model_output_dir.mkdir(exist_ok=True)
            
            # Save global explanation
            if model_results["global_explanation"]:
                global_file = model_output_dir / "global_explanation.md"
                with open(global_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {model_name} Global Explanation\n\n")
                    f.write(f"**Task Type**: {model_results['task_type']}\n")
                    f.write(f"**Generated**: {model_results['global_explanation']['timestamp']}\n\n")
                    f.write("## Analysis\n\n")
                    f.write(model_results['global_explanation']['content'])
                logger.info(f"Saved global explanation to: {global_file}")
            
            # Save individual explanations - one file per sample
            if model_results["individual_explanations"]:
                # Create individual samples directory
                samples_dir = model_output_dir / "individual_samples"
                samples_dir.mkdir(exist_ok=True)
                
                for explanation in model_results["individual_explanations"]:
                    # Create individual sample file
                    sample_file = samples_dir / f"sample_{explanation['sample_id']}.md"
                    with open(sample_file, 'w', encoding='utf-8') as f:
                        f.write(f"# Sample {explanation['sample_id']} - {model_name} Analysis\n\n")
                        f.write(f"**Task Type**: {model_results['task_type']}\n")
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
                    logger.info(f"Saved individual sample explanation to: {sample_file}")
                
                # Also create a summary file with all samples for reference
                summary_file = model_output_dir / "individual_explanations_summary.md"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {model_name} Individual Explanations Summary\n\n")
                    f.write(f"**Task Type**: {model_results['task_type']}\n")
                    f.write(f"**Generated**: {model_results['processing_timestamp']}\n")
                    f.write(f"**Total Samples**: {len(model_results['individual_explanations'])}\n\n")
                    f.write("## Sample Files\n\n")
                    f.write("Individual sample explanations are saved in separate files:\n\n")
                    for explanation in model_results["individual_explanations"]:
                        f.write(f"- [Sample {explanation['sample_id']}](individual_samples/sample_{explanation['sample_id']}.md)\n")
                    f.write("\n## Quick Overview\n\n")
                    for explanation in model_results["individual_explanations"]:
                        f.write(f"### Sample {explanation['sample_id']}\n")
                        f.write(f"**Generated**: {explanation['timestamp']}\n\n")
                        # Add first few lines of content as preview
                        content_preview = explanation['content'][:200] + "..." if len(explanation['content']) > 200 else explanation['content']
                        f.write(f"{content_preview}\n\n")
                        f.write("---\n\n")
                logger.info(f"Saved individual explanations summary to: {summary_file}")
            
            # Save model-specific JSON
            model_json_file = model_output_dir / "model_explanations.json"
            with open(model_json_file, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved model results to: {model_json_file}")
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """
        Get API usage statistics
        
        Returns:
            Dictionary with API usage statistics
        """
        stats = self.api_stats.copy()
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
            stats["average_tokens_per_call"] = stats["total_tokens"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0
            stats["average_tokens_per_call"] = 0
        
        return stats
    
    def clear_cache(self):
        """Clear the explanation cache"""
        self.explanation_cache.clear()
        logger.info("Explanation cache cleared")
