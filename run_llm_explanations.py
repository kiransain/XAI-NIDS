'''
Note: This file (run_llm_explanations.py) does not exist in the original code contrary as stated in 
README.md.
(#### Option 2: LLM Explanations Only
```bash
python run_llm_explanations.py
```
Generate LLM explanations for existing model results.)
I created this file myself on 25.02.2026.
DeepSeek API key is missing.

'''
import llm_explanation_module
import os

# 1. API Key
# 2. activating explainer module
explainer = llm_explanation_module.DeepseekLLMExplainer(timeout_seconds=10000)

if __name__ == "__main__":
    # Point directly to the folder containing the actual files
    target_path = "results/5GC_PFCP/binary/DecisionTree"
    
    # Use process_model_directory instead of process_results_directory
    # to bypass the folder-looping logic
    results = explainer.process_model_directory(
        model_dir=target_path, 
        task_type="binary", 
        model_name="DecisionTree"
    )
    
    explainer.save_results(results, output_dir="results/5GC_PFCP/binary/DecisionTree/LLM")