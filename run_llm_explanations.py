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
explainer = llm_explanation_module.DeepseekLLMExplainer(api_key="sk-728829d9483141b386e566d3f68402d1")

if __name__ == "__main__":
    print("Starting LLM Explanation Phase...")
    # This matches the 'process_results_directory' you found!
    explainer.process_results_directory(results_dir="results", task_type="binary")