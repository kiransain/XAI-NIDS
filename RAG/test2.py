import os
from deepeval.models import OllamaModel
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# 1. Initialize the weak model
# Using llama3.2:1b as the 'judge'
print("--- Initializing Ollama (Weak Judge) ---")
weak_judge = OllamaModel(
    model="deepseek-r1:8b",
    # Increase timeout for weak hardware
    timeout=300 
)

# 2. Define a simple Faithfulness Metric
# Faithfulness checks if the 'actual_output' is derived from 'retrieval_context'
metric = FaithfulnessMetric(
    threshold=0.5, 
    model=weak_judge,
    async_mode=False # Force sequential to prevent crashing weak CPUs/GPUs
)

# 3. Create a dummy 5G Security test case
test_case = LLMTestCase(
    input="What is the function of the SEPP in 5G?",
    actual_output="The SEPP manages security between different carrier networks (roaming).",
    retrieval_context=[
        "The Security Edge Protection Proxy (SEPP) is a transparent proxy that "
        "implements application layer security for inter-PLMN roaming interfaces."
    ]
)

# 4. Run the test
print("\n--- Running Evaluation (This may take a minute) ---")
try:
    metric.measure(test_case)
    print(f"\nSUCCESS!")
    print(f"Score: {metric.score}")
    print(f"Reasoning: {metric.reason}")
except Exception as e:
    print(f"\nFAILED: The weak LLM likely failed to output valid JSON.")
    print(f"Error Details: {str(e)}")
