import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the manual evaluation results file
with open("generation_eval_results_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. Extract metrics and convert to numeric
metrics = [
    'faithfulness', 'correctness', 'answer_relevancy', 
    'security_specificity', 'context_utilization_score', 'hallucinated_security_reasoning'
]
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors='coerce')

# 3. Compute group means and align logical engine layout
means = df.groupby('engine')[metrics].mean()
engines_order = ['none', 'bm25', 'vector']
means_ordered = means.loc[engines_order]

# 4. Configure descriptive presentation labels
metric_labels = [
    'Faithfulness\n(Context Alignment)',
    'Correctness\n(Ground Truth Alignment)',
    'Answer Relevancy\n(Query Alignment)',
    'Security Specificity\n(Technical Depth)',
    'Context Utilization\n(CUS Score)',
    'No Hallucinations\n(Higher = Fewer Invented Claims)'
]

df_plot = means_ordered.T
df_plot.index = metric_labels

# 5. Initialize layout canvas without calling .figure()
fig, ax = plt.subplots(figsize=(12, 8))

# Palette: Subtle Red for Baseline, Technical Blue/Purple for Active Engines
colors = ['#e06666', '#6fa8dc', '#8e7cc3']

# Render grouped horizontal bars
df_plot.plot(kind='barh', ax=ax, color=colors, width=0.8, edgecolor='black')

# Overlay precision data value labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=5, fontsize=9)

# Formatting axes and gridlines
ax.set_title('5G Network Security RAG Evaluation: Retrieval Engine Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Average Score (0-2 Scale)', fontsize=12, labelpad=10)
ax.set_xlim(0, 2.3)
ax.set_xticks(np.arange(0, 2.1, 0.5))
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.legend(['No Retrieval (none)', 'BM25 Retrieval', 'Vector Retrieval'], loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('rag_engine_comparison.png', dpi=300)