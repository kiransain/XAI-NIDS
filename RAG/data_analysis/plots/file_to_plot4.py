import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
 
# Data: [Pre-ablation, Prompt-change, LLM-change]
data = {
    'Faithfulness': {
        'No RAG':  [0.444, 0.222, 0.000],
        'BM25':    [0.889, 0.778, 0.89],
        'Vector':  [0.556, 0.667, 0.78],
    },
    'Correctness': {
        'No RAG':  [0.556, 1.222, 0.89],
        'BM25':    [0.667, 1.222, 1.22],
        'Vector':  [0.222, 0.889, 1.11],
    },
    'Hallucination': {
        'No RAG':  [0.222, 0.889, 0.11],
        'BM25':    [0.556, 0.889, 0.67],
        'Vector':  [0.333, 0.889, 0.33],
    },
}
 
conditions = ['Pre-ablation', 'Prompt change', 'LLM change']
engines = ['No RAG', 'BM25', 'Vector']
metrics = list(data.keys())
 
# Grayscale shades per engine
grays = ['0.85', '0.45', '0.15']
hatches = ['', '///', '...']
 
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
 
x = np.arange(len(conditions))
n = len(engines)
width = 0.22
offsets = np.array([-1, 0, 1]) * width
 
for ax, metric in zip(axes, metrics):
    for i, (engine, gray, hatch) in enumerate(zip(engines, grays, hatches)):
        vals = data[metric][engine]
        bars = ax.bar(x + offsets[i], vals, width=width,
                      color=gray, edgecolor='black', linewidth=0.7,
                      hatch=hatch, label=engine)
 
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Mean score (0–2)', fontsize=8)
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xlabel(metric, fontsize=9, labelpad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
 
# Single legend below
handles = [
    mpatches.Patch(facecolor=g, edgecolor='black', hatch=h, label=e)
    for e, g, h in zip(engines, grays, hatches)
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
 
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()