import matplotlib.pyplot as plt
import numpy as np

# metrics = ['precision@3', 'recall@3', 'hit@3', 'mrr', 'ndcg@3']
# bm25 = [1.0, 1.0, 1.0, 1.0, 1.0]
# vector = [0.3333333333333333, 0.3333333333333333, 1.0, 1.0, 0.46927872602275644]

metrics = ['AR', 'CP', 'CR', 'Faithfulness']

none = [0.60, 0.00, 0.00, 0.00]
bm25 = [0.65, 0.70, 0.60, 0.45]
vector = [0.60, 0.65, 0.55, 0.40]

x = np.arange(len(metrics))  
width = 0.25  # smaller width for 3 bars

fig, ax = plt.subplots()

ax.bar(x - width, none, width, label='None')
ax.bar(x, bm25, width, label='BM25')
ax.bar(x + width, vector, width, label='Vector')

ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Evaluation Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.0)
ax.legend()

plt.tight_layout()
plt.show()


 

