import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Data setup
data = {
    'Dataset': ['5GC-PFCP', '5G-NIDD', '5G-AD'],
    'Control': [3, 0, 0],
    'Retrieval': [3, 0, 0],
    'Faithfulness': [4, 3, 1],
    'Usefulness': [9, 2, 3]
}

df = pd.DataFrame(data)
df['Total'] = df['Control'] + df['Retrieval'] + df['Faithfulness'] + df['Usefulness']

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.2]})

# Chart 1: Total Queries per Dataset (Sorted Horizontal Bar)
df_sorted = df.sort_values(by='Total', ascending=True)
bars1 = ax1.barh(df_sorted['Dataset'], df_sorted['Total'], color='#1f77b4', edgecolor='black', alpha=0.85, height=0.55)

# ax1.set_title('Query Distribution by Dataset (Total = 28)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Number of Queries', fontsize=11, fontweight='bold')
ax1.set_xlim(0, 21)

# Enforce integer ticks on X-axis
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels next to bars
for bar in bars1:
    width = bar.get_width()
    ax1.text(width + 0.4, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
             va='center', ha='left', fontsize=11, fontweight='bold')

# Chart 2: Category Breakdown per Dataset (Stacked Bar)
categories = ['Control', 'Retrieval', 'Faithfulness', 'Usefulness']
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b0']

bottom = np.zeros(len(df))

for cat, color in zip(categories, colors):
    bars = ax2.bar(df['Dataset'], df[cat], bottom=bottom, label=cat, color=color, edgecolor='white', width=0.45)
    
    # Add values inside stacked segments
    for i, val in enumerate(df[cat]):
        if val > 0:
            ax2.text(i, bottom[i] + val/2, str(val), ha='center', va='center', 
                     color='white', fontweight='bold', fontsize=10)
    bottom += df[cat]

# ax2.set_title('Query Breakdown by Category & Dataset', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel('Number of Queries', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 21)

# Enforce integer ticks on Y-axis
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.legend(title='Category', loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()