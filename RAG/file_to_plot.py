import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ir_metrics_summary.csv')

# Remove bad row
df = df[df['engine'] != 'none']

# Convert each column manually
df['precision@3'] = df['precision@3'].astype(float)
df['recall@3'] = df['recall@3'].astype(float)
df['hit@3'] = df['hit@3'].astype(float)
df['mrr'] = df['mrr'].astype(float)
df['ndcg@3'] = df['ndcg@3'].astype(float)

# Set engine as index
df = df.set_index('engine')

# Plot
df.plot(kind='bar')

plt.ylabel('Score')
plt.title('Evaluation Metrics by Engine')
plt.show()


# df = pd.read_csv('ir_metrics_summary.csv')
# df = df[df['engine'] != 'none'] # only engines with scores plotted; none not considered
# df.iloc[:, 1:] #[row, col] = df.iloc[:, 1:].astype(float) # all rows starting from col 1

# # Set engine as index
# df.set_index('engine', inplace=True)

# # df.T.plot(kind='bar')
# # plt.ylabel('Score')
# # plt.title('Metrics Comparison')
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.show()

# df.plot(kind='bar')
# plt.ylabel('Score')
# plt.title('Evaluation Metrics by Engine')
# plt.xticks(rotation=0)
# plt.legend(title='Metrics')
# plt.tight_layout()
# plt.show()