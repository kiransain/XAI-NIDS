import pandas as pd
# calculate mean for NLG mertics and format for LaTeX table

df = pd.read_csv("nlg_results.csv")


metrics = ["rouge1", "rouge2", "rougeL", "bert_precision", "bert_recall", "bert_f1"]

summary = df.groupby("engine")[metrics].agg(['mean', 'std'])

def format_row(engine_name):
    row = summary.loc[engine_name]
    formatted_cols = []
    for m in metrics:
        
        m_val = row[(m, 'mean')]
        s_val = row[(m, 'std')]
        formatted_cols.append(f"${m_val:.3f}\\pm{s_val:.3f}$")
    return " & ".join(formatted_cols)


print("Copy these lines into your LaTeX tabular:")
for engine in ['bm25', 'vector', 'none']:
    print(f"{engine.upper()} & {format_row(engine)} \\\\")