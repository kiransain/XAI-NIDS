import pandas as pd

sheet = pd.read_csv("human_eval_sheet_scored.csv")   # Nasim's filled-in sheet
key = pd.read_csv("human_eval_key.csv")               # private: id -> query_id, engine, judge scores
m = sheet.merge(key, on="id")
m.to_csv("human_vs_judge.csv", index=False)