from experiment_set import EXPERIMENT_SET
import pandas as pd
import csv
n = []
ids = []

for experiment in EXPERIMENT_SET:
    ids.append(experiment["id"])
    n.append(experiment["query"]) 


df = pd.DataFrame(n, ids)
df.to_csv("HELPER_id_and_query.csv")