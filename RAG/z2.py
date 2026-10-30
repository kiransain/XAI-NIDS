from experiment_set import EXPERIMENT_SET
import csv

with open("paper_query_and_id.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    writer.writerow(["id", "query"])
    for i in EXPERIMENT_SET:
        writer.writerow([i["id"], i["query"]])

print("CSV completed")


