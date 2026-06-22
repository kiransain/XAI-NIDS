'''
this script computes ROUGE and BERTScore (NGL metrics)
'''

from rouge_score import rouge_scorer
import bert_score
import json
import csv


with open("main_thesis_evaluation_results.json", "r") as t:      
    results = json.load(t)

rows = []
actual_answer = []
reference = []

for r in results: 
    answer = r["answer"]
    ground_truth = r["ground_truth"]
    actual_answer.append(answer)
    reference.append(ground_truth)
  
P, R, F1 = bert_score.score(actual_answer, reference, lang="en")

for i, r in enumerate(results):
    query_id = r["query_id"]
    engine = r["engine"]
    answer = r["answer"]
    ground_truth = r["ground_truth"]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, answer)

    row = {
        "query_id": query_id,
        "engine": engine,
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "bert_precision": P[i].item(),
        "bert_recall": R[i].item(),
        "bert_f1": F1[i].item()
    }

    rows.append(row)

with open("evaluation_results.csv", "w", newline="") as csvfile:
    fieldnames = ["query_id", "engine", "rouge1", "rouge2", "rougeL", "bert_precision", "bert_recall", "bert_f1"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print("Evaluation completed and results saved to evaluation_results.csv")
