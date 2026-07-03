# XAI-NIDS RAG Experimental Setup

This repository contains the Retrieval-Augmented Generation (RAG) experimental pipeline created for this thesis. It evaluates and benchmarks different retrieval strategies (No RAG, BM25, and Vector Search) on a cybersecurity knowledge base to improve the explainability of Network Intrusion Detection Systems (NIDS).

## Prerequisites & Stack

- **Python:** v3.12.0
- **Base LLM:** DeepSeek-r1-8b (managed locally via Ollama v0.6.1)
- **Vector Database:** ChromaDB v1.5.5
- **Orchestration Framework:** LlamaIndex v0.14.16

### Core Python Dependencies
```bash
pip install llamaindex==0.14.16 chromadb==1.5.5 shap==0.50.0 lime==0.2.0.1 bert-score==0.3.13 rouge_score==0.1.2 pandas==2.3.3
```

### Project Structure
```
RAG/
├── data_analysis/         # Scripts for plotting and statistics
├── evaluation_dataset/    # SHAP and LIME files from NIDS system
├── knowledge_base/        # Reference texts, 3GPP standards, guidelines
├── thesis_db/             # Local ChromaDB vector storage files
├── experiment_set.py      # Definition of the 28 evaluation queries
├── run_benchmark.py       # Main pipeline execution script
├── index_documents.py     # Ingestion and chunk indexing script
├── ir_metrics.py          # Calculates Precision, Recall, MRR, NDCG, Hit
├── nlg_eval.py            # Calculates BERTScore and ROUGE-N
├── friedman_script.py     # Performs statistical tests (Wilcoxon, Friedman)
├── prompt.py              # System and user prompt templates
├── generate_qrels.py      # Utility to compile relevance judgments
├── qrels.json             # Annotator-generated QREL file
└── inspect_chunks.py      # Development utility to search and filter chunks
```

## Getting Started
### 1. Ingest and Index Documents
Before running benchmarks, populate the vector database with the files in your knowledge_base/ directory:
```
python index_documents.py
```

### 2. Generate Relevance Judgments (QRELs)
Compile human-verified relevance boundaries required for information retrieval scoring:

```
python generate_qrels.py
```

### 3. Run the Evaluation Pipeline

Run the 28 experimental queries across all three modes (No RAG, BM25, and Vector Search). The script utilizes automatic checkpointing to prevent progress loss:

```
python run_benchmark.py
```
Output: Generates main_pipeline_raw_results.json.

### 4. Evaluate Outputs
Compute information retrieval (IR) performance and natural language generation (NLG) scores:

#### Compute IR Metrics (Precision, Recall, MRR, NDCG, Hit)
```
python ir_metrics.py
```

#### Compute NLG Metrics (BERTScore, ROUGE)
```
python nlg_eval.py
```
### 5. Compute LLM-as-a-Judge Metrics
To replicate or run the LLM-as-a-judge evaluation phase manually:
1. Open `prompt.py` and copy the standardized evaluation prompt template.
2. Paste the prompt into the **GPT 5.3** web interface alongside your pipeline outputs.
3. Save or manually copy the resulting JSON arrays into `main_pipeline_llm_as_judge_eval.json` for metric extraction.

*Note: If you wish to skip this step, the pre-computed, human-verified evaluation results are already structured and available in `evaluation_results.csv`.*

### 6. Statistical Significance
To calculate the Wilcoxon signed-rank tests, Friedman tests, and Bonferroni corrections on your metrics:

```
python friedman_script.py
```

## Ablation Studies
The pipeline includes scripts under the root directory to run the ablation configurations detailed in the evaluation chapter:

### Prompt Modification: To test system resilience using shortened prompts:

```
python ablation_prompt_change_run_benchmark.py
```
### LLM Modification: To evaluate outputs using Gemini 2.5 Flash via Google AI Studio:

```
python ablation_llm_change_run_benchmark.py
```
Note: This script requires you to obtain 3 free API keys from Google AI Studio and fill them in to enable automatic rotation and bypass free-tier rate limits.