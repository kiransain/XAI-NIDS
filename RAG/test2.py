"""
Minimal RAGAS connectivity test.
Run this first to confirm RAGAS + Ollama works before running the full evaluation.

Requirements:
    pip install ragas openai datasets pandas
"""

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Point langchain-openai at local Ollama ─────────────────────────────────────
# Ollama exposes an OpenAI-compatible API — no real key needed.

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="llama3.2:1b",
        api_key="ollama",                        # dummy key
        base_url="http://localhost:11434/v1",    # local Ollama endpoint
        timeout=600,
        max_retries=2,
        stop=["</think>"],
    )
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="nomic-embed-text",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        check_embedding_ctx_length=False
    )
)

run_config = RunConfig(timeout=600, max_retries=2)

# ── Minimal single test sample ─────────────────────────────────────────────────

sample = SingleTurnSample(
    user_input="Is 13 PFCP heartbeats over 55 seconds normal or a DoS attack?",
    retrieved_contexts=[
        "PFCP Heartbeat messages are used for path management between UPF and SMF. "
        "They are sent at regular intervals of approximately 5 seconds to confirm "
        "the control plane connection is alive."
    ],
    response="13 heartbeats over 55 seconds is consistent with normal keep-alive "
             "signaling and does not indicate a DoS attack.",
    reference="A count of 13 heartbeats in ~55 seconds is consistent with standard "
              "audit intervals. This is a false positive — normal keep-alive signaling.",
)

dataset = EvaluationDataset(samples=[sample])

# ── Run evaluation ─────────────────────────────────────────────────────────────

print("Running RAGAS evaluation...")

metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=run_config,
)

print("\nResult:")
print(result)
print(result.to_pandas().to_string())