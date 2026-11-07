#Vector search prioritizes fast candidate discovery; reranking prioritizes precise relevance. FlashRank improves retrieval quality before the final context reaches the LLM.

import time
import logfire

# Ranker: Loads and runs the local reranking model.
# RerankRequest: Packages the query and candidate documents.
from flashrank import Ranker, RerankRequest


# Lazy initialization:
# Keep the model unloaded until reranking is actually needed.
# This avoids loading the model unnecessarily when the app starts.
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine only once.

    FlashRank uses a local ONNX reranking model.
    The default model is ms-marco-MiniLM-L-12-v2
    in standard FlashRank configurations.
    """

    global _ranker

    # If the model is already loaded, reuse it.
    if _ranker is None:

        logfire.info("🧠 Initializing FlashRank Model locally...")

        try:
            # Load the reranking model and cache its files
            # inside /tmp/flashrank.
            _ranker = Ranker(cache_dir="/tmp/flashrank")

        except Exception:
            # If the custom cache directory fails,
            # try FlashRank's default cache location.
            _ranker = Ranker()

    return _ranker


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5
) -> list[str]:

    """
    Takes documents retrieved from Qdrant and reranks them
    according to their relevance to the user's query.
    """

    # If Qdrant returned no documents, nothing to rerank.
    if not documents:
        return []

    # Start measuring reranking execution time.
    start_time = time.time()

    logfire.info(
        f"📡 [Reranker] Sending {len(documents)} docs to FlashRank..."
    )

    try:

        # Get the locally loaded reranking model.
        ranker = _get_ranker()

        # FlashRank expects passages as dictionaries.
        # Each passage needs an ID and its text.
        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        # Create a reranking request containing:
        # 1. Original user query
        # 2. Candidate documents retrieved from Qdrant
        request = RerankRequest(
            query=query,
            passages=passages
        )

        # IMPORTANT:
        # FlashRank sends the query-document pairs through
        # a Cross-Encoder model.
        #
        # Unlike vector search, it evaluates each query
        # together with each document.
        #
        # It assigns a relevance score to every passage
        # and returns the results sorted by score.
        results = ranker.rerank(request)

        # Keep only the highest-ranked top N documents.
        reranked_docs = []

        for res in results[:top_n]:

            # Extract the document text from each result.
            reranked_docs.append(res["text"])

        # Calculate total time taken by the reranker.
        duration = time.time() - start_time

        # Get the highest relevance score for logging.
        top_score = results[0]["score"] if results else "N/A"

        logfire.info(
            f"✅ [Reranker] Done in {duration:.2f}s. "
            f"Top relevance score: {top_score}"
        )

        # Return the most relevant documents to the RAG pipeline.
        return reranked_docs

    except Exception as e:

        # Log the error if reranking fails.
        logfire.error(
            f"❌ [Reranker] Reranking Failed: {e}"
        )

        # Fallback:
        # Return the original Qdrant ranking rather than
        # stopping the entire RAG pipeline.
        return documents[:top_n]