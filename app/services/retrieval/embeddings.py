import time
# Used to pause execution between retries (time.sleep).

import logfire
# Used for monitoring, logging errors, and tracking embedding operations.

from langchain_google_genai import GoogleGenerativeAIEmbeddings
# LangChain integration for generating embeddings using Google Gemini.

from app.config import settings
# Imports application configuration, including the Gemini API key.


# Maximum number of texts processed in one batch.
BATCH_SIZE = 50

# Expected embedding dimensions for the selected Gemini model.
_GEMINI_DIM = 3072

# Expected embedding dimensions for the local fallback model.
_FALLBACK_DIM = 768  # all-mpnet-base-v2


# Stores the currently selected embedding model.
# Initially None because the model has not been initialized.
_active_model = None

# Tracks which model is currently active: "gemini" or "fallback".
_model_type: str | None = None


# ── MODEL INITIALISATION ───────────────────────────────────────

def _probe_gemini():
    """
    Checks whether Gemini is accessible.
    Returns the initialized model if successful, otherwise None.
    """
    try:
        # Creates a Gemini embedding model instance.
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )

        # Sends a small test request to verify API connectivity.
        model.embed_query("probe")

        # Logs successful initialization.
        logfire.info(
            "Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim)."
        )

        # Returns the working Gemini model.
        return model

    except Exception as e:

        # Logs the failure and indicates that fallback will be used.
        logfire.warning(
            f"Gemini probe failed: {e}. "
            "Will use sentence-transformers fallback."
        )

        # None indicates Gemini is unavailable.
        return None


def _load_fallback():
    """
    Loads a local embedding model if Gemini is unavailable.
    """
    # Imports SentenceTransformer only when fallback is required.
    from sentence_transformers import SentenceTransformer

    # Logs which local model is being loaded.
    logfire.info(
        "Loading sentence-transformers fallback "
        "(all-mpnet-base-v2, 768-dim)."
    )

    # Downloads/loads the model and returns its instance.
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """
    Initializes the embedding model once per process.
    Uses lazy initialization: model loads only when first needed.
    """
    global _active_model, _model_type

    # If a model is already initialized, do nothing.
    if _active_model is not None:
        return

    # First, check whether Gemini is available.
    gemini = _probe_gemini()

    if gemini:
        # If Gemini works, make it the active model.
        _active_model = gemini
        _model_type = "gemini"

    else:
        # Otherwise, load the local Sentence Transformers model.
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── PUBLIC HELPER ──────────────────────────────────────────────

def get_embedding_dim() -> int:
    """
    Returns the vector dimension of the currently active model.
    """
    # Ensure a model has been initialized before checking its dimensions.
    _init()

    # Gemini returns 3072 dimensions; fallback returns 768.
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── BATCH EMBEDDING WITH RETRY ──────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    """
    Generates embeddings for one batch of text.
    Handles Gemini rate limits using exponential backoff.
    """

    if _model_type == "gemini":

        # Attempt Gemini embedding generation up to 4 times.
        for attempt in range(4):

            try:
                # Generates embeddings for all texts in the batch.
                return _active_model.embed_documents(batch)

            except Exception as e:

                # Convert error message to lowercase for easier matching.
                err = str(e).lower()

                # Check whether the error is related to API rate limits.
                is_rate_limit = any(
                    x in err
                    for x in (
                        "429",
                        "rate",
                        "quota",
                        "resource_exhausted"
                    )
                )

                # Retry only for rate-limit errors and if attempts remain.
                if is_rate_limit and attempt < 3:

                    # Exponential backoff:
                    # Attempt 1: wait 1 second.
                    # Attempt 2: wait 2 seconds.
                    # Attempt 3: wait 4 seconds.
                    wait = 2 ** attempt

                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )

                    # Pause before trying again.
                    time.sleep(wait)

                else:
                    # Log the error if it cannot or should not be retried.
                    logfire.error(f"Gemini embedding failed: {e}")

                    # Propagate the exception to the calling function.
                    raise

        # Reached if all four attempts were exhausted.
        raise RuntimeError(
            "Gemini rate limit persisted after 4 attempts."
        )

    else:
        # Local fallback model generates embeddings.
        # encode() converts texts into numerical vectors.
        # show_progress_bar=False disables the progress display.
        # tolist() converts NumPy arrays into regular Python lists.
        return _active_model.encode(
            batch,
            show_progress_bar=False
        ).tolist()


# ── PUBLIC API ─────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Generates an embedding for a single user query.
    """
    # Initialize Gemini or fallback if not already initialized.
    _init()

    if _model_type == "gemini":

        # Generate one query embedding using Gemini.
        return _active_model.embed_query(query)

    # Generate one query embedding using the local model.
    # [0] extracts the first vector from the returned batch.
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for multiple documents using batches.
    """
    # Initialize the active embedding model.
    _init()

    # Stores embeddings generated across all batches.
    all_embeddings: list[list[float]] = []

    # Iterate through the input texts in groups of 50.
    # Example: 120 texts → batches of 50, 50 and 20.
    for i in range(0, len(texts), BATCH_SIZE):

        # Extract the current batch of texts.
        batch = texts[i : i + BATCH_SIZE]

        # Creates a Logfire span to monitor this batch.
        # Records model name, starting index and batch size.
        with logfire.span(
            "Embed batch",
            model=_model_type,
            start=i,
            size=len(batch)
        ):

            # Generate embeddings for this batch and append them
            # to the complete list of embeddings.
            all_embeddings.extend(_embed_batch(batch))

    # Return embeddings for all input texts.
    return all_embeddings