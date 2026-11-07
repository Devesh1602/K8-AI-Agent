import logfire

# QdrantClient: Connects our application to the Qdrant vector database.
from qdrant_client import QdrantClient

# models: Provides Qdrant data structures and query configurations.
from qdrant_client.http import models

# Loads application configuration, including Qdrant URL, API key and collection.
from app.config import settings

# Converts the user's natural-language query into an embedding vector.
from app.services.retrieval.embeddings import embed_query


# Initialize Qdrant Client.
# Establishes a connection to our Qdrant Cloud database using credentials.
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Searches the enterprise knowledge base using semantic vector search.

    query: User's natural-language question.
    limit: Maximum number of matching documents to retrieve.
    """

    try:

        # STEP 1: EMBEDDING
        # Convert the user's query into a numerical vector.
        # This uses the same embedding model used during document ingestion.
        query_vector = embed_query(query)

        # STEP 2: VECTOR SEARCH
        # Search the configured Qdrant collection.
        #
        # Qdrant compares the query vector against stored document vectors
        # using the collection's configured distance metric (e.g. cosine).
        #
        # limit=8 means retrieve the 8 most similar results.
        # with_payload=True tells Qdrant to return document metadata too.
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        # STEP 3: EXTRACT RESULTS
        # Create a clean list to store the retrieved documents.
        results = []

        # Iterate through the matching points returned by Qdrant.
        for res in response.points:

            results.append({

                # Actual document chunk stored in Qdrant.
                "content": res.payload.get("text", ""),

                # Original file/document from which the chunk came.
                "source": res.payload.get("source", "Unknown"),

                # Similarity score assigned by Qdrant.
                "score": res.score
            })

        # STEP 4: RETURN
        # Return the retrieved document chunks and their metadata.
        # These results can now be passed to the FlashRank reranker.
        return results

    except Exception as e:

        # Log any embedding or Qdrant search errors.
        logfire.error(f"❌ Qdrant Search Failed: {e}")

        # Return an empty list instead of crashing the application.
        return []