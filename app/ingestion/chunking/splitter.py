#This function divides a large document into smaller paragraph-based text chunks so that each chunk can be independently converted into embeddings and stored in Qdrant for efficient retrieval.
from typing import List
# Imports List to specify that the function returns a list of strings.

import logfire
# Imports Logfire for monitoring and tracking the chunking process.


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Splits text into smaller chunks based on paragraphs.
    Tries to keep each chunk within the specified character limit.
    """

    # Starts a Logfire span to monitor the chunking operation.
    # text_length records the total input text length.
    with logfire.span("✂️ Text Chunking", text_length=len(text)):

        # Checks whether the input contains meaningful text.
        # If empty, returns an empty list instead of processing it.
        if not text.strip():
            return []

        # Splits the input text wherever two consecutive newlines occur.
        # This separates the document into individual paragraphs.
        paragraphs = text.split("\n\n")

        # Stores the final generated chunks.
        chunks = []

        # Temporarily stores paragraphs being combined into one chunk.
        current_chunk = ""

        # Iterates through each paragraph.
        for p in paragraphs:

            # Checks whether adding the next paragraph keeps the
            # current chunk below the specified character limit.
            if len(current_chunk) + len(p) < chunk_size:

                # If it fits, append the paragraph to the current chunk.
                current_chunk += p + "\n\n"

            else:
                # If it doesn't fit, save the existing chunk
                # provided it contains meaningful text.
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start a new chunk with the current paragraph.
                current_chunk = p + "\n\n"

        # After the loop, save the remaining chunk if it is not empty.
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Removes any empty or whitespace-only chunks.
        valid_chunks = [c for c in chunks if c.strip()]

        # Logs how many valid chunks were generated.
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")

        # Returns the final list of text chunks.
        return valid_chunks