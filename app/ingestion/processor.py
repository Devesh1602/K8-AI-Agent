# Implemented a universal document ingestion pipeline that supports multiple file formats, including PDF, HTML, TXT, DOCX and PPTX. 
# It automatically identifies document types, extracts text using format-specific parsers, divides the content into smaller chunks, and saves processed metadata locally as JSON. 
# It then generates vector embeddings using Gemini with a local Sentence Transformers fallback and indexes those vectors in Qdrant. 
# I also integrated Logfire for end-to-end observability, error tracking and performance monitoring.

import os
# Provides functions for directory traversal, file paths and file operations.

import sys
# Allows reading command-line arguments and exiting the program.

import uuid
# Generates unique IDs for every vector stored in Qdrant.

import json
# Converts Python dictionaries into JSON files for local storage.

import logfire
# Monitors the ingestion pipeline, execution time, events and errors.


from qdrant_client import QdrantClient
# Imports the Qdrant client to communicate with the vector database.

from qdrant_client.http import models
# Provides Qdrant data structures such as PointStruct and VectorParams.


from app.config import settings
# Imports configuration settings such as Qdrant URL, API key and collection name.

from app.services.retrieval.embeddings import embed_texts, get_embedding_dim
# embed_texts(): Generates embeddings for multiple text chunks.
# get_embedding_dim(): Returns the dimensions of the currently active embedding model.

from app.ingestion.loaders.pdf import parse_pdf
# Imports the PDF text extraction function.

from app.ingestion.loaders.html import parse_html
# Imports the HTML text extraction function.

from app.ingestion.loaders.text import parse_text
# Imports the plain-text file extraction function.

from app.ingestion.chunking.splitter import chunk_text
# Imports the function that splits large text into smaller chunks.


# Configures Logfire for this application.
# All monitoring events are associated with this service name.
logfire.configure(service_name="enterprise-ingestion-service")


# Local directory where parsed and chunked document metadata is saved.
# Example: processed_data/general/document.pdf.json
PROCESSED_DATA_DIR = "processed_data"


# ── QDRANT INITIALIZATION ──────────────────────────────────────

# Creates a connection to the Qdrant vector database.
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,          # Qdrant server/cloud URL.
    api_key=settings.QDRANT_API_KEY,  # Authentication API key.
)


# ── 1. SAVE PROCESSED DATA LOCALLY ─────────────────────────────

def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """
    Saves parsed document chunks and metadata as a JSON file.
    """

    # Creates a folder based on the source type.
    # Example: processed_data/true/
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)

    # Creates the folder if it does not already exist.
    # exist_ok=True prevents errors if the folder already exists.
    os.makedirs(folder, exist_ok=True)

    # Creates the destination JSON filename.
    # Example: processed_data/true/document.pdf.json
    dest = os.path.join(folder, f"{filename}.json")

    # Opens the destination file in write mode.
    with open(dest, "w", encoding="utf-8") as f:

        # Converts the Python dictionary into formatted JSON.
        # ensure_ascii=False preserves characters from different languages.
        # indent=2 makes the JSON easier to read.
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Returns the location where the JSON file was saved.
    return dest


# ── 2. PROCESS A SINGLE FILE ────────────────────────────────────

def process_file(file_path: str, filename: str, source_type: str):
    """
    Complete workflow for one document:
    Parse → Chunk → Save → Embed → Store in Qdrant.
    """

    # Starts a Logfire span to monitor processing of this file.
    # Records the filename and its source category.
    with logfire.span("Processing File", file=filename, source=source_type):

        try:
            # STEP 1: DETERMINE FILE TYPE

            # Converts the filename to lowercase and extracts its extension.
            # Example: REPORT.PDF → pdf
            ext = filename.lower().rsplit(".", 1)[-1]

            # Selects the appropriate parser based on file extension.

            if ext == "pdf":
                # Extracts text from PDF files.
                full_text = parse_pdf(file_path)

            elif ext in ("html", "htm"):
                # Extracts readable text from HTML files.
                full_text = parse_html(file_path)

            elif ext == "txt":
                # Reads plain-text files.
                full_text = parse_text(file_path)

            elif ext in ("docx", "pptx"):
                # Imports the Office parser only when needed.
                # Extracts text from Word and PowerPoint files.
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)

            else:
                # Skips unsupported file formats.
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            # Checks whether parsing produced meaningful text.
            # If empty, there is nothing to chunk or embed.
            if not full_text or not full_text.strip():
                logfire.warning(
                    f"No text extracted from {filename} — skipping."
                )
                return


            # STEP 2: CHUNK THE EXTRACTED TEXT

            # Splits the complete document into smaller text chunks.
            # Uses the default chunk size configured in chunk_text().
            chunks = chunk_text(full_text)

            # If no chunks were generated, stop processing this file.
            if not chunks:
                return


            # STEP 3: SAVE PROCESSED DATA LOCALLY

            # Creates a dictionary containing document information.
            processed_data = {
                "filename": filename,       # Original document name.
                "source_type": source_type, # Document category.
                "chunks": chunks,           # List of extracted text chunks.
            }

            # Saves the dictionary as a JSON file locally.
            local_path = save_processed_locally(
                processed_data, source_type, filename
            )

            # Logs the location of the saved JSON file.
            logfire.info(f"Saved processed data → {local_path}")


            # STEP 4: GENERATE EMBEDDINGS AND INDEX IN QDRANT

            # Starts a separate monitoring span for vectorization and indexing.
            with logfire.span("Vectorizing & Indexing"):

                # Converts all text chunks into numerical embedding vectors.
                # Uses Gemini or the configured local fallback model.
                embeddings = embed_texts(chunks)

                # Creates one Qdrant point for every chunk and its embedding.
                points = [
                    models.PointStruct(

                        # Generates a unique ID for this vector record.
                        id=str(uuid.uuid4()),

                        # Stores the numerical embedding vector.
                        vector=vector,

                        # Stores additional information alongside the vector.
                        payload={
                            "text": chunk,             # Original chunk text.
                            "source": filename,        # Original document name.
                            "source_type": source_type # Document category.
                        },
                    )

                    # Pairs each text chunk with its corresponding embedding.
                    for chunk, vector in zip(chunks, embeddings)
                ]

                # Inserts or updates all generated points in Qdrant.
                # If a point with the same ID exists, it is updated.
                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )

                # Logs the number of vectors successfully indexed.
                logfire.info(
                    f"Indexed {len(points)} points to Qdrant from {filename}."
                )

        except Exception as e:

            # Logs any error occurring during this file's processing.
            logfire.error(f"Failed to process {filename}: {e}")

            # Note: The exception is swallowed here, so processing
            # can continue with the next file.


# ── 3. PROCESS AN ENTIRE DIRECTORY ──────────────────────────────

def process_directory(dir_path: str, source_type: str):
    """
    Processes every file directly inside a directory.
    """

    # Monitors the directory scanning operation.
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):

        # Gets all entries that are actual files, excluding subdirectories.
        files = [
            f for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]

        # Logs the number of files discovered.
        logfire.info(f"Found {len(files)} files in {dir_path}.")

        # Processes each discovered file individually.
        for filename in files:

            # Builds the complete file path and sends it to process_file().
            process_file(
                os.path.join(dir_path, filename),
                filename,
                source_type
            )


# ── 4. UNIVERSAL INGESTION ORCHESTRATOR ─────────────────────────

def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: str = None,
    wipe: bool = False
):
    """
    Main ingestion controller.

    Scans the base directory, determines document categories,
    creates the Qdrant collection if necessary,
    and processes all documents.
    """

    # Tracks the complete ingestion execution in Logfire.
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # STEP 1: OPTIONALLY DELETE EXISTING COLLECTION

        if wipe:

            # Creates a monitoring span for the deletion operation.
            with logfire.span("Wiping Collection"):

                # Checks whether the collection already exists.
                if qdrant_client.collection_exists(
                    settings.QDRANT_COLLECTION
                ):

                    # Deletes the existing collection and its stored vectors.
                    qdrant_client.delete_collection(
                        settings.QDRANT_COLLECTION
                    )

                    logfire.info(
                        f"Collection '{settings.QDRANT_COLLECTION}' deleted."
                    )


        # STEP 2: CREATE COLLECTION IF IT DOES NOT EXIST

        if not qdrant_client.collection_exists(
            settings.QDRANT_COLLECTION
        ):

            # Determines the embedding dimension dynamically.
            # This also initializes the embedding model if necessary.
            dim = get_embedding_dim()

            # Creates a new Qdrant collection.
            qdrant_client.create_collection(

                # Name of the collection where vectors will be stored.
                collection_name=settings.QDRANT_COLLECTION,

                # Defines the vector configuration.
                vectors_config=models.VectorParams(

                    # Every vector must have this many dimensions.
                    size=dim,

                    # Uses cosine similarity to compare vectors.
                    distance=models.Distance.COSINE,
                ),
            )

            # Logs successful collection creation.
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )


        # STEP 3: IDENTIFY SUBDIRECTORIES

        # Finds all immediate subdirectories inside the base directory.
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]


        # STEP 4: DECIDE HOW TO PROCESS THE DIRECTORY

        # CASE A: No subdirectories exist.
        # Treat the base directory itself as a source.
        if not subdirs:

            if explicit_source_type:

                # Uses the source type explicitly supplied by the user.
                source_type = explicit_source_type

            else:

                # Uses the directory name to infer its source category.
                base_name = os.path.basename(
                    os.path.normpath(base_dir)
                ).lower()

                # Assigns a category based on the directory name.
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )

            # Logs the selected source category.
            logfire.info(
                f"No sub-folders found — processing "
                f"'{base_dir}' as '{source_type}'."
            )

            # Processes all files in the base directory.
            process_directory(base_dir, source_type)


        # CASE B: Subdirectories exist.
        # Treat each subdirectory as a separate source category.
        else:

            # Iterates through each subdirectory.
            for subdir in subdirs:

                # Determines the category from the subdirectory name.
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )

                # Processes all files inside the current subdirectory.
                process_directory(
                    os.path.join(base_dir, subdir),
                    source_type
                )


# ── 5. PROGRAM ENTRY POINT ──────────────────────────────────────

# This block runs only when this Python file is executed directly
# or through python -m, not when it is imported by another module.

if __name__ == "__main__":

    # Example commands:
    #
    # python -m app.ingestion.processor DATA --wipe
    # python -m app.ingestion.processor DATA/true_data true

    # Checks whether the user passed the --wipe argument.
    # If present, the existing Qdrant collection will be deleted.
    wipe_requested = "--wipe" in sys.argv

    # Removes --wipe from the argument list.
    # This keeps directory and source-type arguments separate.
    clean_args = [a for a in sys.argv if a != "--wipe"]

    # Gets the target directory from the command line.
    # Defaults to "DATA" if no directory was supplied.
    target_dir = (
        clean_args[1] if len(clean_args) > 1 else "DATA"
    )

    # Gets an optional source category from the command line.
    # Defaults to None if not supplied.
    explicit_type = (
        clean_args[2] if len(clean_args) > 2 else None
    )

    # Checks whether the target directory exists.
    if not os.path.exists(target_dir):

        # Prints an error if the path does not exist.
        print(f"Error: path '{target_dir}' does not exist.")

        # Stops the program with exit status 1 (failure).
        sys.exit(1)

    # Starts the complete ingestion workflow.
    run_universal_ingestion(
        target_dir,
        explicit_source_type=explicit_type,
        wipe=wipe_requested
    )

    # Logs completion of the ingestion job.
    logfire.info("Ingestion job completed.")