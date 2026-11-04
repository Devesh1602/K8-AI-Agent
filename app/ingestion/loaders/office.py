#This function automatically identifies Office document formats, extracts their content using Unstructured, combines it into plain text, and uses Logfire to monitor the process.

import logfire
# Imports Logfire for monitoring, logging and tracking execution.

from unstructured.partition.auto import partition
# Imports Unstructured's automatic document parser.
# It detects the document format (DOCX, PPTX, etc.)
# and extracts its content.


def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using Unstructured.
    """

    # Starts a Logfire span to monitor the entire parsing operation.
    # filename records which document is being processed.
    with logfire.span("📄 Office Document Parsing", filename=file_path):

        try:
            # Automatically detects the file type and extracts
            # document elements such as paragraphs, titles and text.
            elements = partition(filename=file_path)

            # Converts every extracted element into a string
            # and combines them into one text string.
            # "\n" separates individual elements with newlines.
            full_text = "\n".join([str(el) for el in elements])

            # Checks whether the extracted document contains any text.
            if not full_text.strip():

                # Logs a warning if no meaningful text was extracted.
                logfire.warning(
                    f"⚠️ Unstructured returned empty text for {file_path}"
                )

            else:
                # Logs the number of successfully extracted characters.
                logfire.info(
                    f"✅ Successfully parsed {len(full_text)} characters"
                )

            # Returns the complete extracted document text
            # for downstream processing (chunking and embeddings).
            return full_text

        except Exception as e:

            # Records parsing errors in Logfire for debugging.
            logfire.error(f"❌ Office Parse Failed: {e}")

            # Re-raises the exception so the calling function
            # knows that document parsing failed.
            raise