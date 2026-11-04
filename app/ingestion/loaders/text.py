#This function reads a text file and returns its content, while Logfire monitors the operation and records any errors.
import logfire  # Imports Logfire for monitoring, logging and tracing execution.

def parse_text(file_path: str):
    """
    Parses plain text files and returns their content.
    """

    # Creates a monitoring span in Logfire.
    # "Text Parsing" identifies this operation in the monitoring dashboard.
    # filename records which file is being processed.
    with logfire.span("📄 Text Parsing", filename=file_path):

        try:
            # Opens the file in read mode ('r').
            # UTF-8 handles standard text encoding.
            # errors='ignore' skips characters that cannot be decoded.
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:

                # Reads the entire file content and returns it as a string.
                return f.read()

        except Exception as e:
            # Records the error in Logfire for debugging and monitoring.
            logfire.error(f"❌ Text Parse Failed: {e}")

            # Re-raises the exception so the calling function knows parsing failed.
            raise e