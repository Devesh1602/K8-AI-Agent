#This function extracts text from PDFs using pypdf, retries failed pages with pdfplumber, monitors the process through Logfire, and returns the extracted content for downstream RAG processing.

import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF using pypdf.
    If some pages return no text, it tries pdfplumber as a fallback.
    """

    # Starts a Logfire span to monitor the entire PDF parsing operation.
    # filename=file_path records which PDF is being processed.
    with logfire.span("PDF Parsing (local)", filename=file_path):

        try:
            # Opens the PDF and creates a reader object.
            reader = PdfReader(file_path)

            # Gets the total number of pages in the PDF.
            total_pages = len(reader.pages)

            # Logs the number of pages in Logfire.
            logfire.info(f"PDF has {total_pages} pages.")

            # Stores successfully extracted text from pages.
            text_parts: list[str] = []

            # Stores page numbers from which pypdf could not extract text.
            blank_pages: list[int] = []

            # Iterates through every page in the PDF.
            for i, page in enumerate(reader.pages):

                # Extracts text from the current page.
                # If extraction returns None, uses an empty string instead.
                text = page.extract_text() or ""

                # Checks whether the extracted text contains actual content.
                if text.strip():

                    # If text exists, add it to the extracted text list.
                    text_parts.append(text)

                else:
                    # If the page is blank, save its page number.
                    # i starts from 0, so we add 1 for human-readable numbering.
                    blank_pages.append(i + 1)

            # FALLBACK:
            # If pypdf could not extract text from some pages,
            # attempt extraction using pdfplumber.
            if blank_pages:

                # Logs which pages need another extraction attempt.
                logfire.info(
                    f"pypdf returned blank on pages {blank_pages} "
                    "— retrying with pdfplumber."
                )

                try:
                    # Imports pdfplumber only when fallback is needed.
                    import pdfplumber

                    # Opens the same PDF using pdfplumber.
                    with pdfplumber.open(file_path) as pdf:

                        # Processes only the pages that pypdf returned blank.
                        for page_num in blank_pages:

                            # Retrieves the corresponding page.
                            # Subtracts 1 because Python indexing starts at 0.
                            page = pdf.pages[page_num - 1]

                            # Attempts to extract text from that page.
                            fallback_text = page.extract_text() or ""

                            # If pdfplumber successfully extracts text:
                            if fallback_text.strip():

                                # Adds the recovered text to the list.
                                text_parts.append(fallback_text)

                except Exception as plumber_err:
                    # Logs a warning if the fallback extraction fails.
                    # The main PDF parsing process can still continue.
                    logfire.warning(
                        f"pdfplumber fallback failed: {plumber_err}"
                    )

            # Combines all extracted page text into one string.
            # Adds a newline between individual page contents.
            full_text = "\n".join(text_parts)

            # Checks whether any meaningful text was extracted.
            if not full_text.strip():

                # Warns that the PDF may contain scanned images
                # instead of machine-readable text.
                logfire.warning(
                    f"No text extracted from {file_path}. "
                    "File may be fully image-based."
                )

            else:
                # Logs the total number of extracted characters.
                logfire.info(
                    f"Extracted {len(full_text)} characters from {file_path}."
                )

            # Returns the complete extracted text to the caller.
            return full_text

        except Exception as e:

            # Logs unexpected PDF parsing errors in Logfire.
            logfire.error(f"PDF Parse Failed for {file_path}: {e}")

            # Re-raises the original exception to the calling function.
            raise