#This function converts raw HTML into clean, readable plain text using BeautifulSoup, preparing it for downstream RAG ingestion.
from bs4 import BeautifulSoup
# Imports BeautifulSoup to parse HTML and extract readable content.

import logfire
# Imports Logfire for monitoring and tracking the parsing process.


def parse_html(file_path: str):
    """
    Parses HTML files, removes unnecessary elements,
    and extracts clean text for RAG.
    """

    # Starts a Logfire span to monitor the HTML parsing operation.
    # filename records which HTML file is being processed.
    with logfire.span("📄 HTML Parsing", filename=file_path):

        try:
            # Opens the HTML file in read mode.
            # UTF-8 handles text encoding.
            # errors='ignore' skips characters that cannot be decoded.
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:

                # Reads the entire HTML file as a string.
                content = f.read()

            # Parses raw HTML and creates a structured HTML tree.
            # "html.parser" is Python's built-in HTML parser.
            soup = BeautifulSoup(content, "html.parser")

            # 1. REMOVE UNNECESSARY ELEMENTS
            # Finds script, style, meta and noscript tags.
            for script in soup(["script", "style", "meta", "noscript"]):

                # Completely removes these elements and their contents
                # from the HTML tree.
                script.decompose()

            # 2. EXTRACT TEXT
            # Extracts readable text from the remaining HTML.
            # separator="\n" adds a newline between text elements.
            text = soup.get_text(separator="\n")

            # 3. CLEAN WHITESPACE

            # Splits text into lines and removes leading/trailing spaces.
            # Generator expression processes lines one at a time.
            lines = (line.strip() for line in text.splitlines())

            # Splits lines containing double spaces into smaller phrases.
            # Also removes unnecessary surrounding whitespace.
            chunks = (
                phrase.strip()
                for line in lines
                for phrase in line.split("  ")
            )

            # Joins non-empty phrases using newline characters.
            # This produces clean, readable text.
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)

            # Returns cleaned HTML content as plain text.
            # This text can now be passed to the RAG chunking stage.
            return text_clean

        except Exception as e:

            # Records HTML parsing errors in Logfire.
            logfire.error(f"❌ HTML Parse Failed: {e}")

            # Re-raises the original exception to the calling function.
            raise