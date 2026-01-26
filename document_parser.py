"""
Document Parser Module

Extracts text and tables from PDF documents using pdfplumber.
Designed for financial document analysis with production-ready error handling.

Usage:
    from document_parser import extract_document_content

    result = extract_document_content("financial_report.pdf")
    print(result['text'])      # List of text per page
    print(result['tables'])    # List of DataFrames
    print(result['metadata'])  # page_count, filename
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
from pdfplumber.pdf import PDF

# Configure module logger
logger = logging.getLogger(__name__)


class PDFParsingError(Exception):
    """Custom exception for PDF parsing errors."""
    pass


class EmptyPDFError(PDFParsingError):
    """Raised when a PDF has no pages or no extractable content."""
    pass


class CorruptedPDFError(PDFParsingError):
    """Raised when a PDF file is corrupted or cannot be read."""
    pass


def extract_document_content(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text and tables from a PDF document.

    This function parses a PDF file and extracts all text content (organized by page)
    and all tables (as pandas DataFrames). It includes comprehensive error handling
    for common PDF issues.

    Args:
        pdf_path: Path to the PDF file to parse. Can be absolute or relative path.

    Returns:
        Dictionary containing:
            - 'text': List[str] - Text content from each page (index 0 = page 1)
            - 'tables': List[pd.DataFrame] - All tables found in the document
            - 'metadata': Dict with 'page_count' (int) and 'filename' (str)

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        CorruptedPDFError: If the PDF file is corrupted or cannot be parsed.
        EmptyPDFError: If the PDF has no pages.
        PDFParsingError: For other parsing-related errors.

    Example:
        >>> result = extract_document_content("report.pdf")
        >>> print(f"Pages: {result['metadata']['page_count']}")
        >>> print(f"First page text: {result['text'][0][:100]}")
        >>> print(f"Tables found: {len(result['tables'])}")
    """
    # Validate file path
    file_path = Path(pdf_path)
    logger.info(f"Starting PDF extraction: {file_path}")

    if not file_path.exists():
        error_msg = f"PDF file not found: {file_path.absolute()}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not file_path.is_file():
        error_msg = f"Path is not a file: {file_path.absolute()}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if file_path.suffix.lower() != '.pdf':
        error_msg = f"File does not have .pdf extension: {file_path.name}"
        logger.warning(error_msg)
        # Continue anyway - might still be a valid PDF

    # Initialize result containers
    text_by_page: List[str] = []
    tables: List[pd.DataFrame] = []

    # Attempt to open and parse the PDF
    try:
        logger.debug(f"Opening PDF file: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            logger.info(f"PDF opened successfully. Page count: {page_count}")

            # Check for empty PDF
            if page_count == 0:
                error_msg = f"PDF has no pages: {file_path.name}"
                logger.error(error_msg)
                raise EmptyPDFError(error_msg)

            # Process each page
            for page_num, page in enumerate(pdf.pages, start=1):
                logger.debug(f"Processing page {page_num}/{page_count}")

                # Extract text from page
                page_text = _extract_page_text(page, page_num)

                # Only append if page has meaningful content (>50 words after cleaning)
                if page_text and len(page_text.split()) > 50:
                    text_by_page.append(page_text)

                # Extract tables from page
                page_tables = _extract_page_tables(page, page_num)
                tables.extend(page_tables)

            logger.info(
                f"Extraction complete. Text pages: {len(text_by_page)}/{page_count}, "
                f"Tables found: {len(tables)}"
            )

    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as e:
        error_msg = f"PDF file is corrupted or has invalid syntax: {file_path.name}. Error: {str(e)}"
        logger.error(error_msg)
        raise CorruptedPDFError(error_msg) from e

    except pdfplumber.pdfminer.pdfdocument.PDFEncryptionError as e:
        error_msg = f"PDF file is encrypted and cannot be read: {file_path.name}"
        logger.error(error_msg)
        raise CorruptedPDFError(error_msg) from e

    except Exception as e:
        # Catch any other pdfplumber/pdfminer errors
        if "PDF" in type(e).__name__ or "pdf" in str(e).lower():
            error_msg = f"Failed to parse PDF: {file_path.name}. Error: {str(e)}"
            logger.error(error_msg)
            raise CorruptedPDFError(error_msg) from e
        raise

    # Check if we extracted any content
    total_text = sum(len(t) for t in text_by_page)
    if total_text == 0 and len(tables) == 0:
        logger.warning(
            f"PDF appears to have no extractable content: {file_path.name}. "
            "This may be a scanned document requiring OCR."
        )

    # Build metadata
    metadata: Dict[str, Any] = {
        'page_count': len(text_by_page),
        'filename': file_path.name
    }

    return {
        'text': text_by_page,
        'tables': tables,
        'metadata': metadata
    }


def fix_financial_table_structure(table: pd.DataFrame) -> pd.DataFrame:
    """Fix financial tables where first column should be the index"""
    if table.empty or len(table.columns) < 2:
        return table

    # Check if first column contains metric labels (mostly text)
    first_col = table.iloc[:, 0].astype(str)

    # Count non-numeric cells in first column
    text_cells = 0
    for cell in first_col:
        clean_cell = cell.replace('%', '').replace('$', '').replace(',', '').replace('-', '').strip()
        if not clean_cell.replace('.', '').isdigit():
            text_cells += 1

    # If >60% of first column is text, treat it as row labels
    if text_cells / max(len(first_col), 1) > 0.6:
        # Set first column as index
        table = table.set_index(table.columns[0])

        # If index name is too long (>50 chars), it's metadata, not a real column name
        if len(str(table.index.name)) > 50:
            table.index.name = 'Metric'

        # Reset index to make it a regular column again but with clean name
        table = table.reset_index()

    return table


def _clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted PDF text"""
    import re

    if not text:
        return ""

    # Fix spaced-out words (F I N A N C I A L → FINANCIAL)
    text = re.sub(r'(\b[A-Z])\s+(?=[A-Z]\s+[A-Z])', r'\1', text)
    text = re.sub(r'([A-Z])\s+([A-Z])\s+([A-Z])', r'\1\2\3', text)

    # Split into lines for filtering
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty or very short lines
        if not line or len(line) < 15:
            continue

        # Skip page numbers (just digits or "Page X")
        if re.match(r'^(Page\s*)?\d+\s*$', line, re.IGNORECASE):
            continue

        # Skip lines that are mostly numbers/symbols (table fragments)
        alpha_count = sum(c.isalpha() for c in line)
        if len(line) > 0 and alpha_count / len(line) < 0.4:
            continue

        # Skip headers/footers (all caps, short)
        if line.isupper() and len(line.split()) < 5:
            continue

        cleaned_lines.append(line)

    # Join with spaces
    cleaned = ' '.join(cleaned_lines)

    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Fix common OCR/extraction issues
    cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)  # Remove space before punctuation

    return cleaned.strip()


def _extract_page_text(page: Any, page_num: int) -> str:
    """
    Extract text content from a single PDF page.

    Uses default extraction method (better for narrative text).

    Args:
        page: A pdfplumber page object.
        page_num: The page number (1-indexed) for logging purposes.

    Returns:
        Extracted and cleaned text as a string. Returns empty string if no text found.
    """
    text = None

    try:
        text = page.extract_text()  # Remove layout=True - better for narrative text
        if text and text.strip():
            text = _clean_extracted_text(text)
            logger.debug(f"Page {page_num}: Extracted {len(text)} chars")
            return text
        else:
            logger.debug(f"Page {page_num}: extract_text() returned {'None' if text is None else 'empty string'}")
    except Exception as e:
        logger.debug(f"Page {page_num}: Text extraction failed: {str(e)}")

    # No text extracted - log warning
    logger.warning(
        f"Page {page_num}: NO TEXT EXTRACTED. "
        f"Page dimensions: {page.width}x{page.height}, "
        f"Chars in page: {len(page.chars) if hasattr(page, 'chars') else 'N/A'}"
    )

    return ""


def _extract_page_tables(page: Any, page_num: int) -> List[pd.DataFrame]:
    """
    Extract all tables from a single PDF page.

    Args:
        page: A pdfplumber page object.
        page_num: The page number (1-indexed) for logging purposes.

    Returns:
        List of pandas DataFrames representing tables found on the page.
        Returns empty list if no tables found or extraction fails.
    """
    dataframes: List[pd.DataFrame] = []

    try:
        raw_tables = page.extract_tables()

        if not raw_tables:
            logger.debug(f"Page {page_num}: No tables found")
            return dataframes

        logger.debug(f"Page {page_num}: Found {len(raw_tables)} raw table(s)")

        for table_idx, raw_table in enumerate(raw_tables):
            df = _convert_to_dataframe(raw_table, page_num, table_idx)
            if df is not None:
                dataframes.append(df)

    except Exception as e:
        logger.warning(
            f"Page {page_num}: Failed to extract tables. Error: {str(e)}"
        )

    return dataframes


def _convert_to_dataframe(
    raw_table: List[List[Optional[str]]],
    page_num: int,
    table_idx: int
) -> Optional[pd.DataFrame]:
    """
    Convert a raw table (list of lists) to a pandas DataFrame.

    Handles common issues including empty cells, whitespace normalization,
    and header row detection.

    Args:
        raw_table: Raw table data as nested lists from pdfplumber.
        page_num: Page number for logging.
        table_idx: Table index on the page for logging.

    Returns:
        pandas DataFrame if conversion successful, None otherwise.
    """
    if not raw_table or len(raw_table) == 0:
        return None

    # Clean the table data
    cleaned_rows: List[List[str]] = []
    for row in raw_table:
        if row is None:
            continue
        cleaned_row = [
            _clean_cell(cell) for cell in row
        ]
        cleaned_rows.append(cleaned_row)

    if len(cleaned_rows) == 0:
        return None

    # Check if first row looks like headers
    if len(cleaned_rows) > 1 and _is_header_row(cleaned_rows[0]):
        headers = _make_unique_columns(cleaned_rows[0])
        data = cleaned_rows[1:]
        df = pd.DataFrame(data, columns=headers)
    else:
        df = pd.DataFrame(cleaned_rows)

    # Remove completely empty rows and columns
    df = df.replace('', pd.NA)
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df = df.fillna('')

    if df.empty:
        return None

    # Fix financial table structure
    df = fix_financial_table_structure(df)

    logger.debug(
        f"Page {page_num}, Table {table_idx}: "
        f"Converted to DataFrame with shape {df.shape}"
    )

    return df


def _clean_cell(cell: Optional[str]) -> str:
    """
    Clean a single table cell value.

    Args:
        cell: Raw cell value (may be None).

    Returns:
        Cleaned string with normalized whitespace.
    """
    if cell is None:
        return ""

    # Convert to string and strip
    text = str(cell).strip()

    # Normalize internal whitespace
    text = ' '.join(text.split())

    return text


def _is_header_row(row: List[str]) -> bool:
    """
    Determine if a row is likely a header row using heuristics.

    Headers typically contain text labels rather than numeric data.

    Args:
        row: List of cell values.

    Returns:
        True if the row appears to be a header row.
    """
    if not row:
        return False

    # Count non-empty cells
    non_empty = [cell for cell in row if cell.strip()]
    if len(non_empty) < len(row) * 0.5:
        return False

    # Count how many cells look numeric
    numeric_count = 0
    for cell in non_empty:
        if _looks_numeric(cell):
            numeric_count += 1

    # Headers should be mostly non-numeric
    return numeric_count < len(non_empty) * 0.3


def _looks_numeric(value: str) -> bool:
    """
    Check if a string value appears to be numeric.

    Handles common financial formatting like currency symbols,
    percentages, and parentheses for negatives.

    Args:
        value: String to check.

    Returns:
        True if the value appears to be numeric.
    """
    # Remove common financial formatting
    cleaned = value.replace(',', '').replace('$', '').replace('%', '')
    cleaned = cleaned.replace('(', '').replace(')', '').replace('-', '')
    cleaned = cleaned.replace('€', '').replace('£', '').strip()

    if not cleaned:
        return False

    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _make_unique_columns(headers: List[str]) -> List[str]:
    """
    Ensure all column headers are unique by appending numeric suffixes.

    Args:
        headers: List of header strings (may contain duplicates or empty strings).

    Returns:
        List of unique header strings.
    """
    seen: Dict[str, int] = {}
    unique: List[str] = []

    for header in headers:
        # Handle empty headers
        if not header.strip():
            header = "Column"

        # Make unique
        if header in seen:
            seen[header] += 1
            unique.append(f"{header}_{seen[header]}")
        else:
            seen[header] = 0
            unique.append(header)

    return unique


def _setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the module.

    Args:
        verbose: If True, set DEBUG level. Otherwise, INFO level.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# =============================================================================
# Test Block
# =============================================================================

if __name__ == "__main__":
    """
    Test the document parser with a PDF file.

    Usage:
        python document_parser.py [pdf_path]

    If no path provided, uses a default test path.
    """
    # Setup logging for testing
    _setup_logging(verbose=True)

    # Get PDF path from command line or use default
    DEFAULT_TEST_PATH = "test_document.pdf"

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = DEFAULT_TEST_PATH
        print(f"No PDF path provided. Using default: {pdf_path}")

    print("=" * 60)
    print("Document Parser Test")
    print("=" * 60)
    print(f"Input file: {pdf_path}\n")

    try:
        # Parse the document
        result = extract_document_content(pdf_path)

        # Print results
        print(f"✓ Successfully parsed PDF")
        print(f"\n--- Metadata ---")
        print(f"Filename: {result['metadata']['filename']}")
        print(f"Number of pages: {result['metadata']['page_count']}")

        print(f"\n--- Text Content ---")
        print(f"Pages with text: {len(result['text'])}")

        if result['text'] and result['text'][0]:
            first_page_text = result['text'][0]
            preview = first_page_text[:500]
            print(f"\nFirst 500 characters of page 1:")
            print("-" * 40)
            print(preview)
            if len(first_page_text) > 500:
                print(f"... [{len(first_page_text) - 500} more characters]")
            print("-" * 40)
        else:
            print("No text found on first page.")

        print(f"\n--- Tables ---")
        print(f"Number of tables found: {len(result['tables'])}")

        if result['tables']:
            print(f"\nFirst table preview (first 5 rows):")
            print("-" * 40)
            first_table = result['tables'][0]
            print(f"Shape: {first_table.shape[0]} rows × {first_table.shape[1]} columns")
            print(f"Columns: {list(first_table.columns)}")
            print()
            print(first_table.head(5).to_string())
            print("-" * 40)
        else:
            print("No tables found in document.")

        print(f"\n{'=' * 60}")
        print("Test completed successfully!")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"\nPlease provide a valid PDF file path.")
        sys.exit(1)

    except EmptyPDFError as e:
        print(f"✗ Error: {e}")
        print(f"\nThe PDF file has no pages to parse.")
        sys.exit(1)

    except CorruptedPDFError as e:
        print(f"✗ Error: {e}")
        print(f"\nThe PDF file appears to be corrupted or encrypted.")
        sys.exit(1)

    except PDFParsingError as e:
        print(f"✗ Parsing Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"✗ Unexpected Error: {type(e).__name__}: {e}")
        logger.exception("Unexpected error during PDF parsing")
        sys.exit(1)
