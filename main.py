#!/usr/bin/env python3
"""
Document Intelligence System - Main Pipeline

This is the main entry point for the Document Intelligence System, which combines
PDF parsing, question answering (both text and table-based), question routing,
and document summarization into a unified pipeline.

Architecture Overview:
----------------------
1. document_parser: Extracts text and tables from PDF documents
2. question_router: Determines if a question is better suited for text or table QA
3. text_qa: Answers questions from unstructured text using DistilBERT
4. table_qa: Answers questions from tabular data using TAPAS
5. summarizer: Generates concise summaries using Flan-T5-Large

Pipeline Flow:
--------------
User Question + PDF → Parse PDF → Route Question → Select QA Model → Generate Answer

Usage:
------
    # Command line
    python main.py <pdf_path> <question>
    python main.py document.pdf "What was the revenue?"

    # Python API
    from main import process_document_query, summarize_document_content

    result = process_document_query("document.pdf", "What was the revenue?")
    summary = summarize_document_content("document.pdf", max_length=100)
"""

import sys
import time
import logging
import argparse
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODULE IMPORTS
# =============================================================================

# Import all component modules
# Using try/except to provide helpful error messages if modules are missing
try:
    import document_parser
except ImportError as e:
    logger.error("Failed to import document_parser module")
    raise ImportError(
        "document_parser module not found. Ensure document_parser.py "
        "is in the same directory."
    ) from e

try:
    import text_qa
except ImportError as e:
    logger.error("Failed to import text_qa module")
    raise ImportError(
        "text_qa module not found. Ensure text_qa.py is in the same directory."
    ) from e

try:
    import table_qa
except ImportError as e:
    logger.error("Failed to import table_qa module")
    raise ImportError(
        "table_qa module not found. Ensure table_qa.py is in the same directory."
    ) from e

try:
    import question_router
except ImportError as e:
    logger.error("Failed to import question_router module")
    raise ImportError(
        "question_router module not found. Ensure question_router.py "
        "is in the same directory."
    ) from e

try:
    import summarizer
except ImportError as e:
    logger.error("Failed to import summarizer module")
    raise ImportError(
        "summarizer module not found. Ensure summarizer.py is in the same directory."
    ) from e


# =============================================================================
# TIMING UTILITIES
# =============================================================================

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.

    Wraps a function to track how long it takes to execute,
    logging the duration and returning it in the result if applicable.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function with timing.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            # If result is a dict, add timing info
            if isinstance(result, dict):
                result['processing_time'] = round(elapsed, 3)

            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}")
            raise

    return wrapper


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.description} completed in {self.elapsed:.3f}s")


# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

@timing_decorator
def process_document_query(
    pdf_path: str,
    question: str,
    confidence_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Process a question against a PDF document using the appropriate QA model.

    This is the main pipeline function that:
    1. Parses the PDF document to extract text and tables
    2. Routes the question to determine the best QA approach
    3. Applies the appropriate QA model (text or table)
    4. Returns the answer with metadata and source attribution

    Pipeline Decision Logic:
    ------------------------
    - Questions routed to 'table':
      * First attempts to answer using extracted tables
      * If table answer has low confidence (score < 0.7), also tries text QA
      * Returns whichever source has higher confidence
    - Questions routed to 'text':
      * Uses text QA on combined document text
      * No table fallback (text questions aren't suited for tables)

    Args:
        pdf_path: Path to the PDF document to query.
        question: The question to answer.
        confidence_threshold: Minimum confidence score to accept (default: 0.3).

    Returns:
        Dictionary containing:
            - 'question': Original question
            - 'answer': The extracted answer
            - 'score': Confidence score (0.0 to 1.0)
            - 'source_type': 'text' or 'table'
            - 'source_page': Page number where answer was found (1-indexed)
            - 'source_details': Additional source information
            - 'routing_decision': How the question was routed
            - 'success': Boolean indicating if answer was found
            - 'message': Status or error message
            - 'processing_time': Time taken (added by decorator)

    Raises:
        FileNotFoundError: If PDF file doesn't exist.
        ValueError: If question is empty.
    """
    # =================================
    # Input Validation
    # =================================

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    question = question.strip()

    # =================================
    # Step 1: Parse PDF Document
    # =================================

    logger.info(f"Parsing PDF: {pdf_path}")

    with Timer("PDF parsing"):
        try:
            document_content = document_parser.extract_document_content(pdf_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        except Exception as e:
            logger.error(f"PDF parsing failed: {str(e)}")
            return {
                'question': question,
                'answer': '',
                'score': 0.0,
                'source_type': 'unknown',
                'source_page': -1,
                'source_details': {},
                'routing_decision': 'unknown',
                'success': False,
                'message': f'PDF parsing failed: {str(e)}'
            }

    # Extract content from parsed document
    text_pages = document_content.get('text', [])
    tables = document_content.get('tables', [])
    metadata = document_content.get('metadata', {})

    logger.info(f"Extracted {len(text_pages)} pages and {len(tables)} tables")

    # Check if document has content
    if not text_pages:
        return {
            'question': question,
            'answer': '',
            'score': 0.0,
            'source_type': 'unknown',
            'source_page': -1,
            'source_details': {'total_pages': len(metadata.get('page_count', 0)), 'total_tables': len(tables)},
            'routing_decision': 'unknown',
            'success': False,
            'message': 'No text content found in PDF'
        }

    # =================================
    # Step 2: Route Question
    # =================================

    logger.info(f"Routing question: {question}")

    with Timer("Question routing"):
        route = question_router.route_question(question)
        routing_details = question_router.route_question_with_details(question)

    logger.info(f"Question routed to: {route}")

    # =================================
    # Step 3: Apply Appropriate QA Model
    # =================================

    if route == 'table' and tables:
        # Try table QA first
        table_answer = _answer_from_tables(
            question, tables, confidence_threshold
        )

        # If low confidence from tables, try text QA as fallback
        if not table_answer['success'] or table_answer['score'] < 0.7:
            logger.info("Low confidence from tables, trying text QA as fallback")
            text_answer = _answer_from_text(
                question, text_pages, confidence_threshold
            )

            # Return the best answer based on confidence score
            if text_answer['success'] and text_answer['score'] > table_answer.get('score', 0):
                text_answer['question'] = question
                text_answer['routing_decision'] = route
                text_answer['source_details']['routing_reason'] = routing_details.get('reason', '')
                text_answer['source_details']['fallback'] = True
                text_answer['message'] = text_answer.get('message', '') + ' (fallback from table QA)'
                return text_answer

        # Return table answer (either high confidence, or text wasn't better)
        table_answer['question'] = question
        table_answer['routing_decision'] = route
        table_answer['source_details']['routing_reason'] = routing_details.get('reason', '')
        return table_answer

    # Use text QA for text-routed questions
    result = _answer_from_text(
        question, text_pages, confidence_threshold
    )

    result['question'] = question
    result['routing_decision'] = route
    result['source_details']['routing_reason'] = routing_details.get('reason', '')

    return result


def _answer_from_tables(
    question: str,
    tables: List[Any],
    confidence_threshold: float
) -> Dict[str, Any]:
    """
    Attempt to answer a question using extracted tables.

    Tries each table and returns the answer with the highest confidence.

    Args:
        question: The question to answer.
        tables: List of pandas DataFrames from document_parser.
        confidence_threshold: Minimum confidence to accept.

    Returns:
        Answer dictionary with source information.
    """
    import pandas as pd

    best_result = None
    best_score = -1.0
    best_table_idx = -1

    for i, df in enumerate(tables):
        try:
            # Tables from document_parser are already DataFrames
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Table {i} is not a DataFrame, skipping")
                continue

            if df.empty:
                continue

            # Query this table
            result = table_qa.answer_table_question(
                question=question,
                table=df,
                table_name=f"table_{i}",
                confidence_threshold=confidence_threshold
            )

            if result['score'] > best_score:
                best_score = result['score']
                best_result = result
                best_table_idx = i

        except Exception as e:
            logger.warning(f"Failed to query table {i}: {str(e)}")
            continue

    if best_result is None or not best_result.get('success', False):
        return {
            'answer': '',
            'score': 0.0,
            'source_type': 'table',
            'source_page': -1,
            'source_details': {'tables_tried': len(tables)},
            'success': False,
            'message': 'No confident answer found in any table'
        }

    return {
        'answer': str(best_result['answer']),
        'score': best_result['score'],
        'source_type': 'table',
        'source_page': -1,
        'source_details': {
            'table_index': best_table_idx,
            'aggregation': best_result.get('aggregation', 'NONE'),
            'cells': best_result.get('cells', []),
            'tables_tried': len(tables)
        },
        'success': True,
        'message': best_result.get('message', 'Answer found in table')
    }


def _answer_from_text(
    question: str,
    text_pages: List[str],
    confidence_threshold: float
) -> Dict[str, Any]:
    """
    Answer a question using extracted text from document pages.

    Combines all page text and uses text QA model.

    Args:
        question: The question to answer.
        text_pages: List of text strings from document_parser (one per page).
        confidence_threshold: Minimum confidence to accept.

    Returns:
        Answer dictionary with source information.
    """
    # Combine all page text with page markers for source attribution
    combined_text = ""
    page_boundaries = []  # Track where each page starts in combined text

    for page_num, page_text in enumerate(text_pages, start=1):
        if page_text.strip():
            start_pos = len(combined_text)
            combined_text += page_text + "\n\n"
            page_boundaries.append({
                'page': page_num,
                'start': start_pos,
                'end': len(combined_text)
            })

    if not combined_text.strip():
        return {
            'answer': '',
            'score': 0.0,
            'source_type': 'text',
            'source_page': -1,
            'source_details': {'total_pages': len(text_pages)},
            'success': False,
            'message': 'No text content found in document'
        }

    try:
        # Query the combined text
        result = text_qa.answer_text_question(
            question=question,
            context=combined_text,
            confidence_threshold=confidence_threshold
        )

        # Determine which page the answer came from
        source_page = -1
        if result.get('start', -1) >= 0:
            answer_pos = result['start']
            for boundary in page_boundaries:
                if boundary['start'] <= answer_pos < boundary['end']:
                    source_page = boundary['page']
                    break

        return {
            'answer': result.get('answer', ''),
            'score': result.get('score', 0.0),
            'source_type': 'text',
            'source_page': source_page,
            'source_details': {
                'total_pages': len(text_pages),
                'answer_position': result.get('start', -1),
                'context_length': len(combined_text)
            },
            'success': result.get('success', False),
            'message': result.get('message', '')
        }

    except Exception as e:
        logger.error(f"Text QA failed: {str(e)}")
        return {
            'answer': '',
            'score': 0.0,
            'source_type': 'text',
            'source_page': -1,
            'source_details': {'error': str(e)},
            'success': False,
            'message': f'Text QA failed: {str(e)}'
        }


@timing_decorator
def summarize_document_content(
    pdf_path: str,
    max_length: int = 150,
    min_length: int = 50
) -> Dict[str, Any]:
    """
    Generate a summary of a PDF document.

    Parses the PDF, combines all text content, and generates a summary
    using the BART summarization model.

    Args:
        pdf_path: Path to the PDF document.
        max_length: Maximum summary length in words (default: 150).
        min_length: Minimum summary length in words (default: 50).

    Returns:
        Dictionary containing:
            - 'summary': The generated summary text
            - 'original_length': Word count of original document
            - 'summary_length': Word count of summary
            - 'compression_ratio': Ratio of summary to original length
            - 'total_pages': Number of pages in document
            - 'success': Boolean indicating success
            - 'message': Status message
            - 'processing_time': Time taken (added by decorator)
    """
    # =================================
    # Parse PDF
    # =================================

    logger.info(f"Parsing PDF for summarization: {pdf_path}")

    try:
        document_content = document_parser.extract_document_content(pdf_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        return {
            'summary': '',
            'original_length': 0,
            'summary_length': 0,
            'compression_ratio': 0.0,
            'total_pages': 0,
            'success': False,
            'message': f'PDF parsing failed: {str(e)}'
        }

    text_pages = document_content.get('text', [])

    # =================================
    # Combine Text
    # =================================

    combined_text = "\n\n".join(
        page_text for page_text in text_pages
        if page_text.strip()
    )

    if not combined_text.strip():
        return {
            'summary': '',
            'original_length': 0,
            'summary_length': 0,
            'compression_ratio': 0.0,
            'total_pages': len(text_pages),
            'success': False,
            'message': 'No text content found in PDF'
        }

    original_words = len(combined_text.split())

    # =================================
    # Generate Summary
    # =================================

    logger.info(f"Generating summary (target: {min_length}-{max_length} words)")

    try:
        summary_text = summarizer.summarize_document(
            text=combined_text,
            max_length=max_length,
            min_length=min_length
        )

        summary_words = len(summary_text.split())
        compression = summary_words / original_words if original_words > 0 else 0

        return {
            'summary': summary_text,
            'original_length': original_words,
            'summary_length': summary_words,
            'compression_ratio': round(compression, 3),
            'total_pages': len(text_pages),
            'success': True,
            'message': 'Summary generated successfully'
        }

    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        return {
            'summary': '',
            'original_length': original_words,
            'summary_length': 0,
            'compression_ratio': 0.0,
            'total_pages': len(text_pages),
            'success': False,
            'message': f'Summarization failed: {str(e)}'
        }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_multiple_questions(
    pdf_path: str,
    questions: List[str],
    confidence_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Process multiple questions against a single PDF document.

    More efficient than calling process_document_query multiple times
    because the PDF is only parsed once.

    Args:
        pdf_path: Path to the PDF document.
        questions: List of questions to answer.
        confidence_threshold: Minimum confidence score.

    Returns:
        List of answer dictionaries, one per question.
    """
    results = []

    # Parse PDF once
    logger.info(f"Parsing PDF for batch processing: {pdf_path}")
    try:
        document_content = document_parser.extract_document_content(pdf_path)
    except Exception as e:
        # Return error results for all questions
        return [{
            'question': q,
            'answer': '',
            'score': 0.0,
            'source_type': 'unknown',
            'source_page': -1,
            'success': False,
            'message': f'PDF parsing failed: {str(e)}'
        } for q in questions]

    text_pages = document_content.get('text', [])
    tables = document_content.get('tables', [])

    # Process each question
    for question in questions:
        start_time = time.time()

        try:
            route = question_router.route_question(question)

            if route == 'table' and tables:
                table_answer = _answer_from_tables(question, tables, confidence_threshold)

                # If low confidence from tables, try text QA as fallback
                if not table_answer['success'] or table_answer['score'] < 0.7:
                    text_answer = _answer_from_text(question, text_pages, confidence_threshold)

                    # Return the best answer based on confidence score
                    if text_answer['success'] and text_answer['score'] > table_answer.get('score', 0):
                        result = text_answer
                        result['source_details']['fallback'] = True
                    else:
                        result = table_answer
                else:
                    result = table_answer
            else:
                result = _answer_from_text(question, text_pages, confidence_threshold)

            result['question'] = question
            result['routing_decision'] = route
            result['processing_time'] = round(time.time() - start_time, 3)

        except Exception as e:
            result = {
                'question': question,
                'answer': '',
                'score': 0.0,
                'source_type': 'unknown',
                'source_page': -1,
                'success': False,
                'message': f'Error: {str(e)}',
                'processing_time': round(time.time() - start_time, 3)
            }

        results.append(result)

    return results


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_result(result: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format a result dictionary for human-readable output.

    Args:
        result: The result dictionary from process_document_query.
        verbose: If True, include additional details.

    Returns:
        Formatted string representation.
    """
    lines = []

    lines.append(f"Question: {result.get('question', 'N/A')}")
    lines.append("-" * 50)

    if result.get('success', False):
        lines.append(f"Answer: {result.get('answer', 'N/A')}")
        lines.append(f"Confidence: {result.get('score', 0):.2%}")
        lines.append(f"Source: {result.get('source_type', 'unknown').upper()}")

        if result.get('source_page', -1) > 0:
            lines.append(f"Page: {result['source_page']}")
    else:
        lines.append(f"Answer: Could not find answer")
        lines.append(f"Reason: {result.get('message', 'Unknown error')}")

    if verbose:
        lines.append("")
        lines.append("Details:")
        lines.append(f"  Routing: {result.get('routing_decision', 'N/A')}")
        lines.append(f"  Processing time: {result.get('processing_time', 0):.3f}s")

        if 'source_details' in result:
            for key, value in result['source_details'].items():
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def format_summary_result(result: Dict[str, Any]) -> str:
    """
    Format a summary result dictionary for human-readable output.

    Args:
        result: The result dictionary from summarize_document_content.

    Returns:
        Formatted string representation.
    """
    lines = []

    lines.append("Document Summary")
    lines.append("=" * 50)

    if result.get('success', False):
        lines.append("")
        lines.append(result.get('summary', 'N/A'))
        lines.append("")
        lines.append("-" * 50)
        lines.append(f"Original: {result.get('original_length', 0)} words")
        lines.append(f"Summary: {result.get('summary_length', 0)} words")
        lines.append(f"Compression: {result.get('compression_ratio', 0):.1%}")
        lines.append(f"Pages: {result.get('total_pages', 0)}")
        lines.append(f"Time: {result.get('processing_time', 0):.3f}s")
    else:
        lines.append(f"Failed: {result.get('message', 'Unknown error')}")

    return "\n".join(lines)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Document Intelligence System - Query PDFs with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf "What was the revenue?"
  python main.py report.pdf "Why did margins decline?" --verbose
  python main.py annual_report.pdf --summarize
  python main.py document.pdf --summarize --max-length 100
        """
    )

    parser.add_argument(
        'pdf_path',
        nargs='?',
        help='Path to PDF document'
    )

    parser.add_argument(
        'question',
        nargs='?',
        help='Question to answer about the document'
    )

    parser.add_argument(
        '--summarize', '-s',
        action='store_true',
        help='Generate a summary instead of answering a question'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=150,
        help='Maximum summary length in words (default: 150)'
    )

    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimum summary length in words (default: 50)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test suite with sample data'
    )

    return parser.parse_args()


def run_test_suite():
    """
    Run comprehensive test suite demonstrating system capabilities.

    Tests both text and table question answering, as well as summarization.
    """
    print("=" * 70)
    print("Document Intelligence System - Test Suite")
    print("=" * 70)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check for sample PDF
    import os
    sample_pdfs = [
        'annualreport-2023.pdf',
        'sample.pdf',
        'test.pdf',
        'document.pdf'
    ]

    pdf_path = None
    for pdf in sample_pdfs:
        if os.path.exists(pdf):
            pdf_path = pdf
            break

    if pdf_path is None:
        print("No sample PDF found for testing.")
        print("Please provide a PDF file or place one of these in the current directory:")
        for pdf in sample_pdfs:
            print(f"  - {pdf}")
        print()
        print("Running module import tests only...")
        print()

        # Test module imports
        print("Module Import Tests:")
        print("-" * 50)
        modules = [
            ('document_parser', document_parser),
            ('text_qa', text_qa),
            ('table_qa', table_qa),
            ('question_router', question_router),
            ('summarizer', summarizer)
        ]

        for name, module in modules:
            print(f"  {name}: OK")

        print()
        print("Question Router Tests:")
        print("-" * 50)

        test_questions = [
            ("What was the revenue?", "table"),
            ("Why did margins decline?", "text"),
            ("What is the total profit?", "table"),
            ("Explain the company strategy", "text"),
        ]

        for question, expected in test_questions:
            result = question_router.route_question(question)
            status = "✓" if result == expected else "✗"
            print(f"  {status} '{question}' → {result}")

        print()
        print("=" * 70)
        print("Basic tests completed. Provide a PDF for full testing.")
        print("=" * 70)
        return

    # Full test suite with PDF
    print(f"Using PDF: {pdf_path}")
    print()

    # Test 1: Document Parsing
    print("-" * 70)
    print("Test 1: Document Parsing")
    print("-" * 70)

    try:
        with Timer("Document parsing"):
            content = document_parser.extract_document_content(pdf_path)

        text_pages = content.get('text', [])
        tables = content.get('tables', [])
        print(f"  Pages extracted: {len(text_pages)}")
        print(f"  Tables extracted: {len(tables)}")
        print(f"  Status: SUCCESS")
    except Exception as e:
        print(f"  Status: FAILED - {str(e)}")

    print()

    # Test 2: Table-oriented Questions
    print("-" * 70)
    print("Test 2: Table-Oriented Questions")
    print("-" * 70)

    table_questions = [
        "What was the revenue?",
        "What is the total profit?",
        "Which quarter had the highest margin?",
    ]

    for question in table_questions:
        print(f"\n  Q: {question}")
        try:
            result = process_document_query(pdf_path, question)
            print(f"  A: {result.get('answer', 'No answer')}")
            print(f"  Score: {result.get('score', 0):.2%}")
            print(f"  Source: {result.get('source_type', 'unknown')}")
            print(f"  Time: {result.get('processing_time', 0):.3f}s")
        except Exception as e:
            print(f"  Error: {str(e)}")

    print()

    # Test 3: Text-oriented Questions
    print("-" * 70)
    print("Test 3: Text-Oriented Questions")
    print("-" * 70)

    text_questions = [
        "What is the company's strategy?",
        "What are the risk factors?",
        "What is the outlook for next year?",
    ]

    for question in text_questions:
        print(f"\n  Q: {question}")
        try:
            result = process_document_query(pdf_path, question)
            print(f"  A: {result.get('answer', 'No answer')[:100]}...")
            print(f"  Score: {result.get('score', 0):.2%}")
            print(f"  Source: {result.get('source_type', 'unknown')}")
            print(f"  Time: {result.get('processing_time', 0):.3f}s")
        except Exception as e:
            print(f"  Error: {str(e)}")

    print()

    # Test 4: Summarization
    print("-" * 70)
    print("Test 4: Document Summarization")
    print("-" * 70)

    summary_lengths = [50, 100, 150]

    for max_len in summary_lengths:
        print(f"\n  Target length: {max_len} words")
        try:
            result = summarize_document_content(
                pdf_path,
                max_length=max_len,
                min_length=max_len // 2
            )
            if result['success']:
                print(f"  Generated: {result['summary_length']} words")
                print(f"  Compression: {result['compression_ratio']:.1%}")
                print(f"  Time: {result['processing_time']:.3f}s")
                print(f"  Summary: {result['summary'][:150]}...")
            else:
                print(f"  Failed: {result['message']}")
        except Exception as e:
            print(f"  Error: {str(e)}")

    print()
    print("=" * 70)
    print("Test Suite Completed")
    print("=" * 70)


def main():
    """Main entry point for command line interface."""
    args = parse_arguments()

    # Run test suite if requested
    if args.test:
        run_test_suite()
        return

    # Check for required arguments
    if not args.pdf_path:
        print("Error: PDF path is required")
        print("Usage: python main.py <pdf_path> <question>")
        print("       python main.py <pdf_path> --summarize")
        print("       python main.py --test")
        sys.exit(1)

    # Summarization mode
    if args.summarize:
        print(f"Summarizing: {args.pdf_path}")
        print()

        try:
            result = summarize_document_content(
                args.pdf_path,
                max_length=args.max_length,
                min_length=args.min_length
            )
            print(format_summary_result(result))

        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        return

    # Question answering mode
    if not args.question:
        print("Error: Question is required (or use --summarize)")
        print("Usage: python main.py <pdf_path> <question>")
        sys.exit(1)

    print(f"Document: {args.pdf_path}")
    print()

    try:
        result = process_document_query(args.pdf_path, args.question)
        print(format_result(result, verbose=args.verbose))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
