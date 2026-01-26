"""
Table Question Answering Module

This module provides functionality to answer questions from tabular data
using Google's TAPAS (Table Parser) model fine-tuned on WikiTableQuestions.

TAPAS Overview:
---------------
TAPAS (Table Parser) is a BERT-based model designed specifically for table
understanding tasks. Unlike traditional QA models that work on plain text,
TAPAS can understand the structure of tables including:
- Row and column relationships
- Numerical values and their aggregations
- Cell positions and their semantic meaning

The model uses special embeddings to encode:
1. Token embeddings (standard BERT)
2. Position embeddings (row/column indices)
3. Segment embeddings (to distinguish question from table)
4. Column/Row rank embeddings (relative position in table)
"""

from typing import Dict, List, Optional, Union, Any
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to cache the model and tokenizer
_tapas_model = None
_tapas_tokenizer = None


def get_tapas_model_and_tokenizer():
    """
    Get or initialize the TAPAS model and tokenizer with lazy loading.

    TAPAS requires both a specialized tokenizer and model:
    - TapasTokenizer: Handles table-specific tokenization, converting
      DataFrame cells into tokens while preserving table structure
    - TapasForQuestionAnswering: The actual model that predicts which
      cells contain the answer and what aggregation to apply

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        RuntimeError: If the model fails to load.
    """
    global _tapas_model, _tapas_tokenizer

    if _tapas_model is None or _tapas_tokenizer is None:
        try:
            from transformers import TapasTokenizer, TapasForQuestionAnswering
            import torch

            logger.info("Loading TAPAS model: google/tapas-base-finetuned-wtq")

            # Load the tokenizer
            # The tokenizer converts tables and questions into the format TAPAS expects
            _tapas_tokenizer = TapasTokenizer.from_pretrained(
                "google/tapas-base-finetuned-wtq"
            )

            # Load the model
            # This model is fine-tuned on WikiTableQuestions (WTQ) dataset
            # WTQ contains complex questions requiring aggregation operations
            _tapas_model = TapasForQuestionAnswering.from_pretrained(
                "google/tapas-base-finetuned-wtq"
            )

            # Set model to evaluation mode (disables dropout, etc.)
            _tapas_model.eval()

            logger.info("TAPAS model loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                "transformers library not installed. "
                "Install with: pip install transformers torch pandas"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load TAPAS model: {str(e)}") from e

    return _tapas_model, _tapas_tokenizer


def preprocess_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame for TAPAS model consumption.

    TAPAS-Specific Preprocessing Requirements:
    ------------------------------------------
    1. Column names must be strings (TAPAS tokenizes them)
    2. All cell values must be strings (TAPAS treats everything as text)
    3. Column names should be clean (no special characters that confuse tokenization)
    4. Empty cells should be handled (TAPAS can struggle with missing values)
    5. Table should not be too large (TAPAS has token limits ~512 tokens)

    The preprocessing steps:
    1. Convert column names to lowercase and strip whitespace
       - Ensures consistency in tokenization
       - Prevents case-sensitivity issues in questions

    2. Convert all values to strings
       - TAPAS internally treats all cell values as strings
       - Numbers are parsed from string representations
       - This ensures consistent handling across data types

    3. Replace NaN/None with empty strings
       - TAPAS handles empty strings better than null values
       - Prevents tokenization errors

    4. Strip whitespace from all cells
       - Ensures clean tokenization
       - Prevents matching issues

    Args:
        table: Input pandas DataFrame

    Returns:
        Preprocessed DataFrame ready for TAPAS

    Raises:
        ValueError: If table is empty or invalid
    """
    if table is None:
        raise ValueError("Table cannot be None")

    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(table).__name__}")

    if table.empty:
        raise ValueError("Table cannot be empty")

    # Create a copy to avoid modifying the original
    processed = table.copy()

    # Step 1: Normalize column names
    # TAPAS tokenizes column names, so they should be clean strings
    processed.columns = [
        str(col).lower().strip() for col in processed.columns
    ]

    # Step 2: Convert all values to strings
    # TAPAS expects string values; it parses numbers internally
    # This is crucial because TAPAS uses BERT tokenization on cell contents
    for col in processed.columns:
        processed[col] = processed[col].apply(
            lambda x: "" if pd.isna(x) else str(x).strip()
        )

    # Step 3: Reset index to ensure clean integer indexing
    # TAPAS uses row indices for position embeddings
    processed = processed.reset_index(drop=True)

    return processed


def _get_aggregation_label(agg_index: int) -> str:
    """
    Convert TAPAS aggregation index to human-readable label.

    TAPAS Aggregation Operations:
    -----------------------------
    TAPAS can predict not just which cells contain the answer, but also
    what operation to perform on those cells. This is crucial for questions
    like "What is the total revenue?" or "What is the average margin?"

    Aggregation indices:
    0 = NONE: Direct cell selection (no aggregation)
    1 = SUM: Add selected cell values
    2 = AVERAGE: Calculate mean of selected cells
    3 = COUNT: Count the number of selected cells

    Args:
        agg_index: The aggregation index from model output

    Returns:
        Human-readable aggregation label
    """
    aggregations = {
        0: "NONE",
        1: "SUM",
        2: "AVERAGE",
        3: "COUNT"
    }
    return aggregations.get(agg_index, "UNKNOWN")


def _extract_answer_from_coordinates(
    table: pd.DataFrame,
    coordinates: List[tuple],
    aggregation: str
) -> Union[str, float, List[str]]:
    """
    Extract answer from table using cell coordinates and aggregation.

    This function takes the predicted cell coordinates from TAPAS and
    extracts the actual answer, applying any necessary aggregation.

    Args:
        table: The preprocessed DataFrame
        coordinates: List of (row, col) tuples indicating answer cells
        aggregation: The aggregation operation to apply

    Returns:
        The extracted answer (string, number, or list)
    """
    if not coordinates:
        return ""

    # Extract values from the specified cells
    values = []
    for row_idx, col_idx in coordinates:
        if row_idx < len(table) and col_idx < len(table.columns):
            cell_value = table.iloc[row_idx, col_idx]
            values.append(cell_value)

    if not values:
        return ""

    # Apply aggregation if needed
    if aggregation == "NONE":
        # Return the value(s) directly
        if len(values) == 1:
            return values[0]
        return values

    elif aggregation in ("SUM", "AVERAGE"):
        # Try to convert to numbers and aggregate
        numeric_values = []
        for v in values:
            try:
                # Remove common formatting (commas, currency symbols, %)
                clean_v = str(v).replace(",", "").replace("$", "").replace("%", "").strip()
                numeric_values.append(float(clean_v))
            except (ValueError, TypeError):
                continue

        if not numeric_values:
            return ", ".join(values)

        if aggregation == "SUM":
            return sum(numeric_values)
        else:  # AVERAGE
            return sum(numeric_values) / len(numeric_values)

    elif aggregation == "COUNT":
        return len(values)

    return ", ".join(str(v) for v in values)


def answer_table_question(
    question: str,
    table: pd.DataFrame,
    table_name: str = "table_0",
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Answer a question based on the provided table data.

    Uses Google's TAPAS model fine-tuned on WikiTableQuestions to perform
    extractive question answering over tabular data. TAPAS can handle
    questions requiring:
    - Direct cell lookup ("What was Q2 revenue?")
    - Aggregation ("What is total revenue?")
    - Comparison ("Which quarter had highest margin?")

    How TAPAS Works:
    ----------------
    1. The question and table are tokenized together with special separators
    2. Position embeddings encode the row/column structure
    3. The model predicts:
       a) Which cells are part of the answer (cell selection)
       b) What aggregation to apply (sum, average, count, or none)
    4. The answer is extracted by combining selected cells with aggregation

    Args:
        question: The question to answer about the table.
        table: A pandas DataFrame containing the data.
        table_name: Optional name/identifier for the table (default: "table_0").
        confidence_threshold: Minimum confidence score to accept answer (default: 0.5).

    Returns:
        Dictionary containing:
            - 'answer': Extracted answer (string, number, or list)
            - 'score': Confidence score (float)
            - 'cells': List of cell coordinates [(row, col), ...] that contain the answer
            - 'aggregation': The aggregation operation applied (NONE, SUM, AVERAGE, COUNT)
            - 'table_used': Name/identifier of the table queried
            - 'success': Boolean indicating if a valid answer was found
            - 'message': Status message

    Raises:
        ValueError: If question is empty or table is invalid.
        RuntimeError: If model loading fails.
    """
    import torch

    # Input validation
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    question = question.strip()

    # Preprocess the table (may raise ValueError for invalid tables)
    processed_table = preprocess_table(table)

    # Get model and tokenizer
    model, tokenizer = get_tapas_model_and_tokenizer()

    try:
        # Tokenize the question and table together
        # TAPAS tokenizer handles the special formatting required:
        # - Adds [CLS] token at start
        # - Separates question from table with [SEP]
        # - Flattens table row by row with special row/column tokens
        # - Truncates if necessary (max 512 tokens typically)
        encoding = tokenizer(
            table=processed_table,
            queries=question,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Run inference
        # The model outputs:
        # - logits: Raw scores for each token being part of answer
        # - logits_aggregation: Scores for each aggregation operation
        with torch.no_grad():
            outputs = model(**encoding)

        # Convert logits to predictions
        # predicted_answer_coordinates: List of (row, col) tuples
        # predicted_aggregation_indices: Which aggregation to use
        predicted_answer_coordinates, predicted_aggregation_indices = (
            tokenizer.convert_logits_to_predictions(
                encoding,
                outputs.logits.detach(),
                outputs.logits_aggregation.detach()
            )
        )

        # Get the coordinates for this question (first/only question)
        coordinates = predicted_answer_coordinates[0]
        aggregation_idx = predicted_aggregation_indices[0]
        aggregation = _get_aggregation_label(aggregation_idx)

        # Calculate confidence score
        # Use softmax over logits to get probability-like scores
        logits = outputs.logits.detach()
        probs = torch.softmax(logits, dim=-1)
        # Get max probability as confidence (simplified scoring)
        max_prob = probs.max().item()

        # Also consider aggregation confidence
        agg_probs = torch.softmax(outputs.logits_aggregation.detach(), dim=-1)
        agg_confidence = agg_probs[0, aggregation_idx].item()

        # Combined confidence score
        confidence_score = (max_prob + agg_confidence) / 2

        # Check if we found any answer cells
        if not coordinates:
            return {
                'answer': '',
                'score': confidence_score,
                'cells': [],
                'aggregation': aggregation,
                'table_used': table_name,
                'success': False,
                'message': 'No answer cells found in table'
            }

        # Extract the actual answer
        answer = _extract_answer_from_coordinates(
            processed_table, coordinates, aggregation
        )

        # Check confidence threshold
        if confidence_score < confidence_threshold:
            return {
                'answer': answer,
                'score': confidence_score,
                'cells': coordinates,
                'aggregation': aggregation,
                'table_used': table_name,
                'success': False,
                'message': f'Answer confidence ({confidence_score:.3f}) below threshold ({confidence_threshold})'
            }

        return {
            'answer': answer,
            'score': confidence_score,
            'cells': coordinates,
            'aggregation': aggregation,
            'table_used': table_name,
            'success': True,
            'message': 'Answer found successfully'
        }

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {
            'answer': '',
            'score': 0.0,
            'cells': [],
            'aggregation': 'NONE',
            'table_used': table_name,
            'success': False,
            'message': f'Inference error: {str(e)}'
        }


def batch_answer_table_questions(
    questions: List[str],
    table: pd.DataFrame,
    table_name: str = "table_0",
    confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Answer multiple questions from the same table.

    Args:
        questions: List of questions to answer.
        table: The pandas DataFrame containing the data.
        table_name: Optional name/identifier for the table.
        confidence_threshold: Minimum confidence score to accept answer.

    Returns:
        List of answer dictionaries, one for each question.
    """
    results = []
    for question in questions:
        try:
            result = answer_table_question(
                question=question,
                table=table,
                table_name=table_name,
                confidence_threshold=confidence_threshold
            )
            result['question'] = question
            results.append(result)
        except Exception as e:
            results.append({
                'question': question,
                'answer': '',
                'score': 0.0,
                'cells': [],
                'aggregation': 'NONE',
                'table_used': table_name,
                'success': False,
                'message': f'Error: {str(e)}'
            })

    return results


if __name__ == "__main__":
    # Create sample financial table for testing
    # This simulates quarterly financial data that might come from an annual report
    sample_data = {
        "Quarter": ["Q1", "Q2", "Q3", "Q4"],
        "Revenue": ["$1,200", "$1,450", "$1,380", "$1,620"],
        "Operating Margin": ["24%", "28%", "26%", "30%"],
        "Net Income": ["$180", "$245", "$210", "$295"]
    }

    sample_table = pd.DataFrame(sample_data)

    print("=" * 70)
    print("Table Question Answering (TAPAS) - Test Suite")
    print("=" * 70)
    print()

    print("Sample Financial Table:")
    print("-" * 50)
    print(sample_table.to_string(index=False))
    print()

    print("Loading TAPAS model (this may take a moment on first run)...")
    print()

    # Test questions
    test_questions = [
        "What was Q2 revenue?",
        "Which quarter had highest margin?",
        "What is total revenue?"
    ]

    # Run tests
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 50)

        try:
            result = answer_table_question(
                question=question,
                table=sample_table,
                table_name="financial_summary",
                confidence_threshold=0.5
            )

            print(f"Answer: {result['answer']}")
            print(f"Confidence Score: {result['score']:.4f}")
            print(f"Aggregation: {result['aggregation']}")
            print(f"Cell Coordinates: {result['cells']}")
            print(f"Status: {'Success' if result['success'] else 'Low confidence or no answer'}")
            print(f"Message: {result['message']}")

        except Exception as e:
            print(f"Error: {str(e)}")

        print()

    # Edge case tests
    print("=" * 70)
    print("Edge Case Tests")
    print("=" * 70)
    print()

    # Test empty question
    print("Test: Empty question")
    try:
        answer_table_question("", sample_table)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    print()

    # Test empty table
    print("Test: Empty table")
    try:
        answer_table_question("What is the revenue?", pd.DataFrame())
    except ValueError as e:
        print(f"Caught expected error: {e}")
    print()

    # Test None table
    print("Test: None table")
    try:
        answer_table_question("What is the revenue?", None)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    print()

    # Test question with likely no answer in table
    print("Test: Question with no relevant answer in table")
    result = answer_table_question(
        "What is the CEO's salary?",
        sample_table,
        confidence_threshold=0.5
    )
    print(f"Answer: '{result['answer']}'")
    print(f"Confidence: {result['score']:.4f}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print()

    # Test with numeric table (no formatting)
    print("Test: Table with plain numeric values")
    numeric_table = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4"],
        "Revenue": [1200, 1450, 1380, 1620],
        "Margin": [24, 28, 26, 30]
    })
    result = answer_table_question(
        "What was Q3 revenue?",
        numeric_table,
        confidence_threshold=0.5
    )
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")
    print(f"Success: {result['success']}")
    print()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
