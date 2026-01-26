"""
Text Question Answering Module

This module provides functionality to answer questions from text content
using transformer-based models (DistilBERT fine-tuned on SQuAD).
"""

from typing import Dict, List, Optional,Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to cache the QA pipeline
_qa_pipeline = None


def get_qa_pipeline():
    """
    Get or initialize the QA pipeline with lazy loading.

    Returns:
        The transformers QA pipeline instance.

    Raises:
        RuntimeError: If the model fails to load.
    """
    global _qa_pipeline

    if _qa_pipeline is None:
        try:
            from transformers import pipeline
            print("Loading model...")
            logger.info("Loading QA model: distilbert-base-cased-distilled-squad")
            _qa_pipeline = pipeline(
                "question-answering",
                model="distilbert/distilbert-base-cased-distilled-squad",
                device=-1  # Force CPU
            )
            print("Model loaded!")
            logger.info("QA model loaded successfully")
        except ImportError as e:
            raise RuntimeError(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load QA model: {str(e)}") from e

    return _qa_pipeline


def _split_into_chunks(
    context: str,
    max_length: int,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Split context into overlapping chunks for processing long texts.

    Args:
        context: The full context text to split.
        max_length: Maximum character length per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of dictionaries with 'text', 'start', and 'end' keys.
    """
    chunks = []
    start = 0
    context_length = len(context)

    while start < context_length:
        end = min(start + max_length, context_length)

        # Try to break at a sentence or word boundary
        if end < context_length:
            # Look for sentence boundary (period followed by space)
            boundary = context.rfind('. ', start + max_length - 200, end)
            if boundary == -1:
                # Fall back to word boundary (space)
                boundary = context.rfind(' ', start + max_length - 200, end)
            if boundary != -1:
                end = boundary + 1

        chunk_text = context[start:end]
        chunks.append({
            'text': chunk_text,
            'start': start,
            'end': end
        })

        # Move start position, accounting for overlap
        if end >= context_length:
            break
        start = end - overlap

    return chunks


def answer_text_question(
    question: str,
    context: str,
    max_context_length: int = 4000,
    confidence_threshold: float = 0.3
) -> Dict:
    """
    Answer a question based on the provided text context.

    Uses a DistilBERT model fine-tuned on SQuAD for extractive question answering.
    For contexts exceeding max_context_length, implements a sliding window approach
    and returns the answer with the highest confidence score.

    Args:
        question: The question to answer.
        context: The text context to search for the answer.
        max_context_length: Maximum context length per chunk (default: 4000).
        confidence_threshold: Minimum confidence score to accept an answer (default: 0.3).

    Returns:
        Dictionary containing:
            - 'answer': The extracted answer string (empty if no confident answer found)
            - 'score': Confidence score (float, 0.0 if no answer)
            - 'start': Start position in original context (-1 if no answer)
            - 'end': End position in original context (-1 if no answer)
            - 'context_used': The chunk of context where answer was found (empty if no answer)
            - 'success': Boolean indicating if a valid answer was found
            - 'message': Status message (useful for errors or low confidence)

    Raises:
        ValueError: If question or context is empty.
        RuntimeError: If model loading fails.
    """
    # Input validation
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not context or not context.strip():
        raise ValueError("Context cannot be empty")

    question = question.strip()
    context = context.strip()

    # Get the QA pipeline (may raise RuntimeError)
    qa_pipeline = get_qa_pipeline()

    # Check if we need to use sliding window approach
    if len(context) <= max_context_length:
        chunks = [{'text': context, 'start': 0, 'end': len(context)}]
    else:
        logger.info(f"Context length ({len(context)}) exceeds max ({max_context_length}), using sliding window")
        chunks = _split_into_chunks(context, max_context_length)
        logger.info(f"Split context into {len(chunks)} chunks")

    # Query each chunk and track the best answer
    best_result = None
    best_score = -1.0
    best_chunk = None

    for i, chunk in enumerate(chunks):
        try:
            result = qa_pipeline(question=question, context=chunk['text'])

            if result['score'] > best_score:
                best_score = result['score']
                best_result = result
                best_chunk = chunk

        except Exception as e:
            logger.warning(f"Error processing chunk {i}: {str(e)}")
            continue

    # Check if we found any answer
    if best_result is None:
        return {
            'answer': '',
            'score': 0.0,
            'start': -1,
            'end': -1,
            'context_used': '',
            'success': False,
            'message': 'Model could not process the context'
        }

    # Check confidence threshold
    if best_score < confidence_threshold:
        return {
            'answer': best_result['answer'],
            'score': best_score,
            'start': best_chunk['start'] + best_result['start'],
            'end': best_chunk['start'] + best_result['end'],
            'context_used': best_chunk['text'],
            'success': False,
            'message': f'Answer confidence ({best_score:.3f}) below threshold ({confidence_threshold})'
        }

    # Calculate positions in original context
    original_start = best_chunk['start'] + best_result['start']
    original_end = best_chunk['start'] + best_result['end']

    return {
        'answer': best_result['answer'],
        'score': best_score,
        'start': original_start,
        'end': original_end,
        'context_used': best_chunk['text'],
        'success': True,
        'message': 'Answer found successfully'
    }


def batch_answer_questions(
    questions: List[str],
    context: str,
    max_context_length: int = 4000,
    confidence_threshold: float = 0.3
) -> List[Dict]:
    """
    Answer multiple questions from the same context.

    Args:
        questions: List of questions to answer.
        context: The text context to search for answers.
        max_context_length: Maximum context length per chunk.
        confidence_threshold: Minimum confidence score to accept an answer.

    Returns:
        List of answer dictionaries, one for each question.
    """
    results = []
    for question in questions:
        try:
            result = answer_text_question(
                question=question,
                context=context,
                max_context_length=max_context_length,
                confidence_threshold=confidence_threshold
            )
            result['question'] = question
            results.append(result)
        except Exception as e:
            results.append({
                'question': question,
                'answer': '',
                'score': 0.0,
                'start': -1,
                'end': -1,
                'context_used': '',
                'success': False,
                'message': f'Error: {str(e)}'
            })

    return results


if __name__ == "__main__":
    # Sample financial text for testing
    sample_context = """
    Company XYZ Financial Report - Q4 2023

    Revenue Performance:
    Total revenue for the quarter was $4.2 billion, representing a 12% increase
    year-over-year. The growth was primarily driven by strong performance in our
    cloud services division, which contributed $1.8 billion to the total revenue.

    Margin Analysis:
    Operating margins declined from 28% to 24% compared to the same period last year.
    The margin compression was primarily due to increased investments in research and
    development, higher employee compensation costs following competitive market pressures,
    and one-time restructuring charges of $150 million related to our European operations.

    Future Outlook:
    Management expects revenue growth to moderate to 8-10% in the coming fiscal year
    as the company focuses on profitability improvements. The outlook remains positive
    with particular strength expected in enterprise software and artificial intelligence
    products. The company plans to expand into three new international markets and
    expects to achieve operating margin recovery to 26% by year-end through cost
    optimization initiatives and improved operational efficiency.

    Key Risks:
    The company faces headwinds from currency fluctuations, regulatory changes in
    key markets, and ongoing supply chain constraints affecting hardware deliveries.
    """

    # Test questions
    test_questions = [
        "What was the revenue?",
        "Why did margins decline?",
        "What is the outlook?"
    ]

    print("=" * 70)
    print("Text Question Answering - Test Suite")
    print("=" * 70)
    print()

    print("Loading model (this may take a moment on first run)...")
    print()

    # Run tests
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 50)

        try:
            result = answer_text_question(
                question=question,
                context=sample_context,
                confidence_threshold=0.3
            )

            print(f"Answer: {result['answer']}")
            print(f"Confidence Score: {result['score']:.4f}")
            print(f"Position: {result['start']} - {result['end']}")
            print(f"Status: {'Success' if result['success'] else 'Low confidence'}")
            print(f"Message: {result['message']}")

        except Exception as e:
            print(f"Error: {str(e)}")

        print()

    # Test edge cases
    print("=" * 70)
    print("Edge Case Tests")
    print("=" * 70)
    print()

    # Test empty question
    print("Test: Empty question")
    try:
        answer_text_question("", sample_context)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    print()

    # Test empty context
    print("Test: Empty context")
    try:
        answer_text_question("What is the revenue?", "")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    print()

    # Test question with likely no answer in context
    print("Test: Question with no relevant answer in context")
    result = answer_text_question(
        "What is the CEO's name?",
        sample_context,
        confidence_threshold=0.3
    )
    print(f"Answer: '{result['answer']}'")
    print(f"Confidence: {result['score']:.4f}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
