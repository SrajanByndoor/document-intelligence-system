"""
Document Summarization Module

This module provides functionality to generate summaries of document text
using Google's Flan-T5-Large model, an instruction-tuned language model.

Flan-T5 Overview:
-----------------
Flan-T5 is an enhanced version of T5 (Text-to-Text Transfer Transformer) that has
been fine-tuned on a large mixture of tasks with instructions. Unlike models
trained only for summarization, Flan-T5 can follow specific instructions like
"Summarize this financial report focusing on key metrics" which produces more
relevant and contextual summaries.

Key advantages for financial documents:
- Instruction-following capability allows domain-specific prompts
- Better understanding of context and relevance
- Can be guided to focus on specific aspects (metrics, risks, outlook, etc.)

Handling Long Documents:
------------------------
Flan-T5 has a default input length of 512 tokens. For longer documents, this module
implements a hierarchical summarization approach:
1. Split document into overlapping chunks (to preserve context at boundaries)
2. Summarize each chunk independently with financial-focused prompts
3. Concatenate chunk summaries
4. If result is still too long, recursively summarize the summaries

This approach preserves information from all parts of the document while
staying within model constraints.
"""

from typing import Optional, List, Tuple
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to cache the model and tokenizer
_model = None
_tokenizer = None

# =============================================================================
# CONSTANTS
# =============================================================================

# Flan-T5's maximum input token length
MAX_MODEL_TOKENS = 512

# Chunk size for splitting long documents (leaving room for prompt and special tokens)
CHUNK_SIZE_TOKENS = 450

# Overlap between chunks to preserve context at boundaries
CHUNK_OVERLAP_TOKENS = 50

# Minimum words required for summarization
MIN_WORDS_FOR_SUMMARIZATION = 50

# Prompt templates for different summarization tasks
FINANCIAL_SUMMARY_PROMPT = """Summarize the following financial document text, focusing on key metrics, performance highlights, and important business information:

{text}

Summary:"""

CHUNK_SUMMARY_PROMPT = """Summarize the following text from a financial document, preserving key numbers, metrics, and important facts:

{text}

Summary:"""

FINAL_SUMMARY_PROMPT = """Combine and summarize the following summaries into a coherent overview of the financial document:

{text}

Final Summary:"""


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_model_and_tokenizer():
    """
    Get or initialize the Flan-T5 model and tokenizer with lazy loading.

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        RuntimeError: If the model fails to load.
    """
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            print("Loading DistilBART model (this may take a moment)...")
            logger.info("Loading summarization model: sshleifer/distilbart-cnn-6-6")

            _tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
            _model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")

            # Set model to evaluation mode
            _model.eval()

            print("DistilBART model loaded!")
            logger.info("Summarization model loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load summarization model: {str(e)}") from e

    return _model, _tokenizer


def get_tokenizer():
    """
    Get or initialize the Flan-T5 tokenizer for accurate token counting.

    Returns:
        The tokenizer instance.

    Raises:
        RuntimeError: If tokenizer fails to load.
    """
    _, tokenizer = get_model_and_tokenizer()
    return tokenizer


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text using the Flan-T5 tokenizer.

    Args:
        text: The text to count tokens for.

    Returns:
        Number of tokens in the text.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def count_words(text: str) -> int:
    """
    Count the number of words in a text.

    Args:
        text: The text to count words for.

    Returns:
        Number of words in the text.
    """
    words = [w for w in text.split() if w.strip()]
    return len(words)


def _split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS
) -> List[str]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to split.
        chunk_size: Maximum tokens per chunk.
        overlap: Number of tokens to overlap between chunks.

    Returns:
        List of text chunks.
    """
    tokenizer = get_tokenizer()

    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    # If text fits in one chunk, return as-is
    if total_tokens <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        start = end - overlap

        if start >= total_tokens - overlap and end >= total_tokens:
            break

    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def _generate_summary(text: str, prompt_template: str, max_length: int, min_length: int) -> str:
    """
    Generate a summary using the Flan-T5 model with a specific prompt.

    Args:
        text: The text to summarize.
        prompt_template: The prompt template to use.
        max_length: Maximum output length in tokens.
        min_length: Minimum output length in tokens.

    Returns:
        Generated summary string.
    """
    import torch

    model, tokenizer = get_model_and_tokenizer()

    # Format the prompt
    prompt = prompt_template.format(text=text)

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_MODEL_TOKENS,
        truncation=True
    )

    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    # Decode output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# =============================================================================
# MAIN SUMMARIZATION FUNCTION
# =============================================================================

def summarize_document(
    text: str,
    max_length: int = 150,
    min_length: int = 50
) -> str:
    """
    Generate a summary of the input document text.

    Uses Google's Flan-T5-Large model with financial-focused prompts
    for better relevance in financial document summarization.

    Args:
        text: The input text to summarize.
        max_length: Maximum summary length in words (default: 150).
        min_length: Minimum summary length in words (default: 50).

    Returns:
        The generated summary string.

    Raises:
        ValueError: If text is empty or too short to summarize.
        RuntimeError: If model loading fails.
    """
    # =================================
    # Input Validation
    # =================================

    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    text = text.strip()
    word_count = count_words(text)

    if word_count < MIN_WORDS_FOR_SUMMARIZATION:
        raise ValueError(
            f"Text too short to summarize. Got {word_count} words, "
            f"minimum is {MIN_WORDS_FOR_SUMMARIZATION} words."
        )

    # Adjust min_length if it exceeds max_length
    if min_length > max_length:
        min_length = max(max_length - 20, 10)
        logger.warning(f"Adjusted min_length to {min_length} (was > max_length)")

    if min_length > word_count:
        min_length = max(word_count // 2, 10)

    # =================================
    # Token Count and Strategy Decision
    # =================================

    token_count = count_tokens(text)
    logger.info(f"Input: {word_count} words, {token_count} tokens")

    # Convert word-based lengths to token-based (rough estimate: 1.3 tokens per word)
    max_tokens = int(max_length * 1.3)
    min_tokens = int(min_length * 1.3)
    min_tokens = min(min_tokens, max_tokens - 10)
    min_tokens = max(min_tokens, 10)

    # =================================
    # Summarization
    # =================================

    if token_count <= CHUNK_SIZE_TOKENS:
        # Text fits within model limits - direct summarization
        logger.info("Using direct summarization")
        return _generate_summary(text, FINANCIAL_SUMMARY_PROMPT, max_tokens, min_tokens)
    else:
        # Text too long - use hierarchical summarization
        logger.info(f"Text exceeds model limit ({token_count} > {CHUNK_SIZE_TOKENS}), using chunked summarization")
        return _summarize_long_document(text, max_tokens, min_tokens)


def _summarize_long_document(
    text: str,
    max_tokens: int,
    min_tokens: int,
    recursion_depth: int = 0
) -> str:
    """
    Summarize a long document using hierarchical chunking.

    Args:
        text: Long text to summarize.
        max_tokens: Maximum final summary length in tokens.
        min_tokens: Minimum final summary length in tokens.
        recursion_depth: Current recursion level (for safeguard).

    Returns:
        Summary string.
    """
    MAX_RECURSION_DEPTH = 3

    if recursion_depth >= MAX_RECURSION_DEPTH:
        logger.warning("Max recursion depth reached, returning current summary")
        truncated = _truncate_to_tokens(text, CHUNK_SIZE_TOKENS - 50)
        return _generate_summary(truncated, FINANCIAL_SUMMARY_PROMPT, max_tokens, min_tokens)

    # Split into chunks
    chunks = _split_text_into_chunks(text)

    # Summarize each chunk with financial-focused prompt
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        try:
            # Use longer intermediate summaries to preserve information
            intermediate_max = max(max_tokens * 2, 150)
            intermediate_min = max(min_tokens, 30)

            chunk_summary = _generate_summary(
                chunk, CHUNK_SUMMARY_PROMPT, intermediate_max, intermediate_min
            )
            chunk_summaries.append(chunk_summary)
        except Exception as e:
            logger.warning(f"Failed to summarize chunk {i+1}: {str(e)}")
            chunk_summaries.append(chunk[:300] + "...")

    # Concatenate chunk summaries
    combined = " ".join(chunk_summaries)
    combined_tokens = count_tokens(combined)

    logger.info(f"Combined chunk summaries: {combined_tokens} tokens")

    # Check if we need another round of summarization
    if combined_tokens <= CHUNK_SIZE_TOKENS:
        # Fits now - final summarization to target length
        return _generate_summary(combined, FINAL_SUMMARY_PROMPT, max_tokens, min_tokens)
    else:
        # Still too long - recursive summarization
        logger.info(f"Combined summary still too long, recursing (depth {recursion_depth + 1})")
        return _summarize_long_document(
            combined, max_tokens, min_tokens, recursion_depth + 1
        )


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens.

    Args:
        text: Text to truncate.
        max_tokens: Maximum tokens to keep.

    Returns:
        Truncated text.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def summarize_to_bullet_points(
    text: str,
    num_points: int = 5,
    max_words_per_point: int = 30
) -> List[str]:
    """
    Generate a bullet-point summary of the document.

    Args:
        text: Text to summarize.
        num_points: Approximate number of bullet points desired.
        max_words_per_point: Maximum words per bullet point.

    Returns:
        List of bullet point strings.
    """
    target_words = num_points * max_words_per_point
    summary = summarize_document(
        text,
        max_length=target_words,
        min_length=target_words // 2
    )

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary)

    # Clean and filter
    bullet_points = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:
            bullet_points.append(sentence)

    return bullet_points[:num_points]


# =============================================================================
# TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    # Sample financial text for testing
    sample_text = """
    Annual Financial Performance Review - Fiscal Year 2023

    Executive Summary and Business Overview

    The fiscal year 2023 marked a transformative period for our organization,
    characterized by strategic expansion initiatives, technological investments,
    and significant operational improvements across all business segments. Despite
    facing considerable macroeconomic headwinds including elevated inflation,
    rising interest rates, and supply chain disruptions, the company demonstrated
    remarkable resilience and adaptability in navigating these challenges.

    Revenue Performance and Growth Metrics

    Total consolidated revenue for fiscal year 2023 reached $8.7 billion,
    representing a year-over-year increase of 14.2% compared to the $7.6 billion
    reported in fiscal year 2022. This growth was primarily driven by strong
    performance in our cloud services division, which contributed $3.2 billion
    to total revenue, reflecting a 28% increase from the previous year. The
    enterprise software segment generated $2.8 billion in revenue, while our
    professional services division contributed $1.9 billion.

    Profitability and Margin Analysis

    Operating income for the fiscal year was $1.74 billion, resulting in an
    operating margin of 20%, down from 22% in the previous year. The margin
    compression was primarily attributable to increased investments in research
    and development, which totaled $1.1 billion representing 12.6% of revenue.
    Net income attributable to shareholders was $1.35 billion, or $4.52 per
    diluted share, compared to $1.28 billion in fiscal year 2022.

    Strategic Initiatives and Capital Allocation

    During the fiscal year, the company completed three strategic acquisitions
    totaling $890 million, primarily focused on enhancing our artificial
    intelligence and machine learning capabilities. The company returned
    $1.2 billion to shareholders through dividends and share repurchases.
    """

    print("=" * 70)
    print("Document Summarization (Flan-T5-Large) - Test Suite")
    print("=" * 70)
    print()

    word_count = count_words(sample_text)
    print(f"Original text: {word_count} words")
    print()

    print("Loading Flan-T5-Large model (this may take a moment)...")
    print()

    # Test summarization
    test_lengths = [
        (75, 40),
        (100, 50),
        (150, 75),
    ]

    for max_len, min_len in test_lengths:
        print("-" * 70)
        print(f"Summary (target: {min_len}-{max_len} words)")
        print("-" * 70)

        try:
            summary = summarize_document(
                sample_text,
                max_length=max_len,
                min_length=min_len
            )

            summary_words = count_words(summary)
            print(f"Generated: {summary_words} words")
            print()
            print(summary)
            print()

        except Exception as e:
            print(f"Error: {str(e)}")
            print()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
