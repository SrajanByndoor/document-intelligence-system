"""
Question Router Module

This module provides functionality to route questions to the appropriate
QA model (text-based or table-based) based on keyword analysis and patterns.

Routing Strategy:
-----------------
The router uses a keyword-based heuristic approach to determine whether a
question is better suited for:

1. TABLE QA (TAPAS model):
   - Questions asking for specific values, numbers, or metrics
   - Questions requiring aggregation (sum, average, count)
   - Questions involving comparison between data points
   - Questions about financial metrics (revenue, profit, margin)

2. TEXT QA (DistilBERT model):
   - Questions asking for explanations or reasoning
   - Questions about strategies, outlooks, or qualitative analysis
   - Questions requiring contextual understanding
   - Questions about risks, factors, or descriptive content

This simple keyword-based approach can be enhanced later with ML-based
classification for more accurate routing.
"""

from typing import List, Tuple, Set
import re


# =============================================================================
# KEYWORD DEFINITIONS
# =============================================================================

# Table-oriented keywords and phrases
# These indicate questions that likely need structured data lookup or aggregation
TABLE_INDICATORS: List[str] = [
    # Quantitative question starters
    "how much",
    "how many",
    "what is the value",
    "what was the value",
    "what is the amount",
    "what was the amount",

    # Aggregation operations
    "total",
    "sum",
    "average",
    "mean",

    # Comparison operations
    "compare",
    "comparison",
    "highest",
    "lowest",
    "maximum",
    "minimum",
    "most",
    "least",
    "greater",
    "less than",
    "between",

    # Percentage and ratio terms
    "percent",
    "percentage",
    "ratio",
    "rate",

    # Counting terms
    "number",
    "count",
    "quantity",

    # Time-based lookups (often in tabular data)
    "q1", "q2", "q3", "q4",
    "quarter",
    "year",
    "month",
    "annual",
    "quarterly",

    # Financial metrics - commonly found in tables
    "revenue",
    "revenues",
    "sales",
    "profit",
    "profits",
    "income",
    "earnings",
    "margin",
    "margins",
    "ebitda",
    "cash flow",
    "assets",
    "liabilities",
    "equity",
    "debt",
    "expenses",
    "costs",
    "capex",
    "opex",
]

# Text-oriented keywords and phrases
# These indicate questions requiring contextual understanding and explanation
TEXT_INDICATORS: List[str] = [
    # Explanation-seeking starters
    "why",
    "explain",
    "describe",
    "discuss",
    "elaborate",
    "clarify",

    # Analysis and reasoning
    "analyze",
    "analysis",
    "evaluate",
    "assess",
    "interpret",

    # Plural/list questions (often need paragraph context)
    "what are",
    "what were",
    "who are",
    "which are",
    "list the",

    # Qualitative business terms
    "outlook",
    "strategy",
    "strategic",
    "risk",
    "risks",
    "factor",
    "factors",
    "challenge",
    "challenges",
    "opportunity",
    "opportunities",

    # Forward-looking terms
    "future",
    "plan",
    "plans",
    "planning",
    "expect",
    "expectation",
    "forecast",
    "predict",
    "projection",

    # Contextual/narrative terms
    "reason",
    "reasons",
    "cause",
    "causes",
    "impact",
    "effect",
    "consequence",
    "implication",
    "overview",
    "summary",
    "background",
    "context",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize_question(question: str) -> str:
    """
    Normalize a question for consistent pattern matching.

    Normalization steps:
    1. Convert to lowercase for case-insensitive matching
    2. Remove extra whitespace
    3. Strip leading/trailing whitespace

    Args:
        question: The original question string

    Returns:
        Normalized question string
    """
    # Convert to lowercase
    normalized = question.lower()

    # Replace multiple whitespace with single space
    normalized = re.sub(r'\s+', ' ', normalized)

    # Strip leading/trailing whitespace
    normalized = normalized.strip()

    return normalized


def _count_indicator_matches(
    question: str,
    indicators: List[str]
) -> Tuple[int, Set[str]]:
    """
    Count how many indicators from a list appear in the question.

    This function performs substring matching, checking if each indicator
    phrase appears anywhere in the question. It returns both the count
    and the set of matched indicators for debugging/logging purposes.

    Args:
        question: The normalized question string
        indicators: List of indicator phrases to search for

    Returns:
        Tuple of (match_count, set_of_matched_indicators)
    """
    matched = set()

    for indicator in indicators:
        # Check if the indicator phrase appears in the question
        # Using word boundary awareness for single words to avoid partial matches
        if len(indicator.split()) == 1:
            # Single word: use word boundary regex to avoid partial matches
            # e.g., "sum" shouldn't match "summary"
            pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(pattern, question):
                matched.add(indicator)
        else:
            # Multi-word phrase: simple substring match is fine
            if indicator in question:
                matched.add(indicator)

    return len(matched), matched


def _check_question_structure(question: str) -> str:
    """
    Analyze question structure for additional routing hints.

    Some question structures strongly indicate the type of answer needed:
    - "What is/was the X?" -> Often table (looking up a value)
    - "Why did X?" -> Always text (asking for explanation)
    - "How did X happen?" -> Text (asking for process/explanation)
    - "How much/many?" -> Table (asking for quantity)

    Args:
        question: The normalized question string

    Returns:
        'table', 'text', or 'neutral' based on structure analysis
    """
    # Strong text indicators based on question structure
    text_patterns = [
        r'^why\b',           # Questions starting with "why"
        r'^how did\b',       # "How did X happen?"
        r'^how does\b',      # "How does X work?"
        r'^how do\b',        # "How do they..."
        r'^how can\b',       # "How can we..."
        r'^how should\b',    # "How should..."
        r'^what caused\b',   # "What caused..."
        r'^what led to\b',   # "What led to..."
        r'^explain\b',       # "Explain..."
        r'^describe\b',      # "Describe..."
    ]

    # Strong table indicators based on question structure
    table_patterns = [
        r'^how much\b',      # "How much revenue..."
        r'^how many\b',      # "How many customers..."
        r'^what is the \w+ of\b',  # "What is the value of..."
        r'^what was the \w+ of\b', # "What was the amount of..."
        r'^what is the total\b',   # "What is the total..."
        r'^what was the total\b',  # "What was the total..."
        r'^which \w+ (has|had|is|was) (the )?(highest|lowest|most|least)\b',  # Superlative questions
    ]

    # Check text patterns first
    for pattern in text_patterns:
        if re.search(pattern, question):
            return 'text'

    # Check table patterns
    for pattern in table_patterns:
        if re.search(pattern, question):
            return 'table'

    return 'neutral'


# =============================================================================
# MAIN ROUTING FUNCTION
# =============================================================================

def route_question(question: str) -> str:
    """
    Route a question to the appropriate QA model based on content analysis.

    Routing Algorithm:
    ------------------
    1. Normalize the question (lowercase, clean whitespace)
    2. Check question structure for strong routing signals
    3. Count matches against table indicators
    4. Count matches against text indicators
    5. Apply decision logic:
       - If structure analysis gives strong signal, use it
       - If table indicators > text indicators, route to 'table'
       - If text indicators > table indicators, route to 'text'
       - If equal (tie), prefer 'table' (more specific answers)
       - If no indicators match, default to 'text'

    The preference for 'table' on ties is based on the reasoning that:
    - Table questions typically have more precise answers
    - If a question could go either way, a table lookup is less likely
      to give an incorrect answer than text extraction
    - Users asking ambiguous questions often want specific data

    Args:
        question: The question to route

    Returns:
        'text' or 'table' indicating which QA model should handle the question

    Examples:
        >>> route_question("What was the revenue in Q2?")
        'table'
        >>> route_question("Why did margins decline?")
        'text'
        >>> route_question("Compare Q1 and Q2 performance")
        'table'
        >>> route_question("Explain the company's growth strategy")
        'text'
    """
    # Handle edge cases
    if not question or not question.strip():
        return 'text'  # Default for empty questions

    # Step 1: Normalize the question
    normalized = _normalize_question(question)

    # Step 2: Check question structure for strong signals
    # Some question patterns are very clear indicators
    structure_hint = _check_question_structure(normalized)

    # If structure analysis gives a strong signal, weight it heavily
    # but don't make it absolute (keywords can override in some cases)
    structure_weight = 1 if structure_hint != 'neutral' else 0

    # Step 3: Count indicator matches
    table_count, table_matches = _count_indicator_matches(
        normalized, TABLE_INDICATORS
    )
    text_count, text_matches = _count_indicator_matches(
        normalized, TEXT_INDICATORS
    )

    # Step 4: Calculate weighted scores
    # Structure hint adds weight to one side
    if structure_hint == 'table':
        table_score = table_count + structure_weight
        text_score = text_count
    elif structure_hint == 'text':
        table_score = table_count
        text_score = text_count + structure_weight
    else:
        table_score = table_count
        text_score = text_count

    # Step 5: Make routing decision
    # Decision logic with explanations:

    if table_score == 0 and text_score == 0:
        # No indicators found - default to text
        # Text QA is more general-purpose and can handle diverse questions
        return 'text'

    if table_score > text_score:
        # More table indicators - route to table QA
        return 'table'

    if text_score > table_score:
        # More text indicators - route to text QA
        return 'text'

    # Tie: prefer table
    # Rationale: Table questions typically need specific values, and if
    # a question has equal indicators for both, it's safer to try the
    # more precise table lookup first
    return 'table'


def route_question_with_details(question: str) -> dict:
    """
    Route a question and return detailed analysis of the routing decision.

    Useful for debugging and understanding why a question was routed
    to a particular model.

    Args:
        question: The question to route

    Returns:
        Dictionary containing:
            - 'route': The routing decision ('text' or 'table')
            - 'question': Original question
            - 'normalized': Normalized question
            - 'structure_hint': Result of structure analysis
            - 'table_matches': Set of matched table indicators
            - 'text_matches': Set of matched text indicators
            - 'table_score': Final table score
            - 'text_score': Final text score
            - 'reason': Human-readable explanation of decision
    """
    if not question or not question.strip():
        return {
            'route': 'text',
            'question': question,
            'normalized': '',
            'structure_hint': 'neutral',
            'table_matches': set(),
            'text_matches': set(),
            'table_score': 0,
            'text_score': 0,
            'reason': 'Empty question - defaulting to text'
        }

    normalized = _normalize_question(question)
    structure_hint = _check_question_structure(normalized)

    table_count, table_matches = _count_indicator_matches(
        normalized, TABLE_INDICATORS
    )
    text_count, text_matches = _count_indicator_matches(
        normalized, TEXT_INDICATORS
    )

    # Calculate scores
    structure_weight = 1 if structure_hint != 'neutral' else 0

    if structure_hint == 'table':
        table_score = table_count + structure_weight
        text_score = text_count
    elif structure_hint == 'text':
        table_score = table_count
        text_score = text_count + structure_weight
    else:
        table_score = table_count
        text_score = text_count

    # Determine route and reason
    if table_score == 0 and text_score == 0:
        route = 'text'
        reason = 'No indicators found - defaulting to text'
    elif table_score > text_score:
        route = 'table'
        reason = f'Table indicators ({table_score}) > Text indicators ({text_score})'
    elif text_score > table_score:
        route = 'text'
        reason = f'Text indicators ({text_score}) > Table indicators ({table_score})'
    else:
        route = 'table'
        reason = f'Tie ({table_score} each) - preferring table for precision'

    return {
        'route': route,
        'question': question,
        'normalized': normalized,
        'structure_hint': structure_hint,
        'table_matches': table_matches,
        'text_matches': text_matches,
        'table_score': table_score,
        'text_score': text_score,
        'reason': reason
    }


# =============================================================================
# TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Question Router - Test Suite")
    print("=" * 70)
    print()

    # Define test questions with expected routes
    # 5 questions expected to route to TABLE
    table_questions = [
        "What was the revenue?",
        "Compare Q1 and Q2 margins",
        "What is the total profit?",
        "Which quarter had the highest growth?",
        "How much did expenses increase?",
    ]

    # 5 questions expected to route to TEXT
    text_questions = [
        "Why did revenue decline?",
        "Explain the risk factors",
        "What is the company's strategy?",
        "Describe the market outlook",
        "What are the main challenges facing the business?",
    ]

    # Combine all questions
    all_questions = [
        (q, 'table') for q in table_questions
    ] + [
        (q, 'text') for q in text_questions
    ]

    print("Testing Question Routing")
    print("-" * 70)
    print()

    correct = 0
    total = len(all_questions)

    for question, expected in all_questions:
        result = route_question(question)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1

        print(f"Q: {question}")
        print(f"   Route: {result.upper()} (expected: {expected.upper()}) {match}")
        print()

    print("-" * 70)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print()

    # Show detailed analysis for a few examples
    print("=" * 70)
    print("Detailed Routing Analysis")
    print("=" * 70)
    print()

    detailed_examples = [
        "What was Q2 revenue and why did it decline?",  # Mixed - should favor table
        "Explain the revenue growth strategy",           # Mixed - should favor text
        "What happened?",                                # Ambiguous - default text
    ]

    for question in detailed_examples:
        details = route_question_with_details(question)

        print(f"Question: {question}")
        print(f"  Route: {details['route'].upper()}")
        print(f"  Reason: {details['reason']}")
        print(f"  Structure hint: {details['structure_hint']}")
        print(f"  Table matches: {details['table_matches'] or 'none'}")
        print(f"  Text matches: {details['text_matches'] or 'none'}")
        print()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
