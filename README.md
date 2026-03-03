# Document Intelligence System

A Python-based document intelligence system that enables natural language question answering over both text and tabular data extracted from documents like annual reports and financial statements.

## Features

- **Text Question Answering** - Answer natural language questions from text content using DistilBERT fine-tuned on SQuAD
- **Table Question Answering** - Answer natural language questions from tabular data using Google's TAPAS model
- **PDF Text Extraction** - Extract text and tables from PDF documents using pdfplumber
- **Long Text Support** - Sliding window approach for processing texts exceeding model token limits
- **Aggregation Support** - Automatically performs SUM, AVERAGE, and COUNT operations on table data
- **Confidence Scoring** - Returns confidence scores for answer reliability

## How It Works

### Text QA
The text QA module uses [DistilBERT](https://huggingface.co/distilbert-base-cased-distilled-squad) fine-tuned on SQuAD (Stanford Question Answering Dataset) for extractive question answering. It:

- Extracts answers directly from text passages
- Handles long documents using a sliding window approach
- Returns confidence scores and answer positions

### Table QA
The table QA module uses [TAPAS (Table Parser)](https://github.com/google-research/tapas), a BERT-based model designed specifically for table understanding. Unlike traditional QA models, TAPAS understands:

- Row and column relationships
- Numerical values and their aggregations
- Cell positions and semantic meaning

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/document-intelligence-system.git
cd document-intelligence-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install transformers torch pandas pdfplumber
```

## Usage

### Text Question Answering

```python
from text_qa import answer_text_question

# Your document text
context = """
Company XYZ reported total revenue of $4.2 billion for Q4 2023,
representing a 12% increase year-over-year. Operating margins
declined from 28% to 24% due to increased R&D investments.
"""

# Ask questions
result = answer_text_question(
    question="What was the revenue?",
    context=context,
    confidence_threshold=0.3
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.2f}")
```

### Batch Text Questions

```python
from text_qa import batch_answer_questions

questions = [
    "What was the revenue?",
    "Why did margins decline?",
    "What is the growth rate?"
]

results = batch_answer_questions(questions, context)
for r in results:
    print(f"Q: {r['question']} -> A: {r['answer']}")
```

### Table Question Answering

```python
import pandas as pd
from table_qa import answer_table_question

# Create or load your table
data = {
    "Quarter": ["Q1", "Q2", "Q3", "Q4"],
    "Revenue": ["$1,200", "$1,450", "$1,380", "$1,620"],
    "Operating Margin": ["24%", "28%", "26%", "30%"]
}
table = pd.DataFrame(data)

# Ask questions
result = answer_table_question(
    question="What was Q2 revenue?",
    table=table,
    table_name="financial_summary"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.2f}")
```

### Batch Questions

```python
from table_qa import batch_answer_table_questions

questions = [
    "What was Q2 revenue?",
    "Which quarter had highest margin?",
    "What is total revenue?"
]

results = batch_answer_table_questions(questions, table)
for r in results:
    print(f"Q: {r['question']} -> A: {r['answer']}")
```

### PDF Text Extraction

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        tables = page.extract_tables()
```

## API Reference

### `answer_text_question(question, context, max_context_length, confidence_threshold)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | str | Natural language question about the text |
| `context` | str | The text content to search for answers |
| `max_context_length` | int | Maximum chunk size for long texts (default: 4000) |
| `confidence_threshold` | float | Minimum confidence score (default: 0.3) |

**Returns:** Dictionary containing:
- `answer` - The extracted answer
- `score` - Confidence score (0-1)
- `start` - Start position in original text
- `end` - End position in original text
- `success` - Boolean indicating if answer was found

### `answer_table_question(question, table, table_name, confidence_threshold)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | str | Natural language question about the table |
| `table` | pd.DataFrame | The table data to query |
| `table_name` | str | Optional identifier for the table |
| `confidence_threshold` | float | Minimum confidence score (default: 0.5) |

**Returns:** Dictionary containing:
- `answer` - The extracted answer
- `score` - Confidence score (0-1)
- `cells` - Coordinates of answer cells
- `aggregation` - Operation applied (NONE, SUM, AVERAGE, COUNT)
- `success` - Boolean indicating if answer was found

## Project Structure

```
document-intelligence-system/
├── text_qa.py           # Text QA module using DistilBERT
├── table_qa.py          # Table QA module using TAPAS
├── README.md            
└── .venv/               # Virtual environment
```

## Requirements

- Python 3.8+
- transformers
- torch
- pandas
- pdfplumber

## Supported Question Types

### Text Questions

| Type | Example |
|------|---------|
| Factual | "What was the revenue?" |
| Causal | "Why did margins decline?" |
| Descriptive | "What is the company's outlook?" |

### Table Questions

| Type | Example |
|------|---------|
| Direct lookup | "What was Q2 revenue?" |
| Aggregation | "What is total revenue?" |
| Comparison | "Which quarter had highest margin?" |
| Counting | "How many quarters exceeded $1,400?" |

## License

MIT License

## Acknowledgments

- [DistilBERT SQuAD](https://huggingface.co/distilbert-base-cased-distilled-squad) - Text QA model
- [Google TAPAS](https://github.com/google-research/tapas) - Table Parser model
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Model hosting and inference
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF extraction
