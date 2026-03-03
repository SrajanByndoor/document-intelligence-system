# Document Intelligence System

A Python-based document intelligence system that enables natural language question answering over both text and tabular data extracted from documents like annual reports and financial statements.

## Features

- **Text Question Answering** - Answer natural language questions from text content using DistilBERT fine-tuned on SQuAD
- **Table Question Answering** - Answer natural language questions from tabular data using Google's TAPAS model
- **PDF Text Extraction** - Extract text and tables from PDF documents using pdfplumber
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
