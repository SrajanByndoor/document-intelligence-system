#!/usr/bin/env python3
"""
Financial Document Intelligence System - Streamlit Web UI

A polished, production-ready web interface for analyzing financial documents
using AI-powered question answering and summarization.

Running the Application:
------------------------
    streamlit run app.py

    Or with custom port:
    streamlit run app.py --server.port 8501

Requirements:
-------------
    pip install streamlit

Author: Document Intelligence System
"""

import os
import sys
import time
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import main pipeline functions
import main


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Financial Document Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-org/document-intelligence-system',
        'Report a bug': 'https://github.com/your-org/document-intelligence-system/issues',
        'About': '''
        ## Financial Document Intelligence System

        AI-powered document analysis for financial reports.

        **Features:**
        - Intelligent question answering
        - Automatic document summarization
        - Table and text extraction
        '''
    }
)


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for professional styling."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    /* Answer boxes */
    .answer-box-high {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    .answer-box-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    .answer-box-low {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    .answer-text {
        font-size: 1.1rem;
        color: #212529;
        margin: 0;
        line-height: 1.6;
    }

    /* Info cards */
    .info-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }

    .source-badge-text {
        background-color: #e3f2fd;
        color: #1565c0;
    }

    .source-badge-table {
        background-color: #e8f5e9;
        color: #2e7d32;
    }

    /* History items */
    .history-item {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    .history-question {
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }

    .history-answer {
        color: #495057;
        font-size: 0.95rem;
    }

    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }

    .footer a {
        color: #1e3a5f;
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }

    /* File upload styling */
    .uploadedFile {
        border: 2px dashed #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e3a5f;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #1e3a5f;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Instructions box */
    .instructions-box {
        background: #e8f4f8;
        border: 1px solid #b8daff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None

    if 'last_summary' not in st.session_state:
        st.session_state.last_summary = None

    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False


# =============================================================================
# CACHING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Pre-load ML models to avoid repeated loading.
    Uses Streamlit's cache_resource for persistent caching.
    """
    # Import modules to trigger model loading
    import text_qa
    import table_qa
    import summarizer
    import question_router

    return True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Save uploaded file to a temporary location.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Path to saved file or None if failed.
    """
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(upload_dir, filename)

        # Save file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    except Exception as e:
        st.error(f"Failed to save file: {str(e)}")
        return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_confidence_level(score: float) -> str:
    """Get confidence level string from score."""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"


def render_answer_box(answer: str, confidence: str):
    """Render answer in colored box based on confidence."""
    css_class = f"answer-box-{confidence}"
    st.markdown(f"""
    <div class="{css_class}">
        <p class="answer-text">{answer}</p>
    </div>
    """, unsafe_allow_html=True)


def add_to_history(question: str, answer: str, score: float, source_type: str):
    """Add question-answer pair to history."""
    st.session_state.history.insert(0, {
        'question': question,
        'answer': answer,
        'score': score,
        'source_type': source_type,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })

    # Keep only last 5 items
    st.session_state.history = st.session_state.history[:5]


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the main header section."""
    st.markdown("""
    <div class="main-header">
        <h1>Financial Document Intelligence System</h1>
        <p>AI-powered analysis for financial documents. Upload a PDF to get instant summaries and answers to your questions.</p>
    </div>
    """, unsafe_allow_html=True)


def render_instructions():
    """Render usage instructions."""
    with st.expander("How to Use This Tool", expanded=False):
        st.markdown("""
        ### Getting Started

        1. **Upload a Document**: Use the file uploader to select a PDF financial document (annual reports, quarterly filings, etc.)

        2. **Generate Summary**: Click "Generate Summary" to get an AI-powered overview of the document

        3. **Ask Questions**: Type your question or select from common financial questions, then click "Get Answer"

        ### Tips for Best Results

        - **Be specific**: "What was the Q3 2023 revenue?" works better than "What was the revenue?"
        - **Financial terms**: The system understands terms like revenue, EBITDA, operating margin, etc.
        - **Table data**: Questions about specific numbers often get answered from tables
        - **Explanations**: Questions starting with "Why" or "How" typically get answered from text

        ### Understanding Confidence Scores

        - **High (70%+)**: The system is confident in the answer
        - **Medium (40-70%)**: Answer may need verification
        - **Low (<40%)**: Answer might not be accurate
        """)


def render_sidebar():
    """Render the sidebar with advanced options."""
    with st.sidebar:
        st.markdown("## Advanced Options")
        st.markdown("---")

        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum confidence score to accept an answer. Lower values return more answers but with less certainty."
        )

        st.markdown("---")

        # Summary length
        st.markdown("### Summary Settings")
        summary_length = st.slider(
            "Summary Length (words)",
            min_value=50,
            max_value=300,
            value=150,
            step=25,
            help="Target length for document summaries."
        )

        st.markdown("---")

        # Technical details toggle
        show_technical = st.toggle(
            "Show Technical Details",
            value=False,
            help="Display additional technical information about the AI processing."
        )

        st.markdown("---")

        # System info
        st.markdown("### System Info")

        if st.session_state.models_loaded:
            st.success("Models loaded")
        else:
            st.warning("Models loading...")

        if st.session_state.uploaded_file_path:
            st.info("Document ready")
        else:
            st.info("No document loaded")

        st.markdown("---")

        # About section
        st.markdown("### About")
        st.markdown("""
        This system uses state-of-the-art AI models:
        - **DistilBERT** for text QA
        - **TAPAS** for table QA
        - **DistilBART** for summarization
        """)

        return confidence_threshold, summary_length, show_technical


def render_file_upload():
    """Render the file upload section."""
    st.markdown("### Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a financial document (annual report, quarterly filing, etc.)",
        key="pdf_uploader"
    )

    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)

        with col2:
            st.metric("File Size", format_file_size(uploaded_file.size))

        with col3:
            st.metric("Type", "PDF")

        # Save file if not already saved or if different file
        if (st.session_state.uploaded_file_path is None or
            not st.session_state.uploaded_file_path.endswith(uploaded_file.name)):

            with st.spinner("Saving document..."):
                file_path = save_uploaded_file(uploaded_file)

                if file_path:
                    st.session_state.uploaded_file_path = file_path
                    st.success("Document uploaded successfully!")

                    # Clear previous results
                    st.session_state.last_summary = None
                else:
                    st.error("Failed to save document. Please try again.")

    return uploaded_file


def render_summary_section(summary_length: int, show_technical: bool):
    """Render the document summary section."""
    st.markdown("### Document Summary")

    if st.session_state.uploaded_file_path is None:
        st.info("Please upload a document first.")
        return

    col1, col2 = st.columns([1, 4])

    with col1:
        generate_summary = st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True
        )

    if generate_summary:
        with st.spinner("Analyzing document and generating summary..."):
            try:
                start_time = time.time()

                result = main.summarize_document_content(
                    st.session_state.uploaded_file_path,
                    max_length=summary_length,
                    min_length=summary_length // 2
                )

                processing_time = time.time() - start_time
                st.session_state.last_summary = result
                st.session_state.last_summary['ui_processing_time'] = processing_time

            except FileNotFoundError:
                st.error("Document file not found. Please re-upload.")
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

    # Display summary if available
    if st.session_state.last_summary:
        result = st.session_state.last_summary

        if result.get('success', False):
            with st.expander("Document Summary", expanded=True):
                st.markdown(result['summary'])

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Original", f"{result['original_length']:,} words")

                with col2:
                    st.metric("Summary", f"{result['summary_length']} words")

                with col3:
                    compression = result.get('compression_ratio', 0) * 100
                    st.metric("Compression", f"{compression:.1f}%")

                with col4:
                    proc_time = result.get('ui_processing_time', result.get('processing_time', 0))
                    st.metric("Time", f"{proc_time:.1f}s")

                if show_technical:
                    st.markdown("---")
                    st.markdown("**Technical Details:**")
                    st.json({
                        'total_pages': result.get('total_pages', 0),
                        'processing_time': result.get('processing_time', 0),
                        'compression_ratio': result.get('compression_ratio', 0)
                    })
        else:
            st.error(f"{result.get('message', 'Failed to generate summary')}")


def render_qa_section(confidence_threshold: float, show_technical: bool):
    """Render the question answering section."""
    st.markdown("### Ask Questions")

    if st.session_state.uploaded_file_path is None:
        st.info("Please upload a document first.")
        return

    # Common questions dropdown
    common_questions = {
        "Select a question or type your own...": "",
        "What was the total revenue?": "What was the total revenue?",
        "What is the operating margin?": "What is the operating margin?",
        "Why did profits change?": "Why did profits change compared to the previous period?",
        "What are the main risks?": "What are the main risk factors mentioned in the document?",
        "What is the company's strategy?": "What is the company's strategy for growth?",
        "What were the key achievements?": "What were the key achievements and milestones?",
    }

    selected_question = st.selectbox(
        "Common Financial Questions",
        options=list(common_questions.keys()),
        help="Select a pre-defined question or choose 'Custom' to type your own"
    )

    # Text input for custom question
    question_input = st.text_input(
        "Your Question",
        value=common_questions.get(selected_question, ""),
        placeholder="e.g., What was the revenue growth in Q4?",
        help="Type your question about the document here"
    )

    # Example questions as help text
    st.caption("**Examples:** 'What was the net income?', 'Explain the revenue decline', 'What are the growth drivers?'")

    col1, col2 = st.columns([1, 4])

    with col1:
        get_answer = st.button(
            "Get Answer",
            type="primary",
            use_container_width=True,
            disabled=not question_input.strip()
        )

    if get_answer and question_input.strip():
        with st.spinner("Processing your question..."):
            try:
                start_time = time.time()

                result = main.process_document_query(
                    st.session_state.uploaded_file_path,
                    question_input.strip(),
                    confidence_threshold=confidence_threshold
                )

                processing_time = time.time() - start_time

                # Display results
                st.markdown("---")
                st.markdown("#### Answer")

                if result.get('success', False):
                    confidence = get_confidence_level(result.get('score', 0))
                    render_answer_box(result.get('answer', 'No answer found'), confidence)

                    # Confidence score as progress bar
                    score = result.get('score', 0)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Confidence Score**")
                        st.progress(score)
                        confidence_label = "High" if score >= 0.7 else ("Medium" if score >= 0.4 else "Low")
                        st.caption(f"{confidence_label}: {score:.1%} confidence")

                    with col2:
                        source_type = result.get('source_type', 'unknown').upper()
                        source_badge = "source-badge-table" if source_type == "TABLE" else "source-badge-text"
                        st.markdown("**Source Type**")
                        st.markdown(f"""
                        <span class="source-badge {source_badge}">{source_type}</span>
                        """, unsafe_allow_html=True)

                        page = result.get('source_page', -1)
                        if page > 0:
                            st.caption(f"Page {page}")

                    with col3:
                        st.markdown("**Processing Time**")
                        st.caption(f"{processing_time:.2f} seconds")

                    # Add to history
                    add_to_history(
                        question_input.strip(),
                        result.get('answer', ''),
                        result.get('score', 0),
                        result.get('source_type', 'unknown')
                    )

                    # Technical details
                    if show_technical:
                        with st.expander("Technical Details", expanded=False):
                            st.json({
                                'routing_decision': result.get('routing_decision', 'unknown'),
                                'source_details': result.get('source_details', {}),
                                'processing_time': result.get('processing_time', 0),
                                'message': result.get('message', '')
                            })

                else:
                    st.warning(f"{result.get('message', 'Could not find an answer to your question.')}")

                    if show_technical:
                        with st.expander("Technical Details", expanded=False):
                            st.json(result)

            except FileNotFoundError:
                st.error("Document file not found. Please re-upload.")
            except ValueError as e:
                st.error(f"Invalid input: {str(e)}")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")


def render_history_section():
    """Render the results history section."""
    st.markdown("### Recent Questions")

    if not st.session_state.history:
        st.info("No questions asked yet. Your recent questions will appear here.")
        return

    col1, col2 = st.columns([4, 1])

    with col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    for i, item in enumerate(st.session_state.history):
        confidence = get_confidence_level(item['score'])
        confidence_label = "[High]" if confidence == "high" else ("[Medium]" if confidence == "medium" else "[Low]")
        source_label = "[Table]" if item['source_type'] == 'table' else "[Text]"

        with st.container():
            st.markdown(f"""
            <div class="history-item">
                <div class="history-question">{confidence_label} Q: {item['question']}</div>
                <div class="history-answer">{source_label} A: {item['answer'][:200]}{'...' if len(item['answer']) > 200 else ''}</div>
                <small style="color: #6c757d;">Score: {item['score']:.1%} | Source: {item['source_type'].upper()} | {item['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)


def render_footer():
    """Render the footer section."""
    st.markdown("""
    <div class="footer">
        <p>
            <strong>Financial Document Intelligence System</strong><br>
            Powered by DistilBERT, TAPAS, and DistilBART models<br>
            <a href="https://github.com/your-org/document-intelligence-system" target="_blank">GitHub</a> |
            <a href="https://github.com/your-org/document-intelligence-system/issues" target="_blank">Report an Issue</a>
        </p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">
            © 2024 Document Intelligence System. For research and educational purposes.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main_app():
    """Main application entry point."""
    # Initialize
    initialize_session_state()
    inject_custom_css()

    # Load models in background
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models... This may take a moment on first run."):
            try:
                load_models()
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"Failed to load models: {str(e)}")
                return

    # Render sidebar and get settings
    confidence_threshold, summary_length, show_technical = render_sidebar()

    # Main content
    render_header()
    render_instructions()

    st.markdown("---")

    # File upload
    render_file_upload()

    st.markdown("---")

    # Two-column layout for summary and QA
    col1, col2 = st.columns(2)

    with col1:
        render_summary_section(summary_length, show_technical)

    with col2:
        render_qa_section(confidence_threshold, show_technical)

    st.markdown("---")

    # History section
    render_history_section()

    # Footer
    render_footer()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main_app()
