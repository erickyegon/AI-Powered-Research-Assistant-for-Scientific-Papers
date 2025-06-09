"""
AI-Powered Research Assistant - Production-Grade Streamlit Frontend

This module provides a comprehensive, professional-grade user interface for the
AI-Powered Research Assistant with the following features:

- Modern, responsive design with professional styling
- Multi-page application with navigation
- File upload and document management
- Real-time query processing with progress tracking
- Interactive results display with citations
- Session management and history
- Error handling and user feedback
- Performance monitoring and analytics
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any

import streamlit as st
import requests

# Configure page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/research-assistant',
        'Report a bug': 'https://github.com/your-repo/research-assistant/issues',
        'About': "AI-Powered Research Assistant v1.0 - Professional research paper analysis tool"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #17a2b8;
        --light-bg: #f8f9fa;
        --dark-bg: #343a40;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom header */
    .main-header {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    /* Card styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }

    /* Query input styling */
    .query-container {
        background: var(--light-bg);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* Results styling */
    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* Citation styling */
    .citation {
        background: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--info-color);
    }

    /* Status indicators */
    .status-success {
        color: var(--success-color);
        font-weight: bold;
    }

    .status-warning {
        color: var(--warning-color);
        font-weight: bold;
    }

    .status-info {
        color: var(--info-color);
        font-weight: bold;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: var(--light-bg);
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
</style>
""", unsafe_allow_html=True)

# Configuration - Support both local and production
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_BASE_URL = f"https://{BACKEND_URL}" if not BACKEND_URL.startswith("http") else BACKEND_URL

# Fallback for local development
if "127.0.0.1" in API_BASE_URL or "localhost" in API_BASE_URL:
    API_BASE_URL = "http://127.0.0.1:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/documents/upload"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"


def check_api_health() -> Dict[str, Any]:
    """Check API health status."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp


def display_header():
    """Display the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 AI-Powered Research Assistant</h1>
        <p>Professional-grade scientific paper analysis and question answering</p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display the application sidebar with navigation and status."""
    with st.sidebar:
        st.markdown("### 📊 System Status")

        # API Health Check
        health_status = check_api_health()
        if health_status["status"] == "healthy":
            st.markdown('<p class="status-success">✅ API Connected</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">❌ API Disconnected</p>', unsafe_allow_html=True)
            st.error(f"Error: {health_status.get('error', 'Unknown error')}")

        st.markdown("---")

        # Session Information
        st.markdown("### 📝 Session Info")
        st.info(f"**Session ID:** {st.session_state.session_id}")
        st.info(f"**Queries:** {len(st.session_state.query_history)}")
        st.info(f"**Documents:** {len(st.session_state.uploaded_documents)}")

        st.markdown("---")

        # Quick Actions
        st.markdown("### ⚡ Quick Actions")

        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.success("History cleared!")
            st.experimental_rerun()

        if st.button("📊 Export Session"):
            export_data = {
                "session_id": st.session_state.session_id,
                "query_history": st.session_state.query_history,
                "uploaded_documents": st.session_state.uploaded_documents,
                "timestamp": datetime.now().isoformat()
            }
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"research_session_{st.session_state.session_id}.json",
                mime="application/json"
            )

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")

        # API Configuration
        with st.expander("🔧 API Configuration"):
            api_url = st.text_input("API Base URL", value=API_BASE_URL)
            if api_url != API_BASE_URL:
                st.warning("API URL changed. Restart required.")

        # Display Options
        with st.expander("🎨 Display Options"):
            show_debug = st.checkbox("Show Debug Info", value=False)
            show_timestamps = st.checkbox("Show Timestamps", value=True)
            max_results = st.slider("Max Results", 1, 20, 5)

            # Store in session state
            st.session_state.show_debug = show_debug
            st.session_state.show_timestamps = show_timestamps
            st.session_state.max_results = max_results


def query_interface():
    """Main query interface."""
    st.markdown("### 🔍 Ask Your Research Question")

    # Query input with enhanced features
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_area(
            "Enter your research question:",
            value=st.session_state.current_query,
            height=100,
            placeholder="e.g., What are the latest developments in machine learning for healthcare?",
            help="Ask detailed questions about scientific papers, methodologies, findings, or comparisons."
        )

    with col2:
        st.markdown("#### Query Options")
        query_type = st.selectbox(
            "Query Type:",
            ["General Q&A", "Summarization", "Comparison", "Methodology", "Citation Analysis"]
        )

        include_citations = st.checkbox("Include Citations", value=True)
        detailed_response = st.checkbox("Detailed Response", value=False)

    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Submit Query", use_container_width=True, type="primary"):
            if query.strip():
                process_query(query, query_type, include_citations, detailed_response)
            else:
                st.warning("Please enter a question before submitting.")

    # Query suggestions
    st.markdown("#### 💡 Example Questions")
    example_queries = [
        "What are the main challenges in deep learning for medical image analysis?",
        "Compare different approaches to natural language processing in recent papers",
        "What methodologies are commonly used in climate change research?",
        "Summarize recent findings on COVID-19 vaccine effectiveness",
        "What are the ethical considerations in AI research?"
    ]

    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(f"📝 Example {i+1}", help=example):
                st.session_state.current_query = example
                st.experimental_rerun()


def process_query(query: str, query_type: str, include_citations: bool, detailed_response: bool):
    """Process a user query and display results."""
    start_time = time.time()

    # Update session state
    st.session_state.current_query = query
    st.session_state.processing_status = "processing"

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Query preprocessing
        status_text.text("🔄 Preprocessing query...")
        progress_bar.progress(20)

        # Prepare request payload
        payload = {
            "query": query,
            "query_type": query_type.lower().replace(" ", "_"),
            "include_citations": include_citations,
            "detailed_response": detailed_response,
            "session_id": st.session_state.session_id,
            "max_results": st.session_state.get("max_results", 5)
        }

        # Step 2: Send request to API
        status_text.text("🌐 Sending request to AI...")
        progress_bar.progress(40)

        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json=payload,
            timeout=60
        )

        progress_bar.progress(60)

        if response.status_code == 200:
            data = response.json()

            # Step 3: Process response
            status_text.text("📊 Processing response...")
            progress_bar.progress(80)

            # Add to query history
            query_record = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "query_type": query_type,
                "response": data.get("response", ""),
                "sources": data.get("sources", []),
                "metadata": data.get("metadata", {}),
                "processing_time": time.time() - start_time
            }

            st.session_state.query_history.append(query_record)

            # Step 4: Display results
            status_text.text("✅ Complete!")
            progress_bar.progress(100)

            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            # Display results
            display_query_results(data, query_record)

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Please try again with a simpler query.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to the API. Please check if the backend is running.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
    finally:
        st.session_state.processing_status = "idle"
        progress_bar.empty()
        status_text.empty()


def display_query_results(data: Dict[str, Any], query_record: Dict[str, Any]):
    """Display query results in a professional format."""
    st.markdown("---")
    st.markdown("## 📋 Results")

    # Response section
    with st.container():
        st.markdown("### 💬 AI Response")

        # Response quality indicator
        quality_score = data.get("metadata", {}).get("quality_score", 0)
        if quality_score > 8:
            st.success("🌟 High Quality Response")
        elif quality_score > 6:
            st.info("✅ Good Quality Response")
        else:
            st.warning("⚠️ Response may need refinement")

        # Main response
        response_text = data.get("response", "No response generated.")
        st.markdown(f"""
        <div class="result-container">
            {response_text}
        </div>
        """, unsafe_allow_html=True)

        # Response metadata
        if st.session_state.get("show_debug", False):
            with st.expander("🔍 Response Metadata"):
                st.json(data.get("metadata", {}))

    # Sources section
    sources = data.get("sources", [])
    if sources:
        st.markdown("### 📚 Sources & Citations")

        for i, source in enumerate(sources, 1):
            with st.expander(f"📄 Source {i}", expanded=i <= 3):
                st.markdown(f"""
                <div class="citation">
                    <strong>Citation [{i}]:</strong><br>
                    {source.get('content', source)[:500]}...
                </div>
                """, unsafe_allow_html=True)

                # Source metadata if available
                if isinstance(source, dict) and 'metadata' in source:
                    metadata = source['metadata']
                    if metadata:
                        st.markdown("**Metadata:**")
                        for key, value in metadata.items():
                            st.markdown(f"- **{key.title()}:** {value}")

    # Performance metrics
    processing_time = query_record.get("processing_time", 0)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")

    with col2:
        st.metric("📄 Sources Found", len(sources))

    with col3:
        quality_score = data.get("metadata", {}).get("quality_score", 0)
        st.metric("⭐ Quality Score", f"{quality_score:.1f}/10")

    with col4:
        response_length = len(data.get("response", ""))
        st.metric("📝 Response Length", f"{response_length} chars")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.query_history)}"):
            st.success("Thank you for your feedback!")

    with col2:
        if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.query_history)}"):
            feedback = st.text_input("How can we improve?", key=f"feedback_{len(st.session_state.query_history)}")
            if feedback:
                st.info("Feedback recorded. Thank you!")

    with col3:
        if st.button("🔄 Refine Query", key=f"refine_{len(st.session_state.query_history)}"):
            st.session_state.current_query = query_record["query"]
            st.experimental_rerun()


def main():
    """Main application function."""
    # Display header
    display_header()

    # Display sidebar
    display_sidebar()

    # Main content area
    query_interface()

    # Query history section
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("## 📚 Query History")

        # History controls
        col1, col2 = st.columns([3, 1])
        with col1:
            show_history = st.checkbox("Show Query History", value=False)
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.query_history = []
                st.experimental_rerun()

        if show_history:
            for i, record in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
                with st.expander(f"Query {len(st.session_state.query_history) - i}: {record['query'][:50]}..."):
                    st.markdown(f"**Time:** {format_timestamp(record['timestamp'])}")
                    st.markdown(f"**Type:** {record['query_type']}")
                    st.markdown(f"**Processing Time:** {record['processing_time']:.2f}s")
                    st.markdown(f"**Response:** {record['response'][:200]}...")

                    if st.button(f"🔄 Rerun Query", key=f"rerun_{i}"):
                        st.session_state.current_query = record['query']
                        st.experimental_rerun()


if __name__ == "__main__":
    main()
