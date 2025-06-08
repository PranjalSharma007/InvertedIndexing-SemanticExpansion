import streamlit as st
from index_builder import IndexBuilder
from query_processor import QueryProcessor
import os
import re

# Set page config
st.set_page_config(
    page_title="Document Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
    }
    .highlight {
        background-color: #ffd9b3;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .document-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
    }
    .document-path {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .score {
        color: #28a745;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .snippet {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border: 1px solid #eee;
        font-family: 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'index_builder' not in st.session_state:
    st.session_state.index_builder = None
if 'query_processor' not in st.session_state:
    st.session_state.query_processor = None

def initialize_system():
    """Initialize the document retrieval system."""
    if st.session_state.index_builder is None:
        with st.spinner("Initializing document retrieval system..."):
            # Initialize the index builder
            docs_dir = "documents"
            index_builder = IndexBuilder(docs_dir)
            
            # Check if index exists
            if os.path.exists('index.json'):
                st.info("Loading existing index...")
                # Load documents to get document contents
                index_builder.load_documents()
            else:
                st.info("Building new index...")
                # Load documents and build index
                index_builder.load_documents()
                index_builder.build_index()
                # Save index for future use
                index_builder.save_index()
            
            # Initialize the query processor
            query_processor = QueryProcessor(docs_dir, index_builder.get_document_contents())
            query_processor.load_index()
            
            # Store in session state
            st.session_state.index_builder = index_builder
            st.session_state.query_processor = query_processor

def highlight_terms(text: str, terms: set) -> str:
    """Highlight search terms in the text."""
    highlighted_text = text
    # Sort terms by length (longest first) to avoid partial matches
    sorted_terms = sorted(terms, key=len, reverse=True)
    
    for term in sorted_terms:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span class="highlight">{term}</span>', highlighted_text)
    return highlighted_text

def main():
    st.title("🔍 Document Search Engine")
    st.markdown("Search through documents using natural language queries with synonym expansion.")
    
    # Initialize the system
    initialize_system()
    
    # Create two columns for search controls
    col1, col2 = st.columns([3, 1])
    
    # Search input in the wider column
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., computer hardware problems",
            key="search_query"
        )
    
    # Number of results slider in the narrower column
    with col2:
        num_results = st.slider(
            "Results to show:",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
    
    if query:
        with st.spinner("Searching..."):
            # Get search results
            results = st.session_state.query_processor.search(
                query,
                top_k=num_results,
                show_snippets=False  # We'll handle display ourselves
            )
            
            if results:
                st.success(f"Found {len(results)} relevant documents")
                
                # Display results
                for doc_path, score in results:
                    # Get document name and relative path
                    doc_name = os.path.basename(doc_path)
                    rel_path = os.path.relpath(doc_path, st.session_state.query_processor.documents_dir)
                    
                    # Get relevant snippets
                    query_terms = st.session_state.query_processor.preprocess_text(query)
                    expanded_terms = set(query_terms)
                    for term in query_terms:
                        expanded_terms.update(st.session_state.query_processor.get_synonyms(term))
                    
                    snippets = st.session_state.query_processor.get_relevant_snippets(
                        doc_path,
                        expanded_terms,
                        max_snippets=3
                    )
                    
                    # Display document card
                    with st.container():
                        st.markdown(f"""
                            <div class="document-card">
                                <h3>{doc_name}</h3>
                                <div class="document-path">Path: {rel_path}</div>
                                <div class="score">Relevance Score: {score:.4f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Display snippets
                        if snippets and snippets[0] != "[Document content not available]":
                            st.markdown("**Relevant Excerpts:**")
                            for i, snippet in enumerate(snippets, 1):
                                highlighted_snippet = highlight_terms(snippet, expanded_terms)
                                st.markdown(f"""
                                    <div class="snippet">
                                        {highlighted_snippet}
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No relevant excerpts found in this document.")
                        
                        # Add a divider between documents
                        st.markdown("---")
            else:
                st.warning("No documents found matching your query.")

if __name__ == "__main__":
    main() 