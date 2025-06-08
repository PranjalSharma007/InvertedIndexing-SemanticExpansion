from index_builder import IndexBuilder
from query_processor import QueryProcessor
import os

def main():
    # Initialize the index builder
    docs_dir = "documents"
    index_builder = IndexBuilder(docs_dir)
    
    # Check if index exists
    if os.path.exists('index.json'):
        print("Loading existing index...")
        # Load documents to get document contents
        index_builder.load_documents()
    else:
        print("Building new index...")
        # Load documents and build index
        index_builder.load_documents()
        index_builder.build_index()
        # Save index for future use
        index_builder.save_index()
    
    # Initialize the query processor with document contents
    query_processor = QueryProcessor(docs_dir, index_builder.get_document_contents())
    query_processor.load_index()
    
    # Example searches
    queries = [
        "computer hardware problems",
        "space exploration mars",
        "car engine repair",
        "medical treatment",
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Search query: {query}")
        print(f"{'='*80}")
        results = query_processor.search(query, top_k=3, show_snippets=True)
        print(f"\nFound {len(results)} relevant documents")

if __name__ == "__main__":
    main() 