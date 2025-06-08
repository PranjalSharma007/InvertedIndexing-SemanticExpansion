import matplotlib.pyplot as plt
import numpy as np
import time
from index_builder import IndexBuilder
from query_processor import QueryProcessor
import os
from collections import defaultdict
import shutil

# Set the style for all plots
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 10

class SystemEvaluator:
    def __init__(self, docs_dir="documents"):
        self.docs_dir = docs_dir
        self.initialize_system()

    def initialize_system(self):
        """Initialize the system with basic setup."""
        self.index_builder = IndexBuilder(self.docs_dir)
        self.index_builder.load_documents()
        if not os.path.exists('index.json'):
            self.index_builder.build_index()
            self.index_builder.save_index()
        self.query_processor = QueryProcessor(self.docs_dir, self.index_builder.get_document_contents())
        self.query_processor.load_index()

    def measure_index_size_vs_documents(self):
        """Measure how index size grows with number of documents."""
        print("Measuring index size vs number of documents...")
        doc_counts = []
        index_sizes = []
        
        # Create temporary directories with increasing document counts
        for i in range(1, 6):
            temp_dir = f"temp_docs_{i}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy a subset of documents
            doc_count = i * 10
            for j, doc_path in enumerate(list(self.index_builder.document_contents.keys())[:doc_count]):
                shutil.copy(doc_path, os.path.join(temp_dir, os.path.basename(doc_path)))
            
            # Build index and measure size
            temp_builder = IndexBuilder(temp_dir)
            temp_builder.load_documents()
            temp_builder.build_index()
            
            doc_counts.append(doc_count)
            index_sizes.append(len(temp_builder.inverted_index))
            
            # Cleanup
            shutil.rmtree(temp_dir)
        
        return doc_counts, index_sizes

    def measure_search_time_vs_query_length(self):
        """Measure search time vs query length."""
        print("Measuring search time vs query length...")
        query_lengths = []
        search_times = []
        
        # Test queries of different lengths
        base_query = "computer hardware"
        for i in range(1, 6):
            query = " ".join([base_query] * i)
            start_time = time.time()
            self.query_processor.search(query, show_snippets=False)
            search_time = time.time() - start_time
            
            query_lengths.append(i)
            search_times.append(search_time)
        
        return query_lengths, search_times

    def measure_recall_with_synonyms(self):
        """Measure recall improvement with synonym expansion."""
        print("Measuring recall with and without synonyms...")
        queries = [
            "computer",
            "hardware",
            "space",
            "medical",
            "car"
        ]
        
        recall_with_synonyms = []
        recall_without_synonyms = []
        
        for query in queries:
            # Search with synonyms
            results_with_synonyms = self.query_processor.search(query, show_snippets=False)
            
            # Search without synonyms (temporarily disable synonym expansion)
            original_get_synonyms = self.query_processor.get_synonyms
            self.query_processor.get_synonyms = lambda x: set()
            results_without_synonyms = self.query_processor.search(query, show_snippets=False)
            self.query_processor.get_synonyms = original_get_synonyms
            
            recall_with_synonyms.append(len(results_with_synonyms))
            recall_without_synonyms.append(len(results_without_synonyms))
        
        return queries, recall_with_synonyms, recall_without_synonyms

    def generate_graphs(self):
        """Generate evaluation graphs."""
        os.makedirs('results', exist_ok=True)
        
        # 1. Index Size vs Number of Documents
        print("\nGenerating Index Size vs Number of Documents graph...")
        doc_counts, index_sizes = self.measure_index_size_vs_documents()
        
        plt.figure()
        plt.plot(doc_counts, index_sizes, 'o-')
        plt.title('Index Size Growth')
        plt.xlabel('Number of Documents')
        plt.ylabel('Unique Terms')
        plt.tight_layout()
        plt.savefig('results/index_size.png')
        plt.close()

        # 2. Search Time vs Query Length
        print("\nGenerating Search Time vs Query Length graph...")
        query_lengths, search_times = self.measure_search_time_vs_query_length()
        
        plt.figure()
        plt.plot(query_lengths, search_times, 's-')
        plt.title('Search Time vs Query Length')
        plt.xlabel('Query Length (terms)')
        plt.ylabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig('results/search_time.png')
        plt.close()

        # 3. Recall with and without Synonyms
        print("\nGenerating Recall Comparison graph...")
        queries, recall_with_synonyms, recall_without_synonyms = self.measure_recall_with_synonyms()
        
        x = np.arange(len(queries))
        width = 0.35
        
        plt.figure()
        plt.bar(x - width/2, recall_without_synonyms, width, label='No Synonyms')
        plt.bar(x + width/2, recall_with_synonyms, width, label='With Synonyms')
        
        plt.title('Recall Comparison')
        plt.xlabel('Query Terms')
        plt.ylabel('Documents Found')
        plt.xticks(x, queries)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/recall.png')
        plt.close()

if __name__ == "__main__":
    evaluator = SystemEvaluator()
    evaluator.generate_graphs() 