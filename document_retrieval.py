import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

class DocumentRetrieval:
    def __init__(self, documents_dir: str):
        self.documents_dir = os.path.abspath(documents_dir)
        self.document_contents: Dict[str, str] = {}
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def normalize_path(self, path: str) -> str:
        """Normalize a path to ensure consistent format."""
        return os.path.abspath(path)
        
    def load_documents(self) -> None:
        """Load all documents from the documents directory."""
        print("Loading documents...")
        for root, dirs, files in tqdm(list(os.walk(self.documents_dir))):
            for file in files:
                if file.startswith('.'): # Skip hidden files
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        normalized_path = self.normalize_path(file_path)
                        self.document_contents[normalized_path] = content
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        return tokens

    def build_index(self) -> None:
        """Build TF-IDF weighted inverted index."""
        print("Building TF-IDF index...")
        
        # Fit TF-IDF vectorizer
        documents = list(self.document_contents.values())
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Build inverted index with TF-IDF weights
        for doc_idx, doc_path in enumerate(self.document_contents.keys()):
            doc_vector = tfidf_matrix[doc_idx].toarray()[0]
            for term_idx, tfidf_value in enumerate(doc_vector):
                if tfidf_value > 0:
                    term = feature_names[term_idx]
                    self.inverted_index[term][doc_path] = tfidf_value

    def get_synonyms(self, word: str) -> Set[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
        return synonyms

    def get_relevant_snippets(self, doc_path: str, query_terms: Set[str], max_snippets: int = 3) -> List[str]:
        """Extract relevant snippets from a document that contain query terms."""
        try:
            content = self.document_contents[doc_path]
        except KeyError:
            print(f"\nWarning: Document not found in document_contents: {doc_path}")
            print("Available documents:")
            for path in list(self.document_contents.keys())[:5]:  # Show first 5 available documents
                print(f"  - {path}")
            return ["[Document content not available]"]
            
        sentences = sent_tokenize(content)
        relevant_snippets = []
        
        for sentence in sentences:
            # Check if any query term is in the sentence
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                # Highlight the matching terms
                highlighted_sentence = sentence
                for term in query_terms:
                    if term in sentence_lower:
                        highlighted_sentence = highlighted_sentence.replace(
                            term, f"\033[1;31m{term}\033[0m"
                        )
                relevant_snippets.append(highlighted_sentence)
                if len(relevant_snippets) >= max_snippets:
                    break
        
        return relevant_snippets

    def search(self, query: str, top_k: int = 10, show_snippets: bool = True) -> List[Tuple[str, float]]:
        """
        Search for documents matching the query with synonym expansion.
        Returns top_k documents sorted by relevance score.
        """
        query_terms = self.preprocess_text(query)
        expanded_terms = set()
        
        # Expand query terms with synonyms
        for term in query_terms:
            expanded_terms.add(term)
            expanded_terms.update(self.get_synonyms(term))
        
        # Calculate document scores
        doc_scores = defaultdict(float)
        for term in expanded_terms:
            if term in self.inverted_index:
                for doc_path, tfidf_score in self.inverted_index[term].items():
                    doc_scores[doc_path] += tfidf_score
        
        # Sort documents by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        if show_snippets:
            print("\nRelevant document snippets:")
            for doc_path, score in sorted_docs[:top_k]:
                print(f"\nDocument: {os.path.relpath(doc_path, self.documents_dir)} (score: {score:.4f})")
                snippets = self.get_relevant_snippets(doc_path, expanded_terms)
                for snippet in snippets:
                    print(f"  - {snippet}")
        
        return sorted_docs[:top_k]

    def save_index(self, index_file: str = 'index.json') -> None:
        """Save the inverted index to a JSON file."""
        print("Saving index...")
        # Convert absolute paths to relative paths before saving
        relative_index = {}
        for term, docs in self.inverted_index.items():
            relative_index[term] = {
                os.path.relpath(doc_path, self.documents_dir): score 
                for doc_path, score in docs.items()
            }
        with open(index_file, 'w') as f:
            json.dump(relative_index, f)

    def load_index(self, index_file: str = 'index.json') -> None:
        """Load the inverted index from a JSON file."""
        print("Loading index...")
        with open(index_file, 'r') as f:
            relative_index = json.load(f)
            # Convert relative paths back to absolute paths
            self.inverted_index = defaultdict(dict)
            for term, docs in relative_index.items():
                for rel_path, score in docs.items():
                    # Remove any leading 'documents/' if it exists
                    if rel_path.startswith('documents/'):
                        rel_path = rel_path[len('documents/'):]
                    abs_path = os.path.join(self.documents_dir, rel_path)
                    self.inverted_index[term][abs_path] = score 