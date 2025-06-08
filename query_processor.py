import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class QueryProcessor:
    def __init__(self, documents_dir: str, document_contents: Dict[str, str]):
        self.documents_dir = os.path.abspath(documents_dir)
        self.document_contents = document_contents
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

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

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        return tokens

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
            expanded_terms.add(term)  # Original term
            expanded_terms.update(self.get_synonyms(term))  # Synonyms
        
        # Calculate document scores with different weights for exact matches and synonyms
        doc_scores = defaultdict(float)
        for term in expanded_terms:
            if term in self.inverted_index:
                for doc_path, tfidf_score in self.inverted_index[term].items():
                    # Weight exact matches higher than synonyms
                    weight = 1.0 if term in query_terms else 0.5  # Exact matches get full weight, synonyms get half
                    doc_scores[doc_path] += tfidf_score * weight
        
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