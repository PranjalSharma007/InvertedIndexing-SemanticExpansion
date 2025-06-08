import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

class IndexBuilder:
    def __init__(self, documents_dir: str):
        self.documents_dir = os.path.abspath(documents_dir)
        self.document_contents: Dict[str, str] = {}
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            tokenizer=self.preprocess_text,
            token_pattern=None  # We'll handle tokenization ourselves
        )
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
        """
        Preprocess text by:
        1. Converting to lowercase
        2. Removing special characters and numbers
        3. Tokenizing
        4. Removing stopwords
        5. Lemmatizing
        6. Removing short tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens, then lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                # Lemmatize
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
        
        return processed_tokens

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
                    # Only add terms that don't contain numbers
                    if not any(c.isdigit() for c in term):
                        self.inverted_index[term][doc_path] = tfidf_value

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

    def get_document_contents(self) -> Dict[str, str]:
        """Return the document contents dictionary."""
        return self.document_contents 