# Document Retrieval System

A lightweight document retrieval system that uses TF-IDF weighted scoring and WordNet-based synonym expansion for improved search results.

## Features

- TF-IDF weighted document scoring
- Inverted index for efficient retrieval
- WordNet-based synonym expansion for better recall
- Persistent index storage in JSON format
- Text preprocessing (tokenization, stopword removal, lemmatization)

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure your documents are in the `documents` directory.

## Usage

Run the main script to index documents and perform example searches:

```bash
python3 main.py
```

The system will:
1. Load documents from the `documents` directory
2. Build a TF-IDF weighted inverted index
3. Save the index to `index.json` for future use
4. Perform example searches with synonym expansion

## How it Works

1. **Document Loading**: The system recursively loads all text documents from the specified directory.

2. **Indexing**:
   - Documents are preprocessed (tokenization, stopword removal, lemmatization)
   - TF-IDF weights are calculated for terms
   - An inverted index is built mapping terms to documents with their TF-IDF scores

3. **Search**:
   - Query terms are preprocessed
   - Synonyms are found using WordNet
   - Documents are scored based on matching terms and synonyms
   - Results are ranked by relevance score

## Customization

You can modify the search behavior by adjusting parameters in the `DocumentRetrieval` class:
- Change the number of results with the `top_k` parameter in the `search` method
- Modify text preprocessing in the `preprocess_text` method
- Adjust synonym expansion in the `get_synonyms` method 