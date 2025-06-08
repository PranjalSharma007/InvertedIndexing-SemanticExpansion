```mermaid
graph TD
    UI[Streamlit Interface] -->|Query| QP[Query Processor]
    QP -->|Results| UI
    
    QP -->|Read Index| II[Inverted Index]
    QP -->|Fetch Content| DC[Document Contents]
    
    IB[Index Builder] -->|Build| II
    IB -->|Store| DC
    
    DOCS[Document Files] -->|Read| IB
    II -->|Save| JSON[index.json]
    JSON -->|Load| II

    QP -->|Use| WN[WordNet]
    QP -->|Use| TFIDF[TF-IDF Scores]
```

# Document Retrieval System Architecture

## Components

### User Interface
- **Streamlit Interface**: Web-based search interface

### Core System
- **Index Builder**: Processes documents and creates index
- **Query Processor**: Handles search queries
- **Inverted Index**: Maps terms to documents
- **Document Contents**: Stores processed documents

### Storage
- **Document Files**: Raw document corpus
- **index.json**: Persistent index storage

### Features
- **WordNet**: Synonym expansion
- **TF-IDF**: Term importance scoring