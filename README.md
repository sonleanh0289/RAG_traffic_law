# Traffic Law RAG Starter

This workspace includes a Python indexing pipeline for chunked traffic-law text files and Qdrant.

## What it does

- Reads all `.txt` chunk files from the configured chunk directory
- Loads text as UTF-8 so Vietnamese content stays correct
- Extracts basic metadata from the chunk header lines
- Creates embeddings with either OpenAI or `sentence-transformers`
- Upserts vectors and payloads into a Qdrant collection

## Setup

1. Create `.env` from `.env.example` and fill in your values.
2. Start Qdrant if it is not already running.
3. Use your local environment interpreter:

```powershell
E:\RAG\.venv\Scripts\python.exe
```

## Index chunks

```powershell
E:\RAG\.venv\Scripts\python.exe E:\RAG\scripts\index_qdrant.py
```

## Search test

```powershell
E:\RAG\.venv\Scripts\python.exe E:\RAG\scripts\search_qdrant.py "muc phat vuot den do"
```

## Embedding options

- `EMBEDDING_PROVIDER=openai`
- `EMBEDDING_PROVIDER=sentence_transformers`

For local multilingual embeddings, a good starting model is:

- `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2`

## Stored payload fields

- `chunk_id`
- `file_name`
- `source_path`
- `title`
- `context`
- `content`
- `chapter`
- `article`
- `clause`

## Next recommended step

After indexing works, add an answer layer that:

- retrieves top chunks from Qdrant
- sends only those chunks to the LLM
- forces citation output from `article`, `clause`, and `source_path`
