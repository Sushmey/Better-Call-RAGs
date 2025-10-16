#  Better-Call-RAGs

Better-Call-RAGs enables you to query documents using a Large Language Model (LLM) grounded in the content of those documents through a Retrieval-Augmented Generation (RAG) architecture.

Originally designed for querying legal case transcripts from the Cambridge Law Corpus, it can actually be used to query any set of documents — PDF, TXT, XML, or CSV.

---

##  Features

- 🗂️ Supports multiple file formats: .pdf, .txt, .xml, .csv
- ✂️ Recursively splits long texts into context-friendly chunks
- 🔍 Creates sentence embeddings for efficient semantic search
- 🧠 Uses an LLM to retrieve the most relevant document chunks to your query
- ⚖️ Pre-trained on Cambridge Law Corpus for specialized legal document understanding
- ⚡ Extensible — plug in your own dataset or LLM endpoint

---

##  Architecture Overview

           ┌────────────────────┐
           │   Input Documents  │
           └────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Recursive Text Split │
         └────────┬─────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │ Sentence Embeddings  │  ← e.g., OpenAI Embeddings / Sentence Transformers
         └────────┬─────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │ Vector Store Index   │
         └────────┬─────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │    LLM Retriever     │  ← Context-aware Query Response
         └──────────────────────┘

---

## Installation
```
git clone https://github.com/Sushmey/Better-Call-RAGs.git
cd Better-Call-RAGs
pip install -r requirements.txt
```
> Ensure you have a valid API key configured for your chosen LLM provider (e.g., OpenAI, Anthropic, etc.).

---

##  Usage

1. Add your documents to the /data folder.
2. Run the script to build embeddings and create the index:
   python build_index.py
3. Start querying your document base:
   python query.py
4. Example query:
   Enter your query: What are the conditions for breach of contract?

   Response:
   Based on precedent cases, a breach of contract requires evidence of agreement, consideration, and a failure of performance obligations...

---

##  Example Use Cases

- Legal research & case retrieval
- Academic literature search
- Enterprise document Q&A systems
- Personal knowledge bases

---

## Model & Data

- Embeddings: SentenceTransformers / OpenAI Embeddings
- Vector Store: FAISS / Chroma
- Corpus: Cambridge Law Corpus (for demonstration)
- LLM: Configurable (e.g., GPT-4, Claude, Llama)

---

##  Future Work

- Web-based interface for interactive querying
- Multi-turn conversation memory
- Fine-tuning on domain-specific datasets

---

##  Contributing

Contributions are welcome!
If you’d like to add new file types, improve the retriever, or extend RAG functionality, feel free to open a pull request.

---

##  License

This project is licensed under the MIT License.

---

###  Author

Sushmey Nirmal
📧 sushmey@gmail.com
🔗 https://linkedin.com/in/Sushmey
