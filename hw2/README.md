# HW2: Source Notebook

A local NotebookLM-like question-answering app built with LangChain and Chainlit.

## Relationship to the required lab

Adapted from [the course RAG directory](https://github.com/icardei/gensec-code/tree/main/02_LangChain/07_RAG):

- `07_rag_loaddb.py`: document loaders, `RecursiveCharacterTextSplitter`, Vertex AI embeddings, and Chroma.
- `09_rag_query.py`: retrieved context -> local prompt -> Gemini -> `StrOutputParser`.
- `10_chainlit_rag_query.py`: Chainlit chat lifecycle and messages.

The extension is `NotebookCellLoader` in `loaders.py`, which extends LangChain's
`BaseLoader` and implements `lazy_load()`. Each nonempty Markdown or code cell
becomes a document with filename, original one-based cell number, and cell type.
Notebook code is never executed. Outputs, attachments, and raw cells are excluded.
This lets students ask questions about lab notebooks and trace evidence to cells.
See the [LangChain loader interface](https://docs.langchain.com/oss/python/integrations/document_loaders).

The app also adds multi-file uploads, numbered evidence excerpts, and per-chat
collections. PDF, CSV, TXT, and Markdown use built-in loaders; these formats
already appear in the lab, so they are not claimed as the novel extension.
This is standard single-query RAG, not Multi-RAG.

## Setup (PowerShell, Python 3.11+)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --no-compile -r hw2/requirements.txt
Copy-Item hw2/.env.example hw2/.env
```

Edit `hw2/.env`: set your Google AI Studio key, an accessible Gemini chat model
ID, and your course Google Cloud project. As in the current lab, embeddings use
Vertex AI, so enable Vertex AI for that project and configure Application Default
Credentials with the Google Cloud CLI:

```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Your account needs permission to call Vertex AI and sufficient quota. AI Studio
chat and Vertex AI embeddings use separate credentials/billing arrangements.
Do not commit `.env` or credentials. Source text is sent to Google services.

```powershell
cd hw2
..\.venv\Scripts\python app.py
```

Open the local URL printed in the terminal. A "Please wait" screen appears while
the app loads and automatically switches to Chainlit when ready. This startup
screen is provided by the direct Python launcher.
Upload up to 10 files, each at most 10 MB.
Use `/upload` to replace sources. A fresh chat starts an empty notebook.
Collections are held locally without persistence and deleted when the chat ends;
this is not an authenticated or durable notebook service. Chainlit manages its
own temporary upload files. Each question is independent (no conversation memory).

## Demo and verification

Upload both files from `examples/`, then ask:

1. "What chunk size and overlap does the study notebook use?" (1200 and 200.)
2. "When and where does the Cedar study group meet?" (Tuesday, 4 PM, room 215.)
3. Open a Source link and show the filename and cell number or text excerpt.
4. "What is Maya's phone number?" (The sources do not contain this answer.)
5. Show `NotebookCellLoader.lazy_load()` and explain metadata surviving splitting.

Record your actual demo and put its URL in `screencast_url.txt` if required by
your submission instructions. That file is intentionally not populated here.

Run offline loader checks from the repository root:

```powershell
.\.venv\Scripts\python -m unittest discover -s hw2 -p "test_*.py" -v
```

Checks cover notebook validation, output exclusion, cell numbering, and citation
metadata through chunking. A live demo additionally requires your credentials.
Scanned PDFs need OCR (not included). Similarity retrieval always returns nearby
chunks, so irrelevant retrieval and incorrect model answers remain possible.
Inspect citations rather than assuming they prove an answer. The prompt requests
source-only answers and abstention, but these behaviors are not guarantees.

AI assistance was used to inspect the course starter, implement the extension,
and add checks and documentation. Review the code and reproduce the demo before
submission so you can explain the pipeline and its limitations.
