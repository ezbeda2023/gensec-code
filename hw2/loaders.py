"""Source loaders, including a custom cell-aware Jupyter notebook loader."""

import json
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class NotebookCellLoader(BaseLoader):
    """Read notebook v4 source cells; never execute code or ingest outputs."""

    def __init__(self, path: str, source: str | None = None):
        self.path = Path(path)
        self.source = source or self.path.name

    def lazy_load(self) -> Iterator[Document]:
        notebook = json.loads(self.path.read_text(encoding='utf-8-sig'))
        if not isinstance(notebook, dict) or notebook.get('nbformat') != 4:
            raise ValueError('Expected a Jupyter notebook in nbformat 4.')
        cells = notebook.get('cells')
        if not isinstance(cells, list):
            raise ValueError('Notebook cells must be a list.')
        for index, cell in enumerate(cells, 1):
            if not isinstance(cell, dict):
                raise ValueError(f'Invalid notebook cell {index}.')
            kind = cell.get('cell_type')
            if kind not in {'markdown', 'code'}:
                continue
            source = cell.get('source', '')
            if isinstance(source, list) and all(isinstance(s, str) for s in source):
                source = ''.join(source)
            if not isinstance(source, str):
                raise ValueError(f'Cell {index} source must be text.')
            if source.strip():
                yield Document(page_content=source, metadata={
                    'source': self.source, 'cell': index, 'cell_type': kind,
                })


def load_source(path: str, name: str) -> list[Document]:
    # Import integrations lazily so custom-loader tests need only langchain-core.
    from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader

    suffix = Path(name).suffix.lower()
    if suffix == '.ipynb':
        loader = NotebookCellLoader(path, name)
    elif suffix == '.pdf':
        loader = PyPDFLoader(path)
    elif suffix == '.csv':
        loader = CSVLoader(path, encoding='utf-8-sig')
    elif suffix in {'.txt', '.md'}:
        loader = TextLoader(path, encoding='utf-8-sig')
    else:
        raise ValueError(f'Unsupported file type: {suffix}')
    docs = loader.load()
    for doc in docs:
        doc.metadata['source'] = name
    return [doc for doc in docs if doc.page_content.strip()]


def source_label(doc: Document) -> str:
    metadata = doc.metadata
    label = metadata.get('source', 'Unknown source')
    for key in ('cell', 'page', 'row'):
        if key in metadata:
            number = metadata[key] + (0 if key == 'cell' else 1)
            label += f', {key} {number}'
    return label
