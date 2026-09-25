import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from loaders import NotebookCellLoader, load_source, source_label
from langchain_text_splitters import RecursiveCharacterTextSplitter


class NotebookLoaderTests(unittest.TestCase):
    def test_uploaded_file_names_and_csv_rows(self):
        with TemporaryDirectory() as directory:
            # Chainlit temporary paths need not have the original extension.
            path = Path(directory) / 'upload-id'
            path.write_text('name,role\nMaya,maintainer\nJordan,reviewer\n', encoding='utf-8')
            docs = load_source(str(path), 'roster.csv')
            self.assertEqual(len(docs), 2)
            self.assertEqual(source_label(docs[1]), 'roster.csv, row 2')
            self.assertIn('Jordan', docs[1].page_content)
            path.write_text('A source paragraph.', encoding='utf-8')
            self.assertEqual(load_source(str(path), 'notes.md')[0].metadata['source'],
                             'notes.md')
            with self.assertRaises(ValueError):
                load_source(str(path), 'unsupported.exe')

    def test_cells_preserve_provenance_and_ignore_outputs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.ipynb'
            path.write_text(json.dumps({'nbformat': 4, 'cells': [
                {'cell_type': 'markdown', 'source': ['A notebook ', 'explains RAG.']},
                {'cell_type': 'code', 'source': 'print("hello")',
                 'outputs': [{'text': 'OUTPUT MUST NOT BE INDEXED'}]},
                {'cell_type': 'markdown', 'source': []},
                {'cell_type': 'raw', 'source': 'ignored'},
            ]}), encoding='utf-8')
            docs = NotebookCellLoader(str(path)).load()
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0].page_content, 'A notebook explains RAG.')
            self.assertEqual(source_label(docs[1]), 'notes.ipynb, cell 2')
            self.assertNotIn('OUTPUT', '\n'.join(d.page_content for d in docs))
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=10, chunk_overlap=2).split_documents(docs)
            self.assertTrue(chunks)
            self.assertTrue(all(c.metadata['source'] == 'notes.ipynb' for c in chunks))
            self.assertEqual({c.metadata['cell'] for c in chunks}, {1, 2})

    def test_rejects_invalid_structure(self):
        for notebook in ({'nbformat': 3}, {'nbformat': 4, 'cells': 'bad'},
                         {'nbformat': 4, 'cells': [{'cell_type': 'code', 'source': 7}]}):
            with self.subTest(notebook=notebook), TemporaryDirectory() as directory:
                path = Path(directory) / 'bad.ipynb'
                path.write_text(json.dumps(notebook), encoding='utf-8')
                with self.assertRaises(ValueError):
                    NotebookCellLoader(str(path)).load()


if __name__ == '__main__':
    unittest.main()
