"""
Document Manager

Responsible for:
- Saving uploaded documents
- Listing available documents
- Deleting documents
- Triggering indexing through IngestionService
"""

from pathlib import Path

from ingestion_service import IngestionService

BASE_DATA_DIR = Path(__file__).parent.parent / "data"


class DocumentManager:

    def __init__(self, username):

        self.username = username

        self.data_dir = BASE_DATA_DIR / username

        self.data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.ingestion_service = IngestionService(username)

    def save_uploaded_file(self, uploaded_file):
        """
        Save uploaded Streamlit file into the data directory.
        """

        file_path = self.data_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    def list_documents(self):
        """
        Return all uploaded documents.
        """

        return sorted(
            [
                file
                for file in self.data_dir.iterdir()
                if file.is_file()
            ]
        )

    def delete_document(self, filename):
        """
        Delete a document.
        """

        file_path = self.data_dir / filename

        if file_path.exists():
            file_path.unlink()
            return True

        return False

    def ingest_documents(
        self,
        config_name="default",
        chunk_size=512,
        chunk_overlap=64,
    ):
        """
        Index all documents using the IngestionService.
        """

        try:

            document_count, chunk_count = self.ingestion_service.ingest(
                config_name=config_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            return (
                True,
                f"Successfully indexed {document_count} documents into {chunk_count} chunks.",
            )

        except Exception as e:

            return False, str(e)