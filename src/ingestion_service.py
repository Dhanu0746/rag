from ingest import (
    load_documents,
    chunk_documents,
    build_dense_index,
    build_sparse_index,
)


class IngestionService:

    def __init__(self, username):
        self.username = username

    def ingest(
        self,
        config_name="default",
        chunk_size=512,
        chunk_overlap=64,
    ):

        documents = load_documents(self.username)

        nodes = chunk_documents(
            documents,
            chunk_size,
            chunk_overlap,
        )

        build_dense_index(
            nodes,
            config_name=self.username,
        )

        build_sparse_index(
          nodes,
          config_name=self.username,
        )

        return len(documents), len(nodes)