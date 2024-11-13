import os
from langchain_community.vectorstores import FAISS
from langchain_chroma.vectorstores import Chroma

class BaseVectorStore:
    def __init__(self, config, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.store = None

    def load_existing(self) -> bool:
        raise NotImplementedError

    def store_documents(self, chunks):
        raise NotImplementedError

    def get_retriever(self, **kwargs):
        raise NotImplementedError

class FAISSVectorStore(BaseVectorStore):
    def load_existing(self) -> bool:
        try:
            self.store = FAISS.load_local(
                self.config.vector_store_path,
                self.embeddings,
                index_name="index"
            )
            return True
        except:
            return False

    def store_documents(self, chunks):
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        self.store = FAISS.from_texts(texts, self.embeddings, metadatas)
        self.store.save_local(self.config.vector_store_path, index_name="index")

    def get_retriever(self, **kwargs):
        return self.store.as_retriever(**kwargs)

class ChromaVectorStore(BaseVectorStore):
    def load_existing(self) -> bool:
        if os.path.exists(self.config.vector_store_path):
            self.store = Chroma(
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings
            )
            return True
        return False


    def store_documents(self, chunks):
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        self.store = Chroma.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas,
            persist_directory=self.config.vector_store_path
        )

    def get_retriever(self, **kwargs):
        return self.store.as_retriever(**kwargs)

class VectorStoreFactory:
    @staticmethod
    def create_vector_store(config, embeddings):
        if config.vector_store_type == "faiss":
            return FAISSVectorStore(config, embeddings)
        elif config.vector_store_type == "chroma":
            return ChromaVectorStore(config, embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")
