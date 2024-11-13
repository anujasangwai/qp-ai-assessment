from stores import VectorStoreFactory
from embeddings import EmbeddingsFactory

from config import QAConfig, QAResponse, Document
from llm import LLMFactory

from processors import DocumentProcessor

from langchain.chains import RetrievalQA
from prompt_templates import QAPromptTemplate

class QASystem:
    def __init__(self, config: QAConfig):
        self.config = config
        self.doc_processor = DocumentProcessor(config)
        self.embeddings = EmbeddingsFactory.create_embeddings(config)
        self.vector_store = VectorStoreFactory.create_vector_store(config, self.embeddings)
        self.llm = LLMFactory.create_llm(config)
        self.qa_prompt = QAPromptTemplate().get_prompt(config.prompt_template)
    

    def initialize(self, file_path: str = None):
        if self.vector_store.load_existing():
            print("Loaded existing vector store")
            return

        if not file_path:
            raise ValueError("No existing vector store found and no file path provided")

        print("Processing new documents")
        documents = self.doc_processor.load_documents(file_path)
        chunks = self.doc_processor.split_documents(documents)
        self.vector_store.store_documents(chunks)

    def get_qa_chain(self):
        retriever = self.vector_store.get_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )

    def query(self, question: str) -> QAResponse:
        qa_chain = self.get_qa_chain()
        result = qa_chain.invoke(question)
        
        return QAResponse(
            answer=result['result'],
            source_documents=[
                Document(doc.page_content, doc.metadata)
                for doc in result['source_documents']
            ],
            metadata={"query": question}
        )