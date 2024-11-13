from dotenv import load_dotenv
import os

from qa_system import QASystem
from config import QAConfig

def main():
    load_dotenv()
    
    config = QAConfig(
        vector_store_type="chroma",
        llm_type="openai",
        llm_config={"openai_api_key": os.getenv("OPENAI_API_KEY"), "temperature": 0.8, "model": "gpt-4o-mini"},
        embeddings_type="openai",
        embeddings_config={"openai_api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    qa_system = QASystem(config)
    qa_system.initialize()
    
    while True:
        question = "Enter query (or 'quit' to exit): "
        dashes = "="*len(question)

        query = question + "\n" + dashes + "\nQuestion: "
        query = input(query)
        if query.lower() == 'quit':
            break
            
        response = qa_system.query(query)
        
        print(f"\nAnswer: {response.answer}\n")
        # print("Sources:")
        # for doc in response.source_documents:
        #     print(f"- Content: {doc.content}")
        #     print(f"- Metadata: {doc.metadata}\n")

if __name__ == "__main__":
    main()