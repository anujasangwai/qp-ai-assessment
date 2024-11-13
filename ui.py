import os
import chainlit as cl
from typing import Optional
from qa_system import QASystem, QAConfig
from dotenv import load_dotenv



class ChainlitQASession:
    def __init__(self):
        self.qa_system: Optional[QASystem] = None
        self.config = QAConfig(
            vector_store_type="chroma",
            llm_type="openai",
            llm_config={"openai_api_key": os.getenv("OPENAI_API_KEY"), "temperature": 0.8, "model": "gpt-4o-mini"},
            embeddings_type="openai",
            embeddings_config={"openai_api_key": os.getenv("OPENAI_API_KEY")}
        )

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Welcome! Please upload a PDF document to begin. I'll analyze it and answer your questions.",
        author="System"
    ).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    try:
        session = ChainlitQASession()
        cl.user_session.set("session", session)

        processing_msg = cl.Message(content="Processing your document... Please wait.")
        await processing_msg.send()

        session.config.vector_store_path = "db_" + file.name
        session.qa_system = QASystem(session.config)
        session.qa_system.initialize(file.path)

        await cl.Message(content=f"Document processed successfully! You can now ask questions about {file.name}").send()
        await cl.Message(
            content="You can ask questions like:\n" +
                    "- What are the main topics covered in this document?\n" +
                    "- Can you summarize the key points?\n" +
                    "- What are the main conclusions?",
            author="System"
        ).send()
    except Exception as e:
            await cl.Message(content=f"Error processing document: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message):
    session: ChainlitQASession = cl.user_session.get("session")
    if session.qa_system is None:
        await cl.Message(
            content="Please upload a PDF document first before asking questions.",
            author="System"
        ).send()
        return

    msg = cl.Message(content="Thinking...")
    await msg.send()

    try:
        response = session.qa_system.query(message.content)
        elements = []
        found_sources = []

        if response.answer != "I don't know":
            for i, source in enumerate(response.source_documents):
                src_name = f"Source {i+1}"
                found_sources.append(src_name)

                elements.append(
                    cl.Text(
                        name=src_name,
                        content=f"{source.content}",
                        display="side"
                    )
                )
        
        answer = response.answer
        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"
        
        await cl.Message(content=answer, elements=elements).send()

    except Exception as e:
        raise e

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)