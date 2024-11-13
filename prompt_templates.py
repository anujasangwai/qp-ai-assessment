from langchain.prompts import PromptTemplate

class QAPromptTemplate:
    def __init__(self) -> None:
        self.prompts = dict()

        quantum_analyst_template =  PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert in quantum mechanics and have a good understanding of quantum algorithms

            Use the following pieces of context to answer the question at the end. 
            Note, you should only use the following context to generate an accurate answer.
            {context}

            Question: {question}
            Always see if the quetion is related to context, if it is not, DO NOT GENERATE any answer.
            If the question is not related to the context, do not generate any answer and just repond with "I don't know"
            """
        )
        self.prompts["quantum_analyst"] = quantum_analyst_template
    
    def get_prompt(self, key):
        return self.prompts.get(key, "")