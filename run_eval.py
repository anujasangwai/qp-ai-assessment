from qa_system import QASystem
from config import QAConfig
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    AnswerRelevancy,
    Faithfulness,
)

from datasets import Dataset
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders.pdf import PyPDFLoader

from typing import List, Dict, Any, Optional
    
from langchain.document_loaders import PyPDFLoader
from typing import List, Dict, Any, Optional
import pandas as pd

class QAEvaluator:
    def __init__(self, qa_system, llm):
        self.qa_system = qa_system
        self.dataset_generator = DatasetGenerator(llm)
        self.metrics = {
            'context_recall': ContextRecall(),
            'context_precision': ContextPrecision(),
            'answer_relevancy': AnswerRelevancy(),
            'faithfulness': Faithfulness()
        }


class QAEvaluator:
    def __init__(self, qa_system, llm, document_path, eval_config):
        self.qa_system = qa_system
        self.dataset_generator = DatasetGenerator(llm)
        self.document_path = document_path
        self.eval_config = eval_config
        self.metrics = {
            'context_recall': ContextRecall(),
            'context_precision': ContextPrecision(),
            'answer_relevancy': AnswerRelevancy(),
            'faithfulness': Faithfulness()
        }

    def extract_text_from_pdf(self) -> List[Dict[str, str]]:
        """Extract text and metadata from a PDF file using LangChain PyPDFLoader."""
        loader = PyPDFLoader(self.document_path)
        pages = loader.load_and_split()
        documents = [
            {
                "content": page.page_content,
                "metadata": {"source": self.document_path, "page": page.metadata["page"]}
            }
            for page in pages
        ]
        return documents

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate QA system using PDF documents.
        
        Args:
            pdf_paths: List of PDF file paths
            num_questions: Number of questions to generate
            
        Returns:
            Dictionary containing evaluation scores and test questions used
        """
        all_documents = []
        all_documents.extend(self.extract_text_from_pdf(self.document_path))

        test_questions = self.dataset_generator.generate_questions_from_context(all_documents, self.eval_config.number_of_questions)
        print(test_questions)
        dataset = self.prepare_evaluation_dataset(test_questions)
        print(dataset)
        results = evaluate(
            dataset=dataset,
            metrics=list(self.metrics.values())
        )

        print("\n=== QA System Evaluation Summary ===")
        print("\nMetrics Scores:")
        df = results.to_pandas()
        for f in df:
            print(f)
        df.to_csv('eval_output.csv', index = False)
        df.to_json('output.json', orient='records', indent=2)
            
        print("\nScore interpretation guide:")
        print("- Context Recall: Measures how well the system retrieves relevant context")
        print("- Context Precision: Measures how precise the retrieved context is")
        print("- Answer Relevancy: Measures how relevant the answer is to the question")
        print("- Faithfulness: Measures how faithful the answer is to the provided context")

        return {
            'metrics': results,
            'test_questions': test_questions
        }

    def prepare_evaluation_dataset(self, test_questions: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare dataset for Ragas evaluation.
        
        Args:
            test_questions: List of dicts containing test questions and ground truth
                          Format: [{'question': str, 'ground_truth': str}]
        """
        evaluated_data = []
        
        for item in test_questions:
            question = item['question']
            response = self.qa_system.query(question)
            
            evaluated_data.append({
                'question': question,
                'answer': response.answer,
                'contexts': [doc.content for doc in response.source_documents],
                'ground_truth': item['ground_truth']
            })
            
        return Dataset.from_pandas(pd.DataFrame(evaluated_data))



    def print_summary(self, evaluation_results: Dict[str, Any]):
        """
        Print a formatted summary of evaluation results.
        """
        print("\n=== QA System Evaluation Summary ===")
        print("\nMetrics Scores:")
        for metric, score in evaluation_results['metrics'].items():
            print(f"{metric:20}: {score:.3f}")
            
        print("\nTest Questions Used:")
        for i, question in enumerate(evaluation_results['test_questions'], 1):
            print(f"\n{i}. Question: {question['question']}")
            print(f"   Ground Truth: {question['ground_truth']}")
            
        print("\nScore interpretation guide:")
        print("- Context Recall: Measures how well the system retrieves relevant context")
        print("- Context Precision: Measures how precise the retrieved context is")
        print("- Answer Relevancy: Measures how relevant the answer is to the question")
        print("- Faithfulness: Measures how faithful the answer is to the provided context")


class DatasetGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.question_generation_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Given the following context, generate {number_of_questions} diverse questions that can be answered using this information. 
            For each question, also provide the ground truth answer based strictly on the given context.
            Format your response as a JSON array with objects containing 'question' and 'ground_truth' keys.

            Context:
            {context}

            Return only the JSON array without any additional text. Example format:
            [
                {{"question": "What is X?", "ground_truth": "X is Y"}},
                {{"question": "How does Z work?", "ground_truth": "Z works by..."}}
            ]
            """)
        
        self.question_chain = LLMChain(
            llm=self.llm,
            prompt=self.question_generation_prompt
        )

    def generate_questions_from_context(self, context: str, number_of_questions: int) -> List[Dict[str, str]]:
        """Generate questions and ground truth answers from a given context."""
        try:
            result = self.question_chain.invoke({"context": context, "number_of_questions": number_of_questions})
            questions = json.loads(result['text'].replace('\n','').replace('json','').replace('```',''))
            print(questions)
            return questions
        except json.JSONDecodeError:
            print("Error: Failed to parse LLM output as JSON")
            return []
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []


config = QAConfig(
    vector_store_type="chroma",
    llm_type="openai",
    llm_config={"openai_api_key": os.getenv("OPENAI_API_KEY"), "temperature": 0.8, "model": "gpt-4o-mini"},
    embeddings_type="openai",
    embeddings_config={"openai_api_key": os.getenv("OPENAI_API_KEY")})
qa_system = QASystem(config)
qa_system.initialize(file_path="/Users/rsaraf/Documents/agoge-main/agoge/demos/qp/source_data.pdf")
llm = ChatOpenAI(temperature=0.8, model="gpt-4o")


evaluator = QAEvaluator(qa_system, llm)
paths = []
paths.append(os.path.join(os.getcwd(), "source_data.pdf"))
results = evaluator.evaluate(paths, num_questions=5)
print(results)

