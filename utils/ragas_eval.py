from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
import os


def evaluate_rag(question, answer, contexts):

    # ✅ FORCE GROQ AS JUDGE LLM (THIS IS THE KEY FIX)
    llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
    )

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm
    )

    return result