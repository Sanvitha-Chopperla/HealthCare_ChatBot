from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy


def evaluate_rag(question, answer, contexts):
    """
    Lightweight RAG evaluation for real-time chatbots.
    No ground-truth required.
    """

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    })

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ]
    )

    return result