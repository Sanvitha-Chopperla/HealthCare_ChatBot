# from langchain.prompts import PromptTemplate

# template = """
# You are an AI Healthcare Assistant specializing in Sea Buckthorn.
# Answer the user's question only from the given context.

# If the answer is not available in the context,
# say:
# 'I do not have enough medical information.'

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

# PROMPT = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )

from langchain.prompts import PromptTemplate

template = """
You are an AI Healthcare Assistant specializing in Sea Buckthorn.
Answer the user's question only from the given context.

If the answer is not available in the context,
say:
'I do not have enough medical information.'

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)