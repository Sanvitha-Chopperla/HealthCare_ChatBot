# test.py
# from utils.chatbot import load_chatbot

# chain = load_chatbot()
# answer = chain.invoke("What is Sea Buckthorn?")
# print(answer)

from utils.chatbot import load_chatbot

chain = load_chatbot()
answer = chain.invoke("What is Sea Buckthorn?")
print(answer)