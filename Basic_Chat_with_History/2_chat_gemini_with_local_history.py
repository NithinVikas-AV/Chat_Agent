import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from all_common_prompts import *

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=api_key)

chat_history = []

system_message = chat_gemini_with_local_history()
chat_history.append(SystemMessage(content = system_message))

print ('\n\n ---------- Message Starts ----------')

while True:
    query = input('\nYou: ')
    if query == 'exit':
        break

    chat_history.append(HumanMessage(content = query))

    response = model.invoke(chat_history)
    print(f'\nAI: {response.content}')
    chat_history.append(AIMessage(content = response.content))

print(f'\n\n---------- Message History ----------\n\n{chat_history}\n\n')