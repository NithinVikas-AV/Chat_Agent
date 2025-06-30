import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from all_common_prompts import *

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
project_id = os.getenv("PROJECT_ID")
collection_name = os.getenv("collection_name")
session_id = os.getenv("SESSION_ID")

# model initialization
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=api_key)

# Firebase Firestore setup
PROJECT_ID = project_id 
COLLECTION_NAME = collection_name
SESSION_ID = session_id

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")

print("\n\nStart chatting with the AI. Type 'exit' to quit.\n")

while True:
    human_input = input("\nUser: ")
    if human_input.lower() == "exit":
        break

    # Add user message to Firestore
    chat_history.add_user_message(human_input)

    # Generate response from Gemini
    ai_response = model.invoke(chat_history.messages)

    # Save AI response to Firestore
    chat_history.add_ai_message(str(ai_response.content))

    print(f"\nAI: {ai_response.content}")