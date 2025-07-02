import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from simple_tools import *


load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")

tools = [
    greet_user, 
    reverse_string,  
    concatenate_strings,
]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm=llm,  
    tools=tools,  
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,  
    tools=tools,  
    verbose=True, 
    handle_parsing_errors=True,
)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)