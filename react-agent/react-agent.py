"""
Language models are transformer networks with input as prompt text and the output as response text. 
Within the network, they compute numerous matrix multiplications on the prompt vector to convert it into the response vector.

Chapter 5 in https://arxiv.org/pdf/2303.12712.pdf

Prompting language models and retrieving the response is a widely used interation pattern in applications.
However, there are root level problems with language models such as a lack of latest world knowledge,
inability to understand symbolic representation as in mathematics, and code exectution. 
Sophisticated approches around using LLMs are evolving. One of those approches is creating Agents powered by LLMs.

Agents use LLMs as a reasoning engine to decide the next steps and their order of execution.
It differs from sequences of actions in traditional applications such RAGs where the 
retrieval and generation are fixed actions occuring one after the another.

Agents perform actions through the interface provided by _tools_. 
A tool is essentially a python function that performs a certain task.
An example is search tool that queries the internet given a prompt. 
Another example is a retriver tool that queries a document index given a prompt. 
A variety of open source tools such as a calculator, Slack integrator, SparkSQL, Wikipedia etc. are already available for use. 

Under the cover, a language model has access to the description of the tool.
If the user query requires the use of the tool,
the language model calls the tool function and uses its output for the next step. 

In this application, let us look at how to create an LLM agent that can search the web.

"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import OpenAI

import streamlit as st

model_service = os.getenv("MODEL_SERVICE_ENDPOINT", "http://localhost:8000/v1")

# st.title("ReAct agent demo")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", 
#                                      "content": "How can I help you?"}]
    
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

llm = ChatOpenAI(base_url=model_service, 
                 api_key="EMPTY",
                 streaming=True)

wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

tools = [search]
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")
# Construct the ReAct agent
agent = create_react_agent(llm,
                           tools,
                           prompt,
                           agent="chat-zero-shot-react-description")
print(prompt)

# # Define the Langchain chain
# prompt = ChatPromptTemplate.from_template("""You are an helpful code assistant that can help developer to code for a given {input}. 
#                                           Generate the code block at first, and explain the code at the end.
#                                           If the {input} is not making sense, please ask for more clarification.""")

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
agent_executor.invoke({"input": "Hi"})
#agent_executor.invoke({"input": "what is LangChain?"})

# chain = (
#     {"input": RunnablePassthrough()}
#     | prompt
#     | llm
# )

# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").markdown(prompt)
    
#     st_callback = StreamlitCallbackHandler(st.container())
#     response = chain.invoke(prompt, {"callbacks": [st_callback]})

#     st.chat_message("assistant").markdown(response.content)    
#     st.session_state.messages.append({"role": "assistant", "content": response.content})
#     st.rerun()


"""
* Debug this
* Put it in the application format
"""