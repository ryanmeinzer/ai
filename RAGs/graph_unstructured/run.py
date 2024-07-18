import os
from dotenv import load_dotenv
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
import time
from langchain_community.callbacks import get_openai_callback
from langchain_core.outputs import LLMResult
from typing import List, Any
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

url = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')
diffbot_api_key = os.getenv('DIFFBOT_KEY')

def check_index() -> bool:
    try:
        Neo4jGraph(
            url=url, 
            username=username, 
            password=password, 
        )
        return True
    except:
        return False
    
index_exists = check_index()

if index_exists:
    print("Using existing Neo4jGraph DB", end="\n\n")
    graph = Neo4jGraph(
        url=url, 
        username=username, 
        password=password, 
    )
else:
    # load docs/text
    wikipedia_query = "Urijah Faber"
    docs = WikipediaLoader(
        query=wikipedia_query, 
        # defaults to 25 (albeit API docs advise default is 100)
        load_max_docs=10
        ).load()

    # convert to graph docs
    diffbot_nlp = DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)
    graph_docs = diffbot_nlp.convert_to_graph_documents(docs)

    # connect to Neo4jGraph & populate graph db with graph docs
    graph = Neo4jGraph(url=url, username=username, password=password)
    graph.add_graph_documents(
        graph_docs,
        baseEntityLabel=True,
        include_source=True
        )
    # graph.refresh_schema()

# global vars
total_tokens = 0
total_time = 0

class CypherHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized: Any, prompts: List[str], **kwargs: Any) -> None:
        print('\nQuestion:', user_query, end="\n\n")
        print('Initial Prompt:', prompts, end="\n\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # Below is handled in GraphCypherQAChain with verbose=True
        # print('Cypher Graph Search:', response.generations[0][0].text, end="\n\n")
        # print ('[Search Result]')
        global total_tokens
        total_tokens += response.llm_output['token_usage']['total_tokens']

class QAHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Any, prompts: List[str], **kwargs: Any) -> None:
        self.start_time = time.time()
        print('\nFinal Prompt:', prompts, end="\n\n")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print('Answer:', response.generations[0][0].text, end="\n")
        global total_tokens
        total_tokens += response.llm_output['token_usage']['total_tokens']
        end_time = time.time()
        global total_time
        total_time += end_time - self.start_time

# Instantiate the handlers
handler1 = CypherHandler()
handler2 = QAHandler()

chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4", callbacks=[handler1]),
    qa_llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", callbacks=[handler2]),
    graph=graph,
    validate_cypher=True,
    return_source_documents=True,
    verbose=True,
)

# query the graph via generated cypher
user_query = "Who was Urijah Faber?"

start_time = time.time()
with get_openai_callback() as cb:
    response = chain.invoke(
        {"query": user_query},
    # return_only_outputs=True
    )
if total_tokens != 0:
    print(f"\nTokens: {total_tokens}", end="\n\n")
end_time = time.time()
total_time += end_time - start_time
formatted_total_time = f"{total_time:.2f}"
print(f"Time: {formatted_total_time} seconds", end="\n\n")