import os
from dotenv import load_dotenv
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

load_dotenv()

url = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')
diffbot_api_key = os.getenv('DIFFBOT_KEY')

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

# query the graph via generated cypher
chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
    qa_llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    graph=graph,
    validate_cypher=True,
    return_source_documents=True
    # verbose=True,
)
user_query = "Who was Urijah Faber?"

print(chain.invoke(
    {"query": user_query},
    # return_only_outputs=True,
))