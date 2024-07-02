import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
import time
from langchain_community.callbacks import get_openai_callback
from typing import List, Any, Sequence
from langchain_core.documents import Document

load_dotenv()

url = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')

def check_index() -> bool:
    try:
        Neo4jVector.from_existing_index(
            OpenAIEmbeddings(), 
            url=url, 
            username=username, 
            password=password, 
            index_name="vector" # default index name
        )
        return True
    except:
        return False
    
index_exists = check_index()

if index_exists:
    print("Using existing Neo4jVector DB", end="\n\n")
    db = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(), 
        url=url, 
        username=username, 
        password=password, 
        index_name="vector" # default index name
    )
else:
    # load docs/text
    wikipedia_query = "Urijah Faber"
    docs = WikipediaLoader(
        query=wikipedia_query, 
        # defaults to 25 (albeit API docs advise default is 100)
        load_max_docs=10
        ).load()

    # split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(docs)

    # connect to Neo4j, create embeddings, create vector db (with Hybrid Search option)
    print("Creating Neo4jVector DB", end="\n\n")
    db = Neo4jVector.from_documents(
        split_documents, 
        OpenAIEmbeddings(), 
        url=url, 
        username=username, 
        password=password, 
        index_name="vector" # default index name
        # # keyword index (only if created) is optional for Hybrid Search
        # keyword_index_name="keyword" # default index name
        # search_type="hybrid"
    )

user_query = "Who was Urijah Faber?"
# docs_with_score = db.similarity_search_with_score(user_query, k=2)
# for doc, score in docs_with_score:
#     print("-" * 80)
#     print("Score: ", score)
#     print(doc.page_content)
#     print("-" * 80)

# retriever for Q&A (default k=4)
retriever = db.as_retriever(search_kwargs={'k': 6})

chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever
)

# Print the prompts of the chain
# def on_llm_start(serialized: Any, prompts: List[str], **kwargs: Any) -> None:
#     print('Prompts:', prompts, end="\n\n")

def on_retriever_end(documents: Sequence[Document], **kwargs: Any) -> Any:
    print('Retrieved Similar Embeddings:')
    for index, doc in enumerate(documents):
        source = doc.metadata.get('source', 'No Source')
        print(f"Document ID {index + 1}: {source}")
    print()

start_time = time.time()
with get_openai_callback() as cb: 
    # cb.on_llm_start = on_llm_start
    cb.on_retriever_end = on_retriever_end
    response = chain.invoke({
        "question": user_query,
    },
    # return_only_outputs=True
)
end_time = time.time()
total_time = end_time - start_time
formatted_total_time = f"{total_time:.2f}"

print('Question:', response.get('question'), end="\n\n")
print('[Langchain Prompts for OpenAI LLM]', end="\n\n")
print('[Langchain RetrievalQAWithSourcesChain Class -> Question into Semantic Embedding]', end="\n\n")
print('Final Source for Answer:', response.get('sources'), end="\n\n")
print('Answer:', response.get('answer'), end="\n")
total_tokens = cb.total_tokens
if total_tokens != 0:
    print(f"Total Tokens: {total_tokens}", end="\n\n")
print(f"Time: {formatted_total_time} seconds", end="\n\n")
