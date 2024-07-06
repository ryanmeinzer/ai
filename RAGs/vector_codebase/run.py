import os
from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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
            OpenAIEmbeddings(disallowed_special=()),
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
        OpenAIEmbeddings(disallowed_special=()),
        url=url, 
        username=username, 
        password=password, 
        index_name="vector" # default index name
    )
else:
    print('Cloning Python codebase repo', end="\n\n")
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Current directory
    repo_path = os.path.join(current_directory, "codebase")  # Create a new directory name
    repo = Repo.clone_from("[repo-url]", to_path=repo_path) # Clone repo

    print('Loading codebase', end="\n\n")
    loader = GenericLoader.from_filesystem(
        'codebase',
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    docs = loader.load()
    # print(len(docs))

    print('Splitting docs', end="\n\n")
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    split_documents = python_splitter.split_documents(docs)
    # print(len(texts))

    print("Creating Neo4jVector DB with loaded & split docs", end="\n\n")
    db = Neo4jVector.from_documents(
        split_documents, 
        OpenAIEmbeddings(disallowed_special=()),
        url=url, 
        username=username, 
        password=password, 
        index_name="vector" # default index name
        # # keyword index (only if created) is optional for Hybrid Search
        # keyword_index_name="keyword" # default index name
        # search_type="hybrid"
    )

user_query = "What is my name?"
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
#     print('Initial Prompt & Final Prompt:', prompts, end="\n\n")

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
print('[Initial Prompt]', end="\n\n")
print('[Langchain RetrievalQAWithSourcesChain Class -> Question into Embedding', end="\n\n")
print('Search Result:', response.get('sources'), end="\n\n")
print('[Final Prompt]', end="\n\n")
print('Answer:', response.get('answer'), end="\n")
total_tokens = cb.total_tokens
if total_tokens != 0:
    print(f"Tokens: {total_tokens}", end="\n\n")
print(f"Time: {formatted_total_time} seconds", end="\n\n")