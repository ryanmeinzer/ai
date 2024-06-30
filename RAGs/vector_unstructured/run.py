import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

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
    print("Using existing Neo4jVector DB")
    db = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(), 
        url=url, 
        username=username, 
        password=password, 
        index_name="vector" # default index name
        # # keyword index (only if created) is optional for Hybrid Search
        # keyword_index_name="keyword" # default index name
        # search_type="hybrid"
    )
else:
    # load docs/text
    wikipedia_query = "Urijah Faber"
    docs = WikipediaLoader(
        query=wikipedia_query, 
        # load_max_docs=2
        ).load()

    # split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(docs)

    # connect to Neo4j, create embeddings, create vector db (with Hybrid Search option)
    print("Creating Neo4jVector DB")
    db = Neo4jVector.from_documents(
        split_documents, 
        OpenAIEmbeddings(), 
        url=url, 
        username=username, 
        password=password, 
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
    ChatOpenAI(temperature=0), chain_type="stuff", retriever=retriever
)

print(chain.invoke(
    {"question": user_query},
    # return_only_outputs=True,
))