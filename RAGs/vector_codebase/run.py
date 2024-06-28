import os
from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Clone Python codebase repo
current_directory = os.path.dirname(os.path.abspath(__file__))  # Current directory
repo_path = os.path.join(current_directory, "codebase")  # Create a new directory name
repo = Repo.clone_from("https://github.com/pallets/flask", to_path=repo_path) # Clone repo

# Load codebase
loader = GenericLoader.from_filesystem(
    'codebase',
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
docs = loader.load()
# print(len(docs))

# split docs
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(docs)
# print(len(texts))

# create db
db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

# ToDo - experiment with gpt-3.5-turbo 
llm = ChatOpenAI(model="gpt-4")

# prompt for LLM to generate this search query
initial_prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, initial_prompt)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, final_prompt)

qa = create_retrieval_chain(retriever_chain, document_chain)

question = "What does the View Class do?"
result = qa.invoke({"input": question})
print(result["answer"])