import os
from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

load_dotenv()

# Clone Python codebase repo
# git clone https://github.com/pallets/[repo-url] [repo-name]
# get directory of current script file
# current_directory = os.path.dirname(os.path.abspath(__file__))
# repo_path = os.path.join(current_directory, "codebase")  # Create a new directory name
# repo = Repo.clone_from("[repo-url]", to_path=repo_path)

# Load codebase
loader = GenericLoader.from_filesystem(
    'codebase',
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
print(len(documents))
