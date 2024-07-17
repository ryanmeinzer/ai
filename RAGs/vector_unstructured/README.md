## Unstructured Data Question-Answer Retrieval Augmented Generation

This repo loads unstructured data from the web, indexes it into a vector database, then queries the database to generate an answer.

## Technologies

[Neo4jVector](https://python.langchain.com/v0.2/docs/integrations/vectorstores/neo4jvector/) - DB  
[OpenAI](https://openai.com/) - LLM for QA & RAG  
[LangChain](https://www.langchain.com/) - Framework to build apps with LLMs  

## QA RAG Chain Logs (outputted by running repo)

[Question]  
[Initial Prompt]  
[Cypher Graph Search]  
[Question into Embedding]  
[Cypher Vector Search]  
[Retrieved Similar Embeddings]  
[Search Result]  
[Final Prompt]  
[Answer]  
[Tokens]  
[Time]  

## Prerequisites

- Sign Up for [OpenAI](https://platform.openai.com/docs/quickstart/account-setup) 
- Sign Up for [Neo4j Aura DB](https://neo4j.com/cloud/platform/aura-graph-database)
- Sign Up for [LangSmith](https://python.langchain.com/v0.1/docs/get_started/quickstart/#langsmith)

## Run

In the root of this repo, create a .env file with the below keys alongside [your-values]:

> OPENAI_API_KEY=[your-value]
> LANGCHAIN_TRACING_V2=true
> LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
> LANGCHAIN_API_KEY=[your-value]
> NEO4J_URI=[your-value]
> NEO4J_USERNAME=[your-value]
> NEO4J_PASSWORD=[your-value]

In `run.py`:
> adjust the `wikipedia_query` and `user_query` variables according to your preference.

While in the root of this repo, in the CLI run:

```python run.py```