## Codebase in Graph Database with Vector Index for Question-Answer Retrieval Augmented Generation

This repo loads a Python codebase from the web, parses, splits then indexes it into a graph database with an additional vector index of embeddings as node properties, then queries the graph database using a generated Cypher statement or with the vector index using semantically similar embeddings to generate an answer.

## Technologies

[Tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Parses code into a syntax tree using a language's grammar (e.g. Python Binding)  
[Code-graph](https://github.com/FalkorDB/code-graph) - Constructs graph by Modules, Classes and Functions with respective vector embeddings as node properties  
[FalkorDB](https://www.falkordb.com/) - Graph DB (Successor to RedisGraph EOL)  
[OpenAI](https://openai.com/) - LLM for QA, Cypher and RAG  
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

- Sign up for [Docker](https://www.docker.com/)
- Sign Up for [OpenAI](https://platform.openai.com/docs/quickstart/account-setup) 

## Run

In the root of this repo, create a .env file with the below keys alongside [your-values]:

> NEXT_PUBLIC_MODE=UNLIMITED  
> FALKORDB_URL=redis://localhost:6379  
> OPENAI_API_KEY=[your-value]  

Run FalkorDB via Docker

```
docker run -p 6379:6379 -it --rm falkordb/falkordb
```

Install node packages

```
npm install
```

Run the development server:

```
npm run dev
```

To interact with the QA web app, open:

> [http://localhost:3000](http://localhost:3000)

Example questions you could ask that use different indexes include:

* For Cypher Graph Search, "Which functions are in the AppGroup Class?"
* For Cypher Vector Search, "Find a few functions which have conditional statements."

## Troubleshooting

If you are rebuilding the Graph DB, first make sure FalkorDB via Docker is empty with

```
docker exec -it [container_name] redis-cli FLUSHDB
```

then inspect `repo_root` for the local folder in `route.ts`