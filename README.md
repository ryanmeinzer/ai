# AI Tools including LLM RAGs with Graph & Vector DBs

## Retrieval Augmented Generations (RAGs)

[Graph Codebase](/RAGs/graph_codebase/)  
This repo loads a Python codebase from the web, parses, splits, converts then indexes it into a graph database with an additional vector index of embeddings as node properties, then queries the graph database using a generated Cypher statement or with the vector index using semantically similar embeddings to generate an answer.

[Graph Unstructured](/RAGs/graph_unstructured/)  
This repo loads unstructured data from the web, converts then indexes it into a graph database, then queries the database using a generated Cypher statement to generate an answer. 

[Vector Codebase](/RAGs/vector_codebase/)
This repo loads a Python codebase from the web, parses, splits then indexes it into a vector database, then queries the database using semantically similar embeddings to generate an answer.  

[Vector Unstructured](/RAGs/vector_unstructured/)  
This repo loads unstructured data from the web, splits then indexes it into a vector database, then queries the database using semantically similar embeddings to generate an answer.
