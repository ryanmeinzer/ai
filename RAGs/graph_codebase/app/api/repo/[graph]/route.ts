import OpenAI from "openai";
import { QUESTIONS } from '../questions';
import { graphSchema } from "../graph_ops";
import { FalkorDB, Graph } from 'falkordb';
import { NextRequest, NextResponse } from "next/server";
import { ChatCompletionCreateParams, ChatCompletionMessageParam, ChatCompletionMessageToolCall, ChatCompletionTool } from 'openai/resources/chat/completions.mjs';

// convert a structured graph schema into a string representation
// used in a model prompt
async function GraphSchemaToPrompt(
    graph: Graph,
    graphId: string,
    db: FalkorDB
) {
    // Retrieve graph schema
    let schema: any = await graphSchema(graphId, db);

    // Build a string description of graph schema
    let desc: string = "The knowledge graph schema is as follows:\n";

    //-------------------------------------------------------------------------
    // Describe labels
    //-------------------------------------------------------------------------

    // list labels
    desc = desc + "The graph contains the following node labels:\n";
    for (const lbl in schema["labels"]) {
        desc = desc + `${lbl}\n`;
    }

    // specify attributes associated with each label
    for (const lbl in schema["labels"]) {
        let node_count = schema["labels"][lbl]['node_count'];
        let attributes = schema["labels"][lbl]['attributes'];
        let attr_count = Object.keys(attributes).length;

        if (attr_count == 0) {
            desc = desc + `the ${lbl} label has ${node_count} nodes and has no attributes\n`;
        } else {
            desc = desc + `the ${lbl} label has ${node_count} nodes and is associated with the following attribute(s):\n`;
            for (const attr in attributes) {
                let type = attributes[attr]['type'];
                desc = desc + `'${attr}' which is of type ${type}\n`;
            }
        }
    }

    desc = desc + "The graph contains the following relationship types:\n"

    //-------------------------------------------------------------------------
    // Describe relationships
    //-------------------------------------------------------------------------

    // list relations
    for (const relation in schema["relations"]) {
        desc = desc + `${relation}\n`;
    }

    // specify attributes associated with each relationship
    for (const relation in schema["relations"]) {
        let connect = schema["relations"][relation]['connect'];
        let edge_count = schema["relations"][relation]['edge_count'];
        let attributes = schema["relations"][relation]['attributes'];
        let attr_count = Object.keys(attributes).length;

        if (attr_count == 0) {
            desc = desc + `the ${relation} relationship has ${edge_count} edges and has no attributes\n`;
        } else {
            desc = desc + `the ${relation} relationship has ${edge_count} edges and is associated with the following attribute(s):\n`;
            for (const attr in attributes) {
                let type = attributes[attr]['type'];
                desc = desc + `'${attr}' which is of type ${type}\n`;
            }
        }

        if (connect.length > 0) {
            desc = desc + `the ${relation} relationship connects the following labels:\n`
            for (let i = 0; i < connect.length; i += 2) {
                let src = connect[i];
                let dest = connect[i + 1];
                desc = desc + `${src} is connected via ${relation} to ${dest}\n`;
            }
        }
    }

    desc = desc + `This is the end of the knowledge graph schema description.\n`

    //-------------------------------------------------------------------------
    // include graph indices
    //-------------------------------------------------------------------------

    // vector indices
    let query = `CALL db.indexes() YIELD label, properties, types, entitytype`;
    let res = await graph.query(query);

    // process indexes
    let indexes: any = res.data;
    if (indexes.length > 0) {
        let index_prompt = "The knowledge graph contains the following indexes:\n"
        for (let i = 0; i < indexes.length; i++) {
            const index = indexes[i];
            const label: string = index['label'];
            const entityType: string = index['entitytype'];
            const props = index['properties'];
            const types = index['types'];

            for (const prop of props) {
                const propTypes: string[] = types[prop];
                for (let j = 0; j < propTypes.length; j++) {
                    const idxType: string = propTypes[j];
                    index_prompt += `${entityType} of type ${label} have a ${idxType} index indexing its ${prop} attribute\n`;
                }
            }
        }
        index_prompt += `This is the end of our indexes list
        To use a Vector index use the following procedure:
        CALL db.idx.vector.queryNodes(<LABEL>, <PROPERTY>, <N>, <VALUE>)) YIELD node

        The procedure returns up to N nodes that have a <PROPERTY> value which is semantically close to
        <VALUE>.

        Here are a few question / answer examples of using the vector index:
        Question: Find 3 functions which have nested loops, return a list of callers
        Cypher query: CALL db.idx.vector.queryNodes('Function', 'source', 3, 'nested loops') YIELD node MATCH (f)-[:CALLS]->(node) RETURN node.name

        Question: List 5 Classes which contain a function that raise an exception
        Cypher query: CALL db.idx.vector.queryNodes('Function', 'source', 5, "raise exception") YIELD node MATCH (c:Class)-[:CONTAINS]->(node) RETURN class.name, node.name
    `;

        desc += index_prompt;

    }

    return desc;
}

interface NodeWithName {
    name: string;
}

// handle instruction from OpenAI
// there are two types of accepted instructions:
// 1. Run query.
// 2. Generate embeddings and run a query.
async function run_query
(
    graph: Graph,
    query: string,
    // add log for QA RAG Chain
    log: string[], // array as param
    totalTokens: number
) {
    let params = {};
    // var to track Vector (vs Graph) search 
    let vectorSearchUsed = false

    // handle cases where the query contains a vector index utilization
    // CALL db.idx.vector.queryNodes('Function', 'src_embeddings', 5, 'x= x   1')
    if(query.indexOf('CALL db.idx.vector.queryNodes(') !== -1) {
        // query utilizes a vector index
        // extract semantic value and produce embeddings

        vectorSearchUsed   = true // if vectorSearchUsed, set to true
        let startIdx       = query.indexOf('CALL db.idx.vector.queryNodes(');
        let endIdx         = query.indexOf(')');
        let proc_call      = query.substring(startIdx + 'CALL db.idx.vector.queryNodes('.length, endIdx);
        let args           = proc_call.split(',');
        let semantic_value = args[args.length - 1];

        const openai   = new OpenAI();
        let response   = await openai.embeddings.create({input:semantic_value, model:'text-embedding-ada-002'});
        let embeddings = response.data[0].embedding;

        // Add the tokens used for creating embeddings, with optional chaining for safety
        totalTokens += response.usage?.total_tokens || 0;

        args[args.length - 1] = "vecf32($embeddings)";

        // rewrite query
        params = {embeddings: embeddings};
        let nodeDetails = 'YIELD node RETURN node.name AS name'
        let rewrite = `CALL db.idx.vector.queryNodes(${args.join(',')}) ${nodeDetails}`;

        query = rewrite;

        // Log semantic value & rewritten query
        log.push(`\nQuestion into Embedding: \n${semantic_value}`)
        log.push(`\nCypher Vector Search: \n${query}`)
    }

    let result = await graph.roQuery(query, {params: params});

    // Log node names if vectorSearchUsed
    if (vectorSearchUsed) {
        const logMessage = result.data && result.data.length > 0
            ? `\nRetrieved Similar Embeddings (Node Names): \n${result.data.map(node => (node as NodeWithName).name).join(', ')}`
            : '\nRetrieved similar embeddings not found';
        log.push(logMessage);
    } else {
        log.push(`\nCypher Graph Search: \n${query}`);
    }

    return result.data;
}

// Chat bot handler
export async function GET(request: NextRequest, { params }: { params: { graph: string } }) {
    const startTime = new Date().getTime();
    let totalTokens = 0;
    const log: string[] = [];

    const graph_id = params.graph;
    let question = request.nextUrl.searchParams.get("q");

    if(!question) {
        log.push('\nQuestion not specified');
        return NextResponse.json({ message: 'Question not specified' }, { status: 400 })
    }

    log.push(`\nQuestion: \n${question}`);

    //-------------------------------------------------------------------------
    // Connect to graph
    //-------------------------------------------------------------------------

    // hard coded graph id
    const db = await FalkorDB.connect({
        url: process.env.FALKORDB_URL || 'redis://localhost:6379',
    });
    const graph = db.selectGraph(graph_id);

    //-------------------------------------------------------------------------
    // Construct prompt
    //-------------------------------------------------------------------------

    let graph_schema = await GraphSchemaToPrompt(graph, graph_id, db);

    let prompt: string = `You're a Cypher expert, with access to the following graph:
    ${graph_schema}
    The graph represents a code base.
    Please note the graph you're querying does NOT supports regular expression matching via the =~ symbol
    Do not generate queries using the '=~' symbol.`;

    //-------------------------------------------------------------------------
    // Send prompt to OpenAI
    //-------------------------------------------------------------------------

    // Define conversation first message
    const openai = new OpenAI();
    let messages: ChatCompletionMessageParam[] = [
        { role: 'system', content: prompt },
        { role: 'user', content: `Question: ${question}`, name: 'user' },
    ];

    // Define OpenAI tool
    const tools:ChatCompletionTool[] = [
    {
        type: "function",
        function: {
            name: "run_query",
            description: "Run a Cypher query",
            parameters: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "Cypher query to run",
                    },
                },
                required: ["query"],
            },
        },
    },
    ];

    // Construct conversation body
    let body: ChatCompletionCreateParams = {
        model: "gpt-3.5-turbo",
        messages: messages,
        tools: tools,
        tool_choice: "auto",
    };

    // Get completion for conversation
    let response = await openai.chat.completions.create(body);
    totalTokens += response.usage?.total_tokens || 0;
    log.push(`\nInitial Prompt: \n${prompt}`);

    //-------------------------------------------------------------------------
    // Perform instruction
    //-------------------------------------------------------------------------    

    //const instruction = response.choices[0]['message']['content'];
    const responseMessage = response.choices[0].message;

    // Step 2: check if the model wanted to call a function
    const toolCalls: ChatCompletionMessageToolCall[] | undefined  = responseMessage.tool_calls;

    let query_result: string = '';  // response from query

    if (toolCalls) {
        for (const toolCall of toolCalls) {
            const functionName = toolCall.function.name;
            const functionArgs = JSON.parse(toolCall.function.arguments);
            
            if(functionName != 'run_query') {
                return NextResponse.json({ result: "Unknown function to run" }, { status: 500 });
            }

            let result = await run_query(graph, functionArgs.query, log, totalTokens);
            query_result = JSON.stringify(result);
            log.push(`\nSearch Result: \n${query_result}`);
        }
    } else {
        log.push('\nUnexpected instruction');
        return NextResponse.json({ result: "Unexpected instruction" }, { status: 500 });
    }

    //-------------------------------------------------------------------------
    // Digest response
    //-------------------------------------------------------------------------

    prompt = `This is the user's question: ${question}
    And this is the data we've got from our knowledge graph: ${query_result}
    Please formulate an answer to the user question based on the data we've got from the knowledge graph`;

    messages = [{ "role": "system", "content": prompt }];
    log.push(`\nFinal Prompt: \n${prompt}`);
    response = await openai.chat.completions.create({
        "model": "gpt-3.5-turbo",
        "messages": messages,
    });
    const endTime = new Date().getTime();
    const responseTime = (endTime - startTime) / 1000;
    const formattedResponseTime = responseTime.toFixed(2);
    const answer = response.choices[0]['message']['content'];
    log.push(`\nAnswer: \n${answer}`);
    totalTokens += response.usage?.total_tokens || 0;
    log.push(`\nTokens: \n${totalTokens}`);
    log.push(`Time: ${formattedResponseTime} seconds`);
    console.log(log.join('\n\n'))

    return NextResponse.json({ result: answer }, { status: 200 });
}