from braintrust import traced, projects
from pinecone.grpc import PineconeGRPC as Pinecone
from pydantic import BaseModel
import voyageai
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

project = projects.create(os.getenv("BRAINTRUST_PROJECT_NAME"))

class Args(BaseModel):
    query: str

class Document(BaseModel):
    content: str


class DocumentOutput(BaseModel):
    text: List[Document]

@traced(type="tool")
def get_documents(query: str):
    #Setup Voyage and Pinecone clients
    vo = voyageai.Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("braintrust-rag-bot")

    #Embed query and perform cosine similarity search to Pinecone, then create a dict of results
    xq = vo.embed(query, model="voyage-3", input_type='query').embeddings[0]
    res = index.query(xq, top_k=10, include_metadata=True)
    res = [{"content": match['metadata']['content'] } for match in res['matches']]
    
    #Grab the text from the results and rerank them using Voyage. Return the top 15 after reranking
    text = [match['content']for match in res]
    reranked_docs = vo.rerank(query, text, model='rerank-2', top_k=3)
    indices = [doc.index for doc in reranked_docs.results ]
    docs = [res[x] for x in indices]
    
    return docs

tool_definition = {
    "type": "function",
    "function": {
        "name": "get_documents",
        "description": "Find information about Braintrust, an LLM evaluation and observability platform",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's message"
                }
            },
            "required": ["query"]
        }
    }
}

# Create the Braintrust tool but don't overwrite the original function
braintrust_tool = project.tools.create(name="Get Documents", 
                                       handler=get_documents, 
                                       parameters=Args, 
                                       returns=DocumentOutput,
                                       slug = 'get-documents-streamlit',
                                       if_exists='replace')

project.prompts.create(name="rag-prompt",
                        description="A prompt for a RAG bot",
                        #prompt="You are a helpful assistant that can answer questions about the documents provided. You can use the get_documents tool to retrieve documents.",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that works for Braintrust, a start up that has built an LLM evaluations and observability platform.  that can answer questions about the documents provided. Please help answer questions about Braintrust and the platform. You can use the get_documents tool to retrieve information from the Braintrust docs."}
                        ],
                        model = "gpt-4o",
                        tools=[braintrust_tool],
                        if_exists='replace')