from braintrust import wrap_openai, projects, load_prompt, start_span
from tools.retriever import tool_definition, get_documents
from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import json

load_dotenv()

def get_project():
    return projects.create(name=os.getenv("BRAINTRUST_PROJECT_NAME"))

def get_model_attributes():
    prompt_object = load_prompt(project="StreamlitRAG", 
                            slug="rag-prompt", 
                            defaults={"tools": [tool_definition], "tool_choice": "auto", "stream": False},
                            version= "c0e6c9abb3a0aa92" if os.getenv("BRAINTRUST_ENV") == "prod" else None
                            )

    MODEL = prompt_object.build()["model"]
    TOOLS = prompt_object.build()["tools"]
    return MODEL, TOOLS

def chat(input): 
    print(input)

    prompt_object = load_prompt(project="StreamlitRAG", 
                            slug="rag-prompt", 
                            defaults={"tools": [tool_definition], "tool_choice": "auto", "stream": False},
                            version= "c0e6c9abb3a0aa92" if os.getenv("BRAINTRUST_ENV") == "prod" else None
                            )

    MODEL = prompt_object.build()["model"]
    TOOLS = prompt_object.build()["tools"]

    client = wrap_openai(
    OpenAI(
        api_key=os.getenv("BRAINTRUST_API_KEY"),
        base_url="https://api.braintrust.dev/v1/proxy",
    )
    )


    response = client.chat.completions.create(
        model=MODEL,
        messages=input,
        tool_choice="auto",
        tools=TOOLS,
        stream=False
    )

    msg = response.choices[0].message
    
    # If model made a tool call
    if msg.tool_calls:
        # Add the assistant's tool call message to conversation history
        input.append({
            "role": "assistant",
            "content": msg.content,  # This might be None
            "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
        })
        
        # Execute the tool call
        tool_call = msg.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)
        
        if tool_call.function.name == "get_documents":
            docs = get_documents(**tool_args)
            docs_str = "\n\n".join([d["content"] for d in docs])
            
            # Add tool result to conversation
            input.append({
                "role": "tool",
                "content": docs_str,
                "tool_call_id": tool_call.id
            })
            
            # Get final response from model with tool results
            final_response = client.chat.completions.create(
                model=MODEL,
                messages=input,
                stream=False
            )
            
            final_content = final_response.choices[0].message.content
            #st.markdown(final_content)
            print(final_content)
            return final_content
    else:
        # No tool call, just display the response
        return msg.content


def format_messages_for_api(messages):
    """Format messages for OpenAI API, handling tool calls properly."""
    formatted_messages = []
    
    for msg in messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            # Assistant message with tool calls
            formatted_msg = {
                "role": "assistant",
                "content": msg["content"] if msg["content"] is not None else "",
                "tool_calls": msg["tool_calls"]
            }
        elif msg["role"] == "tool":
            # Tool response message
            formatted_msg = {
                "role": "tool",
                "content": msg["content"],
                "tool_call_id": msg["tool_call_id"]
            }
        else:
            # Regular user or assistant message
            formatted_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
        
        formatted_messages.append(formatted_msg)
    
    return formatted_messages