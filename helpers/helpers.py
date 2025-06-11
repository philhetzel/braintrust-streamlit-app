from braintrust import wrap_openai, projects, load_prompt, start_span, current_span
from tools.retriever import tool_definition, get_documents
from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import json
import random

load_dotenv()

CHAMPION_PROMPT = "e0a85a1f019f1a45" # e0a85a1f019f1a45 gpt 4.1 
CHALLENGER_PROMPT = "1174fda1b2758de1" or CHAMPION_PROMPT  # 80d80cf5801f16b2 gemini 2.5 flash

def get_project():
    return projects.create(name=os.getenv("BRAINTRUST_PROJECT_NAME"))

def get_model_attributes():
    prompt_object = load_prompt(project="StreamlitRAG", 
                            slug="rag-prompt", 
                            defaults={"tools": [tool_definition], "tool_choice": "auto", "stream": False},
                            version= "e0a85a1f019f1a45"
                            )

    MODEL = prompt_object.build()["model"]
    TOOLS = prompt_object.build()["tools"]
    return MODEL, TOOLS

def chat(input): 

    prompt_object = load_prompt(project="StreamlitRAG", 
                            slug="rag-prompt", 
                            defaults={"tools": [tool_definition], "tool_choice": "auto", "stream": False},
                            version= CHALLENGER_PROMPT # if random.random() > 1.0 else CHALLENGER_PROMPT
                            )

    MODEL = prompt_object.build()["model"]
    TOOLS = prompt_object.build()["tools"]

    client = wrap_openai(
    OpenAI(
        api_key=os.getenv("BRAINTRUST_API_KEY"),
        base_url="https://api.braintrust.dev/v1/proxy",
    )
    )

    # Format messages properly for the API
    formatted_input = format_messages_for_api(input)

    response = client.chat.completions.create(
        model=MODEL,
        messages=formatted_input,
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
        
        # Execute ALL tool calls, not just the first one
        all_docs_content = []
        for tool_call in msg.tool_calls:
            tool_args = json.loads(tool_call.function.arguments)
            
            if tool_call.function.name == "get_documents":
                docs = get_documents(**tool_args)
                docs_str = "\n\n".join([d["content"] for d in docs])
                all_docs_content.append(docs_str)
                
                # Add tool result to conversation for each tool call
                input.append({
                    "role": "tool",
                    "content": docs_str,
                    "tool_call_id": tool_call.id
                })
        
        # Combine all retrieved documents
        combined_docs_str = "\n\n=== SEPARATE QUERY ===\n\n".join(all_docs_content)
        
        # Format messages again for the final call
        formatted_input = format_messages_for_api(input)
        
        # Get final response from model with tool results
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=formatted_input,
            stream=False
        )
        
        final_content = final_response.choices[0].message.content
        #st.markdown(final_content)
        current_span().log(
            metadata={"model": MODEL,
                      "context": combined_docs_str}
        )
        return {"output": final_content, "context": combined_docs_str}
    else:
        # No tool call, just display the response
        return {"output": msg.content, "context": ""}


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