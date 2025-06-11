import streamlit as st
from braintrust import wrap_openai, init_logger, start_span, load_prompt
from openai import OpenAI
import os
from dotenv import load_dotenv
from tools.retriever import get_documents, tool_definition
import json
from helpers.helpers import format_messages_for_api, CHAMPION_PROMPT, CHALLENGER_PROMPT
import random

def validate_conversation(messages):
    """Ensure all tool calls have corresponding responses."""
    validated = []
    pending_tool_calls = {}
    
    for msg in messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            validated.append(msg)
            # Track tool calls that need responses
            for tc in msg["tool_calls"]:
                pending_tool_calls[tc["id"]] = True
        elif msg["role"] == "tool":
            validated.append(msg)
            # Mark this tool call as resolved
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id in pending_tool_calls:
                del pending_tool_calls[tool_call_id]
        else:
            # Regular message
            validated.append(msg)
    
    # If there are pending tool calls, remove the assistant message that made them
    if pending_tool_calls:
        print(f"WARNING: Removing incomplete tool calls: {list(pending_tool_calls.keys())}")
        validated = [msg for msg in validated 
                    if not (msg["role"] == "assistant" and "tool_calls" in msg and 
                           any(tc["id"] in pending_tool_calls for tc in msg["tool_calls"]))]
    
    return validated

load_dotenv()

# Initialize logger to send logs to Braintrust
logger = init_logger(
    api_key=os.getenv("BRAINTRUST_API_KEY"), 
    project="StreamlitRAG"
)

# Initialize model client through Braintrust AI Proxy
# Will be able to swap models regardless of the model provider
client = wrap_openai(
    OpenAI(
        api_key=os.getenv("BRAINTRUST_API_KEY"),
        base_url="https://api.braintrust.dev/v1/proxy",
    )
)

# Load prompt from Braintrust based on a version number
prompt_object = load_prompt(project="StreamlitRAG", 
                            slug="rag-prompt", 
                            defaults={"tools": [tool_definition], "tool_choice": "auto", "stream": False},
                            version= CHAMPION_PROMPT if random.random() < 1.0 else CHALLENGER_PROMPT
                            )

MODEL = prompt_object.build()["model"]
TOOLS = prompt_object.build()["tools"]

st.title("Braintrust Docs Assistant")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = MODEL

if "messages" not in st.session_state:
    st.session_state.messages = [prompt_object.prompt.messages[0].as_dict()]

# Display chat history
for message in st.session_state.messages:
    if message["role"] in ["system", "tool"]:
        continue  # Don't display system or tool messages in the UI
    with st.chat_message(message["role"]):
        st.markdown(message["content"] if message["content"] else "")

# Handle new user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with start_span("chat_completion", type="task") as span:
                # Validate conversation structure before API call
                validated_messages = validate_conversation(st.session_state.messages)
                st.session_state.messages = validated_messages  # Update session state with validated messages
                
                # First call: Get model response (might include tool calls)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=format_messages_for_api(validated_messages),
                    tool_choice="auto",
                    tools=TOOLS,
                    stream=False
                )

                msg = response.choices[0].message
                
                # If model made a tool call
                if msg.tool_calls:
                    # Add the assistant's tool call message to conversation history
                    st.session_state.messages.append({
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
                        st.session_state.messages.append({
                            "role": "tool",
                            "content": docs_str,
                            "tool_call_id": tool_call.id
                        })
                        
                        # Debug: Print the conversation structure
                        formatted_msgs = format_messages_for_api(st.session_state.messages)
                        print(f"DEBUG: Messages before final call: {len(formatted_msgs)} messages")
                        for i, msg in enumerate(formatted_msgs[-3:]):  # Show last 3 messages
                            print(f"  {i}: role={msg['role']}, has_tool_calls={'tool_calls' in msg}, has_tool_call_id={'tool_call_id' in msg}")
                        
                        # Get final response from model with tool results
                        final_response = client.chat.completions.create(
                            model=st.session_state["openai_model"],
                            messages=format_messages_for_api(st.session_state.messages),
                            tools=None,  # Explicitly set to None to prevent more tool calls
                            stream=False
                        )
                        
                        final_content = final_response.choices[0].message.content
                        st.markdown(final_content)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_content
                        })
                else:
                    # No tool call, just display the response
                    st.markdown(msg.content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })
                    final_content = msg.content

                span.log(
                    input=st.session_state.messages[1],
                    output=final_content,
                    metadata={"model": MODEL,
                              "context": docs_str}
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
