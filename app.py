import streamlit as st
from braintrust import wrap_openai, init_logger, start_span, load_prompt
from openai import OpenAI
import os
from dotenv import load_dotenv
from tools.retriever import get_documents, tool_definition
import json
from helpers.helpers import format_messages_for_api
load_dotenv()

logger = init_logger(
    api_key=os.getenv("BRAINTRUST_API_KEY"), 
    project="StreamlitRAG"
)

client = wrap_openai(
    OpenAI(
        api_key=os.getenv("BRAINTRUST_API_KEY"),
        base_url="https://api.braintrust.dev/v1/proxy",
    )
)

prompt_object = load_prompt(project="StreamlitRAG", 
                            slug="rag-prompt", 
                            defaults={"tools": [tool_definition], "tool_choice": "auto", "stream": False},
                            version= "c0e6c9abb3a0aa92" if os.getenv("BRAINTRUST_ENV") == "prod" else None
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
                # First call: Get model response (might include tool calls)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=format_messages_for_api(st.session_state.messages),
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
                        
                        # Get final response from model with tool results
                        final_response = client.chat.completions.create(
                            model=st.session_state["openai_model"],
                            messages=format_messages_for_api(st.session_state.messages),
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
                    final_content = msg.content = msg.content

                span.log(
                    input=st.session_state.messages,
                    output=final_content,
                    metadata={"model": MODEL}
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
