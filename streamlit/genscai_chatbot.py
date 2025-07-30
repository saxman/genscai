from genscai import paths

from aimu.models import HuggingFaceClient, OllamaClient
from aimu.tools import MCPClient
from aimu.memory import ConversationManager

import streamlit as st
import torch
import json

# Avoid torch RuntimeError when using Hugging Face Transformers
torch.classes.__path__ = []

SYSTEM_MESSAGE = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.

LASER is a disease modeling framework that allows users to create and run disease models.
It is used to simulate the spread of infectious diseases and evaluate the impact of interventions.
You can use tools to explore LASER documentation, explore the LASER code, and answer questions about the frramework.

Also use avialable tools to search for research articles, summarize them, and answer questions about their content.
Alwasy provide links to the articles that you reference.
"""

Please introduce yourself.
"""

MODEL_CLIENTS = [
    OllamaClient,
    HuggingFaceClient,
]

MCP_SERVERS = {
    "mcpServers": {
        "genscai": {"command": "python", "args": [str(paths.package / "tools.py")]},
        "laser_core": {"url": "https://gitmcp.io/InstituteforDiseaseModeling/laser"},
        "laser_generic": {"url": "https://gitmcp.io/InstituteforDiseaseModeling/laser-generic"},
        "sequentialthinking": {
            "command": "docker",
            "args": ["run", "--rm", "-i", "mcp/sequentialthinking"],
        },
    }
}

# Initialize the session state if we don't already have a model loaded. This only happens first run.
if "model_client" not in st.session_state:
    st.session_state.model_id = MODEL_CLIENTS[0].TOOL_MODELS[0]
    st.session_state.model_client = MODEL_CLIENTS[0](st.session_state.model_id)

    st.session_state.mcp_client = MCPClient(MCP_SERVERS)
    st.session_state.model_client.mcp_client = st.session_state.mcp_client

    st.session_state.conversation_manager = ConversationManager(
        db_path=str(paths.output / "chat_history.json"),
        use_last_conversation=True,
    )
    st.session_state.model_client.messages = st.session_state.conversation_manager.messages

with st.sidebar:
    st.title("IDM Research Assistant")
    st.write("Discuss infectious disease modeling with access to current disease modeling research.")

    model = st.selectbox("Model", options=st.session_state.model_client.TOOL_MODELS, format_func=lambda x: x.name)
    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    repeat_penalty = st.sidebar.slider("repeat_penalty", min_value=0.9, max_value=1.5, value=1.1, step=0.1)
    model_client = st.selectbox("Model Client", options=MODEL_CLIENTS, format_func=lambda x: x.__name__)

    # If the model client has changed (e.g. OllamaClient to HuggingFaceClient), create a new mode client instance.
    # Otherwise, if the model has changed, create a new instance of the model client using the new model.
    if not isinstance(st.session_state.model_client, model_client):
        del st.session_state.model_client

        st.session_state.model_id = model_client.TOOL_MODELS[0]
        st.session_state.model_client = model_client(st.session_state.model_id)
        st.session_state.model_client.mcp_client = st.session_state.mcp_client

        st.rerun()
    elif st.session_state.model != model:
        del st.session_state.model_client

        st.session_state.model_id = model_id
        st.session_state.model_client = model_client(st.session_state.model_id)
        st.session_state.model_client.mcp_client = st.session_state.mcp_client

        st.rerun()

    if st.button("Reset chat"):
        # Create a new conversation that will be used as the "last" conversation when the app is reloaded.
        st.session_state.conversation_manager.create_new_conversation()

        st.session_state.clear()
        st.rerun()

# Either generate and stream the initial user message response or display the chat message history.
if len(st.session_state.model_client.messages) == 0:
    streamed_response = st.session_state.model_client.chat_streamed(
        INITIAL_USER_MESSAGE,
        generate_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": 1024,
            "repeat_penalty": repeat_penalty,
        },
        tools=st.session_state.mcp_client.get_tools(),
    )

    with st.chat_message("assistant"):
        response = st.write_stream(streamed_response)

    st.session_state.conversation_manager.update_conversation(st.session_state.model_client.messages)
else:
    # Only render assistant and user messages (not tool messages) and not the system message and initial user message.
    messages = [
        x for x in st.session_state.model_client.messages[2:] if x["role"] in ["assistant", "user"] and "content" in x
    ]
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.chat_message("user").markdown(prompt)

    streamed_response = st.session_state.model_client.chat_streamed(
        prompt,
        generate_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": 1024,
            "repeat_penalty": repeat_penalty,
        },
        tools=st.session_state.mcp_client.get_tools(),
    )

    with st.chat_message("assistant"):
        st.write_stream(streamed_response)

    st.session_state.conversation_manager.update_conversation(st.session_state.model_client.messages)

# TODO: Determine better layout
with st.popover("Messages"):
    st.code(
        json.dumps(st.session_state.model_client.messages, indent=4),
        language="json",
        line_numbers=True,
    )
