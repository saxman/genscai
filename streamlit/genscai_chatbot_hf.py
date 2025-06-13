import streamlit as st

from genscai.models import HuggingFaceClient, OllamaClient
from genscai.tools import search_research_articles

import torch
import json

# Avoid torch RuntimeError when using Hugging Face Transformers
torch.classes.__path__ = []

SYSTEM_MESSAGE = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.
Always provide links to the articles you reference.
Please introduce yourself.
"""

MODEL_CLIENTS = [
    OllamaClient,
    HuggingFaceClient,
]

MODEL_TOOLS = [search_research_articles]

# Initialize the session state if we don't already have a model loaded
if "model_client" not in st.session_state:
    st.session_state.model_id = MODEL_CLIENTS[0].TOOL_MODELS[0]
    st.session_state.model_client = MODEL_CLIENTS[0](st.session_state.model_id)

with st.sidebar:
    st.title("IDM Research Assistant")
    st.write("Discuss infectious disease modeling with access to current disease modeling research.")

    model_id = st.selectbox("Model", options=st.session_state.model_client.TOOL_MODELS)
    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    repeat_penalty = st.sidebar.slider("repeat_penalty", min_value=0.9, max_value=1.5, value=1.1, step=0.1)

    model_client = st.selectbox("Model Client", options=MODEL_CLIENTS, format_func=lambda x: x.__name__)

    if not isinstance(st.session_state.model_client, model_client) or st.session_state.model_id != model_id:
        del st.session_state.model_client

        st.session_state.model_id = model_id
        st.session_state.model_client = model_client(st.session_state.model_id)

    if st.button("Reset chat"):
        st.session_state.clear()

# Either generate and display the system message or display the chat message history
if len(st.session_state.model_client.messages) == 0:
    message = {"role": st.session_state.model_client.system_role, "content": SYSTEM_MESSAGE}

    streamed_response = st.session_state.model_client.chat_streamed(
        message,
        generate_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": 1024,
            "repeat_penalty": repeat_penalty,
        },
    )

    with st.chat_message("assistant"):
        response = st.write_stream(streamed_response)
else:
    # Only render assistant and user messages (not tool messages) and not the system (first) message
    messages = [
        x for x in st.session_state.model_client.messages[1:] if x["role"] in ["assistant", "user"] and "content" in x
    ]
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    message = {"role": "user", "content": prompt}

    streamed_response = st.session_state.model_client.chat_streamed(
        message,
        generate_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": 1024,
            "repeat_penalty": repeat_penalty,
        },
        tools=MODEL_TOOLS,
    )

    with st.chat_message("assistant"):
        st.write_stream(streamed_response)

# TODO: Determine better layout
with st.popover("Messages"):
    st.code(json.dumps(st.session_state.model_client.messages, indent=4), language="json", line_numbers=True)
