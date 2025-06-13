import streamlit as st

from genscai.models import HuggingFaceClient as ModelClient
from genscai.tools import search_research_articles

import torch
import json

# Avoid torch RuntimeError when using Hugging Face Transformers
torch.classes.__path__ = []

MODELS = [
    ModelClient.MODEL_MISTRAL_SMALL_3_1_24B,
    ModelClient.MODEL_QWEN_3_8B,
    ModelClient.MODEL_LLAMA_3_2_3B,
]

SYSTEM_MESSAGE = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.
Always provide links to the articles you reference.
Please introduce yourself.
"""

MODEL_TOOLS = [search_research_articles]

with st.sidebar:
    st.title("IDM Research Assistant")
    st.write("Discuss infectious disease modeling with access to current disease modeling research.")

    model_id = st.selectbox("Model", options=MODELS)
    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    repeat_penalty = st.sidebar.slider("repeat_penalty", min_value=0.9, max_value=1.5, value=1.1, step=0.1)

    if st.button("Reset chat"):
        st.session_state.clear()

    # Set the model id in the sesstion state, or update the model client if the model id has changed
    if "model_id" not in st.session_state:
        st.session_state.model_id = model_id
    elif st.session_state.model_id != model_id:
        model_client = st.session_state.model_client
        messages = model_client.messages
        del model_client

        st.session_state.model_client = ModelClient(model_id)
        st.session_state.model_client.messages = messages

# Initialize the session state if we don't already have a model loaded
if "model_client" not in st.session_state:
    model_client = st.session_state.model_client = ModelClient(model_id)

    message = {"role": model_client.system_role, "content": SYSTEM_MESSAGE}

    streamed_response = model_client.chat_streamed(
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
