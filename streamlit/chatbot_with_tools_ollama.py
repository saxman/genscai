import streamlit as st

from genscai import paths
from genscai.models import OllamaClient as ModelClient

from threading import Thread
import json
import torch
import chromadb

# Avoid torch RuntimeError
torch.classes.__path__ = []

model_id = ModelClient.MODEL_MISTRAL_SMALL_3_1_24B
knowledge_base = "medrxiv_chunked_256_cosine"

system_message = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.
Do not provide references in your response, unless the user specifically asks for them.
"""

with st.sidebar:
    st.title("IDM Research Assistant")
    st.write("Discuss infectious disease modeling with access to current disease modeling research.")

    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Initialize the session state if we don't already have a model loaded.
if "model_client" not in st.session_state:
    messages = [{"role": "system", "content": system_message}]
    model_client = ModelClient(model_id=model_id)

    st.session_state.messages = model_client.chat(messages)
    st.session_state.model_client = model_client

# Only render assistant and user messages (not tool messages)
messages = [x for x in st.session_state.messages if x["role"] in ["assistant", "user"] and "content" in x]
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages = st.session_state.model_client.chat(st.session_state.messages)

    st.chat_message("assistant").markdown(st.session_state.messages[-1]["content"])
