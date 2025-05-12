from genscai import paths

import streamlit as st

from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TextIteratorStreamer

from tinydb import TinyDB
from threading import Thread

import torch

torch.classes.__path__ = []

model_id = "meta-llama/Llama-3.1-8B-Instruct"
chat_db_path = paths.output / "chat_db.json"
generate_kwargs = {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}
messages = None

# If this is a new chat session, get the context of the last conversation, and create a document in the db for the current session
if "doc_id" not in st.session_state:
    with TinyDB(chat_db_path) as db:
        try:
            last = db.all()[-1]
            messages = db.get(last.doc_id)["messages"]
            st.session_state.doc_id = last.doc_id
        except Exception:
            st.session_state.doc_id = db.insert({})

# Initialize chat history with system message.
if "messages" not in st.session_state:
    if messages is not None and len(messages) > 0:
        st.session_state.messages = messages
    else:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.",
            }
        ]

if "pipeline" not in st.session_state:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    st.session_state.streamer = streamer

    pipeline = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        streamer=streamer,
        max_new_tokens=1024,
    )

    st.session_state.pipeline = pipeline

    # Send the system message to the model
    pipeline(
        st.session_state.messages,
        kwargs=generate_kwargs,
    )

    # Wait until the system message response is done
    generated_text = ""
    for text in streamer:
        generated_text += text

st.title("MeChat")

# Display chat messages from history on app rerun. Skip first system message.
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    pipeline = st.session_state.pipeline

    model_kwargs = dict(
        text_inputs=st.session_state.messages,
        kwargs=generate_kwargs,
    )

    # Run the inferencing in a thread so that the output can be streamed
    thread = Thread(target=pipeline, kwargs=model_kwargs)
    thread.start()

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state.streamer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    with TinyDB(chat_db_path) as db:
        db.update({"messages": st.session_state.messages}, doc_ids=[st.session_state.doc_id])
