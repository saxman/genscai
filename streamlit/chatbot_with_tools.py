import streamlit as st

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TextIteratorStreamer

from threading import Thread

import torch

torch.classes.__path__ = []

model_id = "meta-llama/Llama-3.1-8B-Instruct"

system_message = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.
"""

# Flag whether to render the message history. For first run, we don't want to redner the message history, since the first response, to the system message, will be rendered durng model initilization.
skip_message_rendering = False


def generate(model, tokenizer, streamer, messages) -> None:
    tools = None

    input_tokens = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    model.generate(
        **input_tokens,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )


def process_messages() -> None:
    generate_kwargs = {
        "model": st.session_state.model,
        "tokenizer": st.session_state.tokenizer,
        "streamer": st.session_state.streamer,
        "messages": st.session_state.messages,
    }

    # Run the inferencing in a thread so that the output can be streamed
    thread = Thread(target=generate, kwargs=generate_kwargs)
    thread.start()

    # Stream the assistant response in chat message feed
    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state.streamer)

    # Capture the system message response before continuing
    # response = ""
    # for text in st.session_state.streamer:
    #     response += text

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


st.title("IDM Chat")

# Initialize the session state if we don't already have a model loaded
if "model" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id)
    st.session_state.streamer = TextIteratorStreamer(
        st.session_state.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    st.session_state.model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    process_messages()

    skip_message_rendering = True

# Display chat messages from history on app re-run. Skip the system message.
if not skip_message_rendering:
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    process_messages()
