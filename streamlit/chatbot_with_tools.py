import streamlit as st

from genscai import paths

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TextIteratorStreamer

from threading import Thread
import json
import torch
import chromadb

torch.classes.__path__ = []

model_id = "meta-llama/Llama-3.1-8B-Instruct"

system_message = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.
Do not provide references in your response, unless the user specifically asks for them.
"""

# Flag whether to render the message history. For first run, we don't want to render the message history,
# since the first response to the system message will be rendered during model initilization.
skip_message_rendering = False


def retrieve_current_disease_research(search_request: str) -> list[dict]:
    """
    Get the current research in infectious diseases and disease modeling per a given search request.

    Args:
        search_request: The information that the user is looking for.
    Returns:
        Current research in infectious diseases and disease modeling for the given topic.
    """

    client = chromadb.PersistentClient(path=str(paths.output / "genscai.db"))
    collection = client.get_collection(name="medxriv")
    results = collection.query(query_texts=[search_request], n_results=5)

    print(f"Search request: {search_request}")

    return results


agent_tools = [retrieve_current_disease_research]


def generate(model, tokenizer, streamer, messages, tools) -> None:
    """
    Generate a response using the model and tokenizer, and stream the output.
    """

    input_tokens = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    model.generate(
        **input_tokens,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )


def stream_response(streamer):
    tool_response = ""

    for text in streamer:
        # If we're processing a tool response, capture the response and don't yield the text
        if text.startswith('{"name":') or len(tool_response) > 0:
            tool_response += text
            continue

        yield text

    # If this is a assistant reponse (not a tool response), we've yielded all of the text and can return
    if len(tool_response) == 0:
        return

    # Currently an issue with the tokenizer: the model returns 'parameters', but the tokenizer expects 'arguments'
    tool_call = json.loads(tool_response)
    tool_call["arguments"] = tool_call.pop("parameters")

    st.session_state.messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})

    tool_name = tool_call["name"]
    tool_parameters = tool_call["arguments"]

    tool_index = -1
    for i, tool in enumerate(agent_tools):
        if tool.__name__ == tool_name:
            tool_index = i
            break

    tool_return = agent_tools[tool_index](**tool_parameters)

    abstracts = [x for x in tool_return["documents"][0]]
    ids = [x for x in tool_return["ids"][0]]
    titles = [x["title"] for x in tool_return["metadatas"][0]]
    authors = [x["authors"] for x in tool_return["metadatas"][0]]

    content = "\n\n".join(abstracts)
    st.session_state.messages.append({"role": "tool", "name": f"{tool_name}", "content": f"{content}"})

    generate_kwargs = {
        "model": st.session_state.model,
        "tokenizer": st.session_state.tokenizer,
        "streamer": st.session_state.streamer,
        "messages": st.session_state.messages,
        "tools": None,
    }

    # Run the inferencing in a thread so that the output can be streamed
    thread = Thread(target=generate, kwargs=generate_kwargs)
    thread.start()

    # Stream the assistant response
    for text in st.session_state.streamer:
        yield text

    yield "\n\n**References:**"

    # Stream the references for the response
    for i in range(len(ids)):
        yield f"\n* [{titles[i]}](https://www.medrxiv.org/content/{ids[i]}). {authors[i]}.\n"


def process_messages(tools=None) -> None:
    """
    Process the messages in the chat history and generate a response using the model.
    """

    generate_kwargs = {
        "model": st.session_state.model,
        "tokenizer": st.session_state.tokenizer,
        "streamer": st.session_state.streamer,
        "messages": st.session_state.messages,
        "tools": tools,
    }

    # Run the inferencing in a thread so that the output can be streamed
    thread = Thread(target=generate, kwargs=generate_kwargs)
    thread.start()

    # Stream the assistant response in chat message feed
    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(st.session_state.streamer))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.title("IDM Chat")
    st.write("Discuss infectious disease modeling with access to current disease modeling research.")

    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Initialize the session state if we don't already have a model loaded.
# This is intentionally positioned after the sidebar so that the sidebar is not blocked by model loading
# and the model parameters can be initiailized.
if "model" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_message}]

    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id)
    st.session_state.streamer = TextIteratorStreamer(
        st.session_state.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    st.session_state.model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Send the system message to the model to initialize the chat
    process_messages()

    skip_message_rendering = True

# Display chat messages from history on app re-run. Skip rendering on first run.
if not skip_message_rendering:
    # Only render assistant and user messages
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

    process_messages(tools=agent_tools)
