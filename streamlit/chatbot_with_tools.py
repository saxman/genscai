import streamlit as st

from genscai import paths

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TextIteratorStreamer

from threading import Thread
import json
import torch
import chromadb

# Avoid torch RuntimeError
torch.classes.__path__ = []

model_id = "mistralai/Mistral-Nemo-Instruct-2407"
knowledge_base = "medrxiv_chunked_256_cosine"

system_message = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.
Do not provide references in your response, unless the user specifically asks for them.
"""

# Flag whether to render the message history. For first run, we don't want to render the message history,
# since the first response to the system message will be rendered during model initilization.
skip_message_rendering = False


def get_research_articles(search_request: str) -> list[dict]:
    """
    Get the current research in infectious diseases and disease modeling per a given search request.

    Args:
        search_request: The information that the user is looking for.
    Returns:
        Current research in infectious diseases and disease modeling for the given topic.
    """

    print(f"Search request: {search_request}")

    client = chromadb.PersistentClient(path=str(paths.output / "genscai.db"))
    collection = client.get_collection(name=knowledge_base)
    results = collection.query(query_texts=[search_request], n_results=10)

    ids = [x for x in results["ids"][0]]
    abstracts = [x for x in results["documents"][0]]
    metadata = [x for x in results["metadatas"][0]]

    # Each article has multiple chunks, and each chunk id is the article's doi with an index appended to it.
    # Therefore, we need to remove the index from the id and keep only one copy of the article.
    articles = []
    id_set = set()
    for i in range(len(ids)):
        id = ids[i].split(":")[0]

        if id in id_set:
            continue

        id_set.add(id)

        metadata[i]["doi"] = id
        metadata[i]["abstract"] = abstracts[i]
        articles.append(metadata[i])

    return articles


agent_tools = [get_research_articles]


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


def process_response(streamer):
    """
    This function processes the response from the model. If the model is calling a tool, it will call the tool and return the results.
    If the model is not calling a tool, it will return the response as is.
    """

    tool_response = ""

    for text in streamer:
        # If we're processing a tool response, capture the response and don't yield the text
        if text.startswith('{"name":') or len(tool_response) > 0:
            tool_response += text
            continue

        yield text

    # If this is a assistant reponse and not a tool response, we've already yielded the response text and can return
    if len(tool_response) == 0:
        return

    # Currently an issue with the tokenizer: the model returns 'parameters', but the tokenizer expects 'arguments'
    tool_call = json.loads(tool_response)
    tool_call["arguments"] = tool_call.pop("parameters")

    st.session_state.messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})

    # Call the tool and get the result
    tool_return = globals()[tool_call["name"]](**tool_call["arguments"])

    # Add the tool return to the message history. In this case, we're adding the abstracts of the articles.
    content = "\n\n".join([x["abstract"] for x in tool_return])
    st.session_state.messages.append({"role": "tool", "name": f"{tool_call['name']}", "content": f"{content}"})

    # We'll now generate a new response using the model, which will be the assistant's response to the tool output
    generate_kwargs = {
        "model": st.session_state.model,
        "tokenizer": st.session_state.tokenizer,
        "streamer": st.session_state.streamer,
        "messages": st.session_state.messages,
        "tools": None,
    }

    # Run the inferencing in a thread so that the output can be streamed
    # We cannot call generate() directly, since we're already yielding text to the streamer
    thread = Thread(target=generate, kwargs=generate_kwargs)
    thread.start()

    # Stream the assistant response from the model
    for text in st.session_state.streamer:
        yield text

    yield "\n\n**References:**"

    # Stream the top 5 references for the response
    for article in tool_return[:5]:
        yield f"\n* [{article['title']}](https://www.medrxiv.org/content/{article['doi']}). {article['authors']}, {article['date'].split('-')[0]}"


def process_messages(tools=None) -> None:
    """
    Process the messages in the chat history and render the response from the model.
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
        response = st.write_stream(process_response(st.session_state.streamer))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.title("IDM Research Assistant")
    st.write("Discuss infectious disease modeling with access to current disease modeling research.")

    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Initialize the session state if we don't already have a model loaded.
# This is intentionally positioned after the sidebar so that the model parameters in the sidebar are initiailized.
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
