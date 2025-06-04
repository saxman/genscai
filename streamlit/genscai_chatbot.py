import streamlit as st

from genscai import paths
from genscai.models import OllamaClient as ModelClient

import torch
import chromadb
import json

# Avoid torch RuntimeError when using Hugging Face Transformers
torch.classes.__path__ = []

MODELS = [ModelClient.MODEL_MISTRAL_SMALL_3_1_24B, ModelClient.MODEL_QWEN_3_8B, ModelClient.MODEL_LLAMA_3_3_70B, ModelClient.MODEL_LLAMA_3_2_3B]

KNOWLEDGE_BASE_PATH = str(paths.output / "medrxiv.db")
KNOWLEDGE_BASE_ID = "articles_cosign_chunked_256"

SYSTEM_MESSAGE = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.
Always provide links to the articles you reference.
Please introduce yourself.
"""

def search_research_articles(search_request: str) -> tuple[str, list[dict]]:
    """
    Search for current research articles in infectious diseases and disease modeling per a given search request.

    Args:
        search_request: The information that the user is looking for.
    Returns:
        Current research articles on infectious diseases and disease modeling for the given topic.
    """

    client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_PATH)
    collection = client.get_collection(name=KNOWLEDGE_BASE_ID)
    results = collection.query(query_texts=[search_request], n_results=10)

    ids = [x for x in results["ids"][0]]
    abstracts = [x for x in results["documents"][0]]
    metadata = [x for x in results["metadatas"][0]]

    # Each article has multiple chunks, and each chunk id is the article's doi with an index appended to it.
    # To get a unique set of articles, we need to remove the index from the id and keep only one copy of article doi.
    articles = []
    content = "Relevant research articles:\n\n"
    id_set = set()
    for i in range(len(ids)):
        id = ids[i].split(":")[0]

        if id in id_set:
            continue

        id_set.add(id)
        articles.append({"id": id, "abstract": abstracts[i], "metadata": metadata[i]})
        content += f"Title: {metadata[i]['title']}\nAbstract: {abstracts[i]}\nAuthors: {metadata[i]['authors']}\nDate: {metadata[i]['date']}\nLink: https://www.medrxiv.org/content/{id}\n\n"

    return content, articles

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
        st.session_state.model_client = ModelClient(model_id=model_id)
        st.session_state.model_client.messages = model_client.messages

# Initialize the session state if we don't already have a model loaded
if "model_client" not in st.session_state:
    model_client = st.session_state.model_client = ModelClient(model_id=model_id)

    message = {"role": "system", "content": SYSTEM_MESSAGE}

    streamed_response = model_client.chat_streamed(
        message,
        generate_kwargs={"temperature": temperature, "top_p": top_p, "max_new_tokens": 512, "repeat_penalty": repeat_penalty}
    )

    with st.chat_message("assistant"):
        response = st.write_stream(streamed_response)
else:
    # Only render assistant and user messages (not tool messages)
    messages = [
        x for x in st.session_state.model_client.messages if x["role"] in ["assistant", "user"] and "content" in x
    ]
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    message = {"role": "user", "content": prompt}

    streamed_response = st.session_state.model_client.chat_streamed(
        message, generate_kwargs={"temperature": temperature, "top_p": top_p, "max_new_tokens": 512, "repeat_penalty": repeat_penalty}, tools=MODEL_TOOLS
    )

    with st.chat_message("assistant"):
        st.write_stream(streamed_response)

# TODO: Determine better layout
with st.popover("Messages"):
    st.code(json.dumps(st.session_state.model_client.messages, indent=4), language="json", line_numbers=True)
