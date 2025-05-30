import streamlit as st

from genscai import paths
from genscai.models import OllamaClient as ModelClient

import torch
import chromadb

# Avoid torch RuntimeError when using Hugging Face Transformers
torch.classes.__path__ = []

MODEL_ID = ModelClient.MODEL_MISTRAL_SMALL_3_1_24B

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
        content += f"Title: {metadata[i]['title']}\nAbstract: {abstracts[i]}\nAuthors: {metadata[i]['authors']}\nDate: {metadata[i]['date']}\nURL: https://www.medrxiv.org/content/{id}\n\n"

    return content, articles


MODEL_TOOLS = [search_research_articles]

with st.sidebar:
    st.title("IDM Research Assistant")
    st.write("Discuss infectious disease modeling with access to current disease modeling research.")

    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Initialize the session state if we don't already have a model loaded
if "model_client" not in st.session_state:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    model_client = ModelClient(model_id=MODEL_ID)

    st.session_state.messages = model_client.chat(
        messages, generate_kwargs={"temperature": temperature, "top_p": top_p}
    )
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

    model_client = st.session_state.model_client
    messages = model_client.chat(
        st.session_state.messages, tools=MODEL_TOOLS, generate_kwargs={"temperature": temperature, "top_p": top_p}
    )

    if "tool_calls" in messages[-1]:
        for tool_call in messages[-1]["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            content, data = globals()[tool_name](**tool_args)

            # Add the tool response to the message history and send it back to the model to generate a final response
            messages.append({"role": "tool", "content": content})

        messages = model_client.chat(messages, generate_kwargs={"temperature": temperature, "top_p": top_p})

    st.session_state.messages = messages

    st.chat_message("assistant").markdown(st.session_state.messages[-1]["content"])
