from genscai import paths

from aimu.agents import Agent
from aimu.models import AnthropicClient, HuggingFaceClient, OllamaClient, StreamingContentType
from aimu.tools import MCPClient
from aimu.history import ConversationManager

import streamlit as st
import torch
import json

# Avoid torch RuntimeError when using Hugging Face Transformers
torch.classes.__path__ = []

SYSTEM_MESSAGE = """
You are a friendly chatbot that assists the user with research in infectious diseases and infectious disease modeling.
Reply in short, concise sentrences, unless the user asks for a more detailed answer.

LASER and Starsim are disease modeling framework that allow users to create and run disease models.
They are used to simulate the spread of infectious diseases and evaluate the impact of interventions.
You can use tools to explore LASER and Starsim documentation, explore code, and answer questions about the frrameworks.

Also use avialable tools to search for research articles, summarize them, and answer questions about their content.
Always provide links to the articles that you reference.
"""

INITIAL_USER_MESSAGE = """
Please introduce yourself by describing your purpose and capabilities, including listing what tools you have access to.
"""

MODEL_CLIENTS = [
    OllamaClient,
    AnthropicClient,
    HuggingFaceClient,
]

# Upper bound on the model<->tools rounds the agent runs before forcing a final answer.
MAX_ITERATIONS = 10

MCP_SERVERS = {
    "mcpServers": {
        "genscai": {"command": "python", "args": [str(paths.package / "tools.py")]},
        "laser_core": {"url": "https://gitmcp.io/InstituteforDiseaseModeling/laser"},
        "laser_generic": {"url": "https://gitmcp.io/InstituteforDiseaseModeling/laser-generic"},
        "starsim_core": {"url": "https://gitmcp.io/starsimhub/starsim"},
        # "sequentialthinking": {
        #     "command": "docker",
        #     "args": ["run", "--rm", "-i", "mcp/sequentialthinking"],
        # },
    }
}


def build_model_client(client_cls, model):
    """Build a base model client wrapped in an Agent that holds the MCP tools, exposed as a client.

    Tool execution now lives in the Agent, not the model client, so tools reach the model only
    through ``Agent(base_client, tools=...)``. ``as_model_client()`` returns a client-like view
    whose ``chat()`` drives the multi-round tool-use loop. The base client is returned alongside
    it for the sidebar selectors and ``isinstance`` checks, which the agentic view doesn't expose.
    """
    base_client = client_cls(model, system_message=SYSTEM_MESSAGE)
    agent = Agent(base_client, tools=st.session_state.tools, max_iterations=MAX_ITERATIONS)
    return base_client, agent.as_model_client()


def stream_chat_response(streamed_response):
    """Render a stream of StreamChunk into the Streamlit UI."""
    current_phase = None
    current_box = None
    current_text = ""

    for chunk in streamed_response:
        if chunk.phase == StreamingContentType.TOOL_CALLING:
            current_phase = None
            with st.expander("🔧 Tool call"):
                st.markdown(f"**Tool call:** {chunk.content['name']}")
                st.markdown(f"**Tool response:** {chunk.content['response']}")
            continue

        if chunk.phase != current_phase:
            current_phase = chunk.phase
            current_text = ""
            current_box = None

        current_text += chunk.content
        if current_text:
            if current_box is None:
                current_box = (
                    st.expander("🤔 Thinking").empty()
                    if chunk.phase == StreamingContentType.THINKING
                    else st.chat_message("assistant").empty()
                )
            current_box.markdown(current_text)


# Initialize the session state if we don't already have a model loaded. This only happens first run.
if "model_client" not in st.session_state:
    # Connect to the MCP servers once and expose their tools as agent tools.
    st.session_state.tools = MCPClient(MCP_SERVERS).as_tools()

    st.session_state.model = MODEL_CLIENTS[0].TOOL_MODELS[0]
    st.session_state.base_client, st.session_state.model_client = build_model_client(
        MODEL_CLIENTS[0], st.session_state.model
    )

    st.session_state.conversation_manager = ConversationManager(
        db_path=str(paths.output / "chat_history.json"),
        use_last_conversation=True,
    )
    st.session_state.model_client.messages = st.session_state.conversation_manager.messages

with st.sidebar:
    st.title("IDM Modeling Research Assistant")
    st.write("Discuss infectious disease modeling with access to current disease modeling research.")

    # Selectors and isinstance checks use base_client; the agentic view doesn't expose TOOL_MODELS.
    model = st.selectbox("Model", options=st.session_state.base_client.TOOL_MODELS, format_func=lambda x: x.name)
    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    repeat_penalty = st.sidebar.slider("repeat_penalty", min_value=0.9, max_value=1.5, value=1.1, step=0.1)
    model_client = st.selectbox("Model Client", options=MODEL_CLIENTS, format_func=lambda x: x.__name__)

    # If the model client has changed (e.g. OllamaClient to HuggingFaceClient), create a new model client instance.
    # Otherwise, if the model has changed, create a new instance of the model client using the new model.
    if not isinstance(st.session_state.base_client, model_client):
        st.session_state.model = model_client.TOOL_MODELS[0]
        st.session_state.base_client, st.session_state.model_client = build_model_client(
            model_client, st.session_state.model
        )
        st.rerun()
    elif st.session_state.model != model:
        st.session_state.model = model
        st.session_state.base_client, st.session_state.model_client = build_model_client(model_client, model)
        st.rerun()

    if st.button("Reset chat"):
        # Create a new conversation that will be used as the "last" conversation when the app is reloaded.
        st.session_state.conversation_manager.create_new_conversation()

        st.session_state.clear()
        st.rerun()

generate_kwargs = {
    "temperature": temperature,
    "top_p": top_p,
    "max_new_tokens": 1024,
    "repeat_penalty": repeat_penalty,
}

# Either generate and stream the initial user message response or display the chat message history.
if len(st.session_state.model_client.messages) == 0:
    stream_chat_response(
        st.session_state.model_client.chat(INITIAL_USER_MESSAGE, generate_kwargs=generate_kwargs, stream=True)
    )
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
    stream_chat_response(st.session_state.model_client.chat(prompt, generate_kwargs=generate_kwargs, stream=True))
    st.session_state.conversation_manager.update_conversation(st.session_state.model_client.messages)

# TODO: Determine better layout
with st.popover("Messages"):
    st.code(
        json.dumps(st.session_state.model_client.messages, indent=4),
        language="json",
        line_numbers=True,
    )
