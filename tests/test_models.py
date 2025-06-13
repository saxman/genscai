import pytest
from typing import Iterable

from genscai.models import ModelClient, HuggingFaceClient

MODELS = [HuggingFaceClient.MODEL_LLAMA_3_1_8B, HuggingFaceClient.MODEL_MISTRAL_7B]


@pytest.fixture(params=MODELS)
def model_client(request) -> Iterable[ModelClient]:
    client = HuggingFaceClient(request.param)
    yield client


def test_generate(model_client):
    prompt = "What is the capital of France?"
    response = model_client.generate(prompt)

    assert isinstance(response, str)
    assert "Paris" in response


def test_generate_with_parameters(model_client):
    prompt = "What is the capital of France?"
    response = model_client.generate(prompt, generate_kwargs={"max_new_tokens": 1})

    assert isinstance(response, str)
    assert "Paris" in response


def test_chat(model_client):
    message = {"role": model_client.system_role, "content": "What is the capital of France?"}

    response = model_client.chat(message)

    assert "Paris" in response
    assert len(model_client.messages) == 2

    # The model should know that we're talking about capitals
    message = {"role": "user", "content": "How about Germany's?"}

    response = model_client.chat(message)

    assert "Berlin" in response
    assert len(model_client.messages) == 4


def test_chat_streamed(model_client):
    message = {"role": model_client.system_role, "content": "What is the capital of France?"}

    response = model_client.chat_streamed(message)

    assert isinstance(response, Iterable)

    content = next(response)
    assert isinstance(content, str)

    for response_part in response:
        content += response_part

    assert "Paris" in content
    assert len(model_client.messages) == 2
