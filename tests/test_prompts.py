import os
import pytest

from genscai.prompts import Prompt, PromptCatalog


@pytest.fixture
def prompt_catalog() -> PromptCatalog:
    catalog = PromptCatalog("test.db")
    yield catalog
    os.remove("test.db")


def test_store_prompt(prompt_catalog):
    model_id = "x"

    prompt1 = Prompt(prompt="test", model_id=model_id, version=1)
    prompt_catalog.store_prompt(prompt1)
    prompt2 = prompt_catalog.retrieve_latest(model_id)

    assert prompt1.id == prompt2.id


def test_retrieve_latest(prompt_catalog):
    model_id = "x"

    prompt1 = Prompt(prompt="a", model_id=model_id, version=1)
    prompt_catalog.store_prompt(prompt1)

    prompt2 = Prompt(prompt="c", model_id=model_id, version=3)
    prompt_catalog.store_prompt(prompt2)

    prompt3 = Prompt(prompt="b", model_id=model_id, version=2)
    prompt_catalog.store_prompt(prompt3)

    prompt4 = prompt_catalog.retrieve_latest(model_id)

    assert prompt2.id == prompt4.id


def test_retrieve_all(prompt_catalog):
    model_id = "x"

    prompt1 = Prompt(prompt="a", model_id=model_id)
    prompt_catalog.store_prompt(prompt1)

    prompt2 = Prompt(prompt="c", model_id=model_id)
    prompt_catalog.store_prompt(prompt2)

    prompts = prompt_catalog.retrieve_all(model_id)

    assert prompt1 in prompts
    assert prompt2 in prompts


def test_delete_all(prompt_catalog):
    model_id = "x"

    prompt1 = Prompt(prompt="a", model_id=model_id)
    prompt_catalog.store_prompt(prompt1)

    prompt2 = Prompt(prompt="c", model_id=model_id)
    prompt_catalog.store_prompt(prompt2)

    prompts = prompt_catalog.retrieve_all(model_id)

    assert prompt1 in prompts
    assert prompt2 in prompts

    prompt_catalog.delete_all(model_id)

    prompts = prompt_catalog.retrieve_all(model_id)

    assert len(prompts) == 0
