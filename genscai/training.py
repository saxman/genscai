MUTATION_GENERATE_KWARGS = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.75,
    "top_k": 50,
    "top_p": 0.95,
}

MUTATION_POS_PROMPT_TEMPLATE = """
Read the language model prompt and scientific paper abstract below. Expand the prompt so that a language model would correctly determine that the abstract
explicitly refers to or uses a disease modeling technique.

Do not include the names of specific diseases in the prompt. Do not include the abstract in the prompt.

Wrap the prompt in a <prompt> tag, e.g. <prompt>This is the mutated prompt</prompt>. Only include the prompt in the <prompt> tag.

Prompt:
{prompt}

Abstract:
{abstract}
""".strip()

MUTATION_NEG_PROMPT_TEMPLATE = """
Read the language model prompt and scientific paper abstract below. Expand the prompt so that a language model would correctly determine that the abstract
DOES NOT explicitly refer to or use a disease modeling technique.

Do not include the names of specific diseases in the prompt. Do not include the abstract in the prompt.

Wrap the prompt in a <prompt> tag, e.g. <prompt>This is the mutated prompt</prompt>. Only include the prompt in the <prompt> tag.

Prompt:
{prompt}

Abstract:
{abstract}
""".strip()
