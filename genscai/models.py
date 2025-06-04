import pprint
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.utils import logging as log
from transformers import TextIteratorStreamer
import ollama
import aisuite
import logging
from typing import Iterator

log.set_verbosity_error()

logger = logging.getLogger(__name__)

class ModelClient:
    MODEL_LLAMA_3_1_8B = None
    MODEL_LLAMA_3_2_3B = None
    MODEL_LLAMA_3_3_70B = None

    MODEL_GEMMA_2_9B = None
    MODEL_GEMMA_3_12B = None

    MODEL_PHI_4_14B = None

    MODEL_DEEPSEEK_R1_8B = None

    MODEL_MISTRAL_7B = None
    MODEL_MISTRAL_NEMO_12B = None
    MODEL_MISTRAL_SMALL_3_1_24B = None

    MODEL_QWEN_2_5_7B = None
    MODEL_QWEN_3_8B = None

    MODEL_GPT_4O_MINI = None
    MODEL_GPT_4O = None

    def __init__(self, model_id, model_kwargs):
        if model_id is None:
            raise ValueError("Model not supported by model client")

        self.model_id = model_id
        self.model_kwargs = model_kwargs

        self.messages = []


class AisuiteClient(ModelClient):
    MODEL_LLAMA_3_1_8B = "ollama:llama3.1:8b"
    MODEL_LLAMA_3_2_3B = "ollama:llama3.2:3b"

    MODEL_GEMMA_2_9B = "ollama:gemma2:9b"

    MODEL_PHI_4_14B = "ollama:phi4:14b"

    MODEL_DEEPSEEK_R1_8B = "ollama:deepseek-r1:8b"

    MODEL_MISTRAL_7B = "ollama:mistral:7b"
    MODEL_MISTRAL_NEMO_12B = "ollama:mistral-nemo"

    MODEL_QWEN_2_5_7B = "ollama:qwen2.5:7b"

    MODEL_GPT_4O_MINI = "openai:gpt-4o-mini"
    MODEL_GPT_4O = "openai:gpt-4o"

    def __init__(self, model_id):
        super().__init__(model_id, None)
        self.client = aisuite.Client()

    def generate(self, prompt, generate_kwargs):
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            self.model_id, messages, temperature=generate_kwargs["temperature"]
        )

        return response.choices[0].message.content


class OllamaClient(ModelClient):
    MODEL_LLAMA_3_3_70B = "llama3.3:70b"
    MODEL_LLAMA_3_1_8B = "llama3.1:8b"
    MODEL_LLAMA_3_2_3B = "llama3.2:3b"

    MODEL_GEMMA_2_9B = "gemma2:9b"
    MODEL_GEMMA_3_12B = "gemma3:12b"

    MODEL_PHI_4_14B = "phi4:14b"

    MODEL_DEEPSEEK_R1_8B = "deepseek-r1:8b"

    MODEL_MISTRAL_7B = "mistral:7b"
    MODEL_MISTRAL_NEMO_12B = "mistral-nemo:12b"
    MODEL_MISTRAL_SMALL_3_1_24B = "mistral-small3.1:24b"

    MODEL_QWEN_2_5_7B = "qwen2.5:7b"
    MODEL_QWEN_3_8B = "qwen3:8b"

    def __init__(self, model_id):
        super().__init__(model_id, None)

        ollama.pull(model_id)

    # for generate_kwargs, see https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    def generate(self, prompt: str, generate_kwargs: dict) -> str:
        response: ollama.GenerateResponse = ollama.generate(model=self.model_id, prompt=prompt, options=generate_kwargs)

        return response["response"]

    def genterate_streamed(self, prompt: str, generate_kwargs: dict) -> Iterator[str]:
        response = ollama.generate(model=self.model_id, prompt=prompt, options=generate_kwargs, stream=True)

        return response

    def _chat(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> None:
        self.messages.append(message)

        if tools is not None and self.model_id == OllamaClient.MODEL_LLAMA_3_1_8B:
            logger.warning(
                "Tool usage with Llama 3.1 8B is not fully supported, ref: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/"
            )

    def _handle_tool_calls(self, response, tools: dict) -> None:
        message = {"role": response.message.role}
        self.messages.append(message)

        message["tool_calls"] = []
        for tool_call in response["message"].tool_calls:
            message["tool_calls"].append(
                {
                    "type": "function",
                    "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                }
            )

            for tool in tools:
                if tool.__name__ == tool_call.function.name:
                    tool_response = tool(**tool_call.function.arguments)

                    tool_message = {"role": "tool", "name": tool_call.function.name, "content": str(tool_response)}

                    self.messages.append(tool_message)

                    break
        
        logger.debug("Tools calls: f{message['tool_calls']}")

        return

    def chat(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> str:
        self._chat(message, generate_kwargs, tools)

        response = ollama.chat(model=self.model_id, messages=self.messages, options=generate_kwargs, tools=tools)

        if response["message"].tool_calls:
            self._handle_tool_calls(response, tools)

            response = ollama.chat(model=self.model_id, messages=self.messages, options=generate_kwargs, tools=tools)

        self.messages.append({"role": response["message"].role, "content": response["message"].content})

        return response["message"].content

    def chat_streamed(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> Iterator[str]:
        self._chat(message, generate_kwargs, tools)

        response = ollama.chat(
            model=self.model_id, messages=self.messages, options=generate_kwargs, tools=tools, stream=True
        )

        response_part = next(response)

        if response_part["message"].tool_calls:
            self._handle_tool_calls(response_part, tools)

            response = ollama.chat(
                model=self.model_id, messages=self.messages, options=generate_kwargs, tools=tools, stream=True
            )

            response_part = next(response)

        content = response_part["message"].content
        yield content

        for chunk in response:
            content += chunk["message"].content
            yield chunk["message"].content

        self.messages.append({"role": response_part["message"].role, "content": content})


class HuggingFaceClient(ModelClient):
    MODEL_LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MODEL_LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"

    MODEL_DEEPSEEK_R1_8B = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    MODEL_GEMMA_2_9B = "google/gemma-2-9b-it"
    MODEL_GEMMA_3_12B = "google/gemma-3-12b-it"

    MODEL_PHI_4_14B = "microsoft/phi-4"

    MODEL_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
    MODEL_MISTRAL_NEMO_12B = "mistralai/Mistral-Nemo-Instruct-2407"

    MODEL_QWEN_2_5_7B = "Qwen/Qwen2.5-7B-Instruct-1M"

    DEFAULT_MODEL_KWARGS = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }

    def __init__(self, model_id, model_kwargs = None):
        super().__init__(model_id, model_kwargs)

        if model_kwargs is None:
            model_kwargs = self.DEFAULT_MODEL_KWARGS

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, attn_implementation="eager")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    def __del__(self):
        if torch.cuda.is_available():
            print("emptying cuda cache")
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            print("emptying mps cache")
            torch.mps.empty_cache()

    def generate(self, prompt, generate_kwargs) -> str:
        generate_kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            self.model.device
        )

        outputs = self.model.generate(input_ids, **generate_kwargs)

        response = outputs[0][input_ids.shape[-1] :]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)

        return response_text

    def _chat(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> None:
        self.messages.append(message)

        if tools is not None and self.model_id == HuggingFaceClient.MODEL_LLAMA_3_1_8B:
            logger.warning(
                "Tool usage with Llama 3.1 8B is not fully supported, ref: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/"
            )

    def chat(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> str:
        self._chat(message, generate_kwargs, tools)

        input_tokens = self.tokenizer.apply_chat_template(
            self.messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        output_tokens = self.model.generate(
            **input_tokens,
            **generate_kwargs
        )

        response_tokens = output_tokens[0][input_tokens["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        self.messages.append({"role": "assistant", "content": response})

        return response

    def chat_streamed(self, message: dict, generate_kwargs: dict = None, tools: dict = None) -> Iterator[str]:
        self._chat(message, generate_kwargs, tools)

        if not hasattr(self, "streamer"):
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        input_tokens = self.tokenizer.apply_chat_template(
            self.messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        self.model.generate(
            **input_tokens,
            streamer=self.streamer,
            **generate_kwargs
        )

        content = ""
        for chunk in self.streamer:
            content += chunk
            yield chunk

        self.messages.append({"role": "assistant", "content": content})

        return content

    def print_model_info(self):
        print(f"model : size : {self.model.get_memory_footprint() // 1024**2} MB")

        try:
            print(f"model : is quantized : {self.model.is_quantized}")
            print(f"model : quantization method : {self.model.quantization_method}")
        except AttributeError:
            print("model : is quantized : False")
            pass

        try:
            print(f"model : 8-bit quantized : {self.model.is_loaded_in_8bit}")
        except AttributeError:
            pass

        try:
            print(f"model : 4-bit quantized : {self.model.is_loaded_in_4bit}")
        except AttributeError:
            pass

        param = next(self.model.parameters())
        print(f"model : on GPU (CUDA) : {param.is_cuda}")
        print(f"model : on GPU (MPS) : {param.is_mps}")

    def print_device_map(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.model.hf_device_map)
