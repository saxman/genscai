import pprint
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.utils import logging as log
import ollama
import aisuite
import logging

log.set_verbosity_error()

logger = logging.getLogger(__name__)

MODEL_KWARGS = {
    "low_cpu_mem_usage": True,
    "device_map": "balanced",
    "torch_dtype": "auto",
}

class ModelClient:
    MODEL_LLAMA_3_1_8B = None
    MODEL_LLAMA_3_2_3B = None
    MODEL_GEMMA_2_9B = None
    MODEL_GEMMA_3_12B = None
    MODEL_PHI_4_14B = None
    MODEL_DEEPSEEK_R1_8B = None
    MODEL_MISTRAL_7B = None
    MODEL_MISTRAL_NEMO_12B = None
    MODEL_QWEN_2_5_7B = None
    MODEL_GPT_4O_MINI = None
    MODEL_GPT_4O = None

    def __init__(self, model_id, model_kwargs):
        if model_id is None:
            raise ValueError("Model not supported by model client")

        self.model_id = model_id
        self.model_kwargs = model_kwargs


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

    def __init__(self, model_id, model_kwargs):
        super().__init__(model_id, model_kwargs)
        self.client = aisuite.Client()

    def generate_text(self, prompt, generate_kwargs):
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            self.model_id, messages, temperature=generate_kwargs["temperature"]
        )

        return response.choices[0].message.content


class OllamaClient(ModelClient):
    MODEL_LLAMA_3_1_8B = "llama3.1:8b"
    MODEL_LLAMA_3_2_3B = "llama3.2:3b"
    MODEL_GEMMA_2_9B = "gemma2:9b"
    MODEL_PHI_4_14B = "phi4:14b"
    MODEL_DEEPSEEK_R1_8B = "deepseek-r1:8b"
    MODEL_MISTRAL_7B = "mistral:7b"
    MODEL_MISTRAL_NEMO_12B = "mistral-nemo"
    MODEL_QWEN_2_5_7B = "qwen2.5:7b"

    def __init__(self, model_id, model_kwargs):
        super().__init__(model_id, model_kwargs)

        ollama.pull(model_id)

    def generate_text(self, prompt, generate_kwargs) -> str:
        response: ollama.ChatResponse = ollama.chat(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": generate_kwargs["temperature"]},
        )

        return response.message.content


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

    def __init__(self, model_id, model_kwargs):
        super().__init__(model_id, model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    def __del__(self):
        if torch.cuda.is_available():
            print("emptying cuda cache")
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            print("emptying mps cache")
            torch.mps.empty_cache()

    def generate_text(self, prompt, generate_kwargs) -> str:
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

    def print_model_info(self):
        print(f"model : size : {self.model.get_memory_footprint() // 1024**2} MB")

        try:
            print(f"model : is quantized : {self.model.is_quantized}")
            print(f"model : quantization method : {self.model.quantization_method}")
        except:
            print("model : is quantized : False")
            pass

        try:
            print(f"model : 8-bit quantized : {self.model.is_loaded_in_8bit}")
        except:
            pass

        try:
            print(f"model : 4-bit quantized : {self.model.is_loaded_in_4bit}")
        except:
            pass

        param = next(self.model.parameters())
        print(f"model : on GPU (CUDA) : {param.is_cuda}")
        print(f"model : on GPU (MPS) : {param.is_mps}")

    def print_device_map(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.model.hf_device_map)
