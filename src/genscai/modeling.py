import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from transformers.utils import logging
logging.set_verbosity_error() 


def load_model(model_id, model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )

    return model, tokenizer


def generate_text(model, prompt, tokenizer, generate_kwargs):
    generate_kwargs["bos_token_id"] = tokenizer.bos_token_id
    generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
    generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    messages = [
        {
            "role": "user",
             "content": prompt
        }
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        **generate_kwargs
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    
    return tokenizer.decode(response, skip_special_tokens=True)


def delete_model(model):
    del model

    if torch.cuda.is_available():
        print('emptying cuda cache')
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        print('emptying mps cache')
        torch.mps.empty_cache()


def print_model_info(model):
    print(f'model : size : {model.get_memory_footprint() // 1024 ** 2} MB')

    try:
        print(f'model : is quantized : {model.is_quantized}')
        print(f'model : quantization method : {model.quantization_method}')
    except:
        print(f'model : is quantized : False')
        pass
        
    try:
        print(f'model : 8-bit quantized : {model.is_loaded_in_8bit}')
    except:
        pass
    
    try:
        print(f'model : 4-bit quantized : {model.is_loaded_in_4bit}')
    except:
        pass

    param = next(model.parameters())
    print(f'model : on GPU (CUDA) : {param.is_cuda}')
    print(f'model : on GPU (MPS) : {param.is_mps}')


def print_device_map(model):
    import pprint
    
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(model.hf_device_map)


def print_cuda_device_info():
    import pynvml
    
    pynvml.nvmlInit()
    
    print(f'driver version : {pynvml.nvmlSystemGetDriverVersion()}')
    
    devices = pynvml.nvmlDeviceGetCount()
    print(f'device count : {devices}')
    
    for i in range(devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        print(f'device {i} : {pynvml.nvmlDeviceGetName(handle)}')
    
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f'device {i} : mem total : {info.total // 1024 ** 2} MB')
        print(f'device {i} : mem used  : {info.used // 1024 ** 2} MB')
        print(f'device {i} : mem free  : {info.free // 1024 ** 2} MB')