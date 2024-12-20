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
    
    print(f'model : on GPU : {next(model.parameters()).is_cuda}')


def print_device_map(model):
    import pprint
    
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(model.hf_device_map)


def print_device_info():
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