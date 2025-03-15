import re
from tinydb import TinyDB
import pynvml


class ReadOnlyTinyDB(TinyDB):
    def insert(self, *args, **kwargs):
        raise PermissionError("Database is in read-only mode")

    def insert_multiple(self, *args, **kwargs):
        raise PermissionError("Database is in read-only mode")

    def update(self, *args, **kwargs):
        raise PermissionError("Database is in read-only mode")

    def remove(self, *args, **kwargs):
        raise PermissionError("Database is in read-only mode")

    def truncate(self, *args, **kwargs):
        raise PermissionError("Database is in read-only mode")


def extract_dates(s: str) -> list:
    """Extract dates from a string using regex. (YYY-MM-DD)"""
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    # Extract dates using re.search
    match = re.findall(date_pattern, s)
    if match:
        return match
    else:
        print("No dates found in the file path.")
        return None


def print_cuda_device_info():
    pynvml.nvmlInit()

    print(f"driver version : {pynvml.nvmlSystemGetDriverVersion()}")

    devices = pynvml.nvmlDeviceGetCount()
    print(f"device count : {devices}")

    for i in range(devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        print(f"device {i} : {pynvml.nvmlDeviceGetName(handle)}")

        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"device {i} : mem total : {info.total // 1024**2} MB")
        print(f"device {i} : mem used  : {info.used // 1024**2} MB")
        print(f"device {i} : mem free  : {info.free // 1024**2} MB")
