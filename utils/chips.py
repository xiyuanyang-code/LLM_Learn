import os
import torch

# setting cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


print(f"Visible GPUs：{torch.cuda.device_count()}")
print(f"current index of GPUs：{torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")
print(f"Device information: {torch.cuda.get_device_properties()}")