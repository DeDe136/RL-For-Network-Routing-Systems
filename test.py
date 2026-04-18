# x = [1, 2, 4, 9]
# print (x[-2])

import torch
print("PyTorch version :", torch.__version__)
print("XPU available   :", hasattr(torch, 'xpu') and torch.xpu.is_available())
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print("GPU name        :", torch.xpu.get_device_name(0))
print("CUDA available  :", torch.cuda.is_available())