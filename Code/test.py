import torch
import torchvision
import torchaudio
import cv2



# 打印版本号
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)
print("OpenCV version:", cv2.__version__)

# 检查设备支持
print("CUDA available:", torch.cuda.is_available())
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) available:", torch.backends.mps.is_available())
