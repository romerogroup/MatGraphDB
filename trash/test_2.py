
import torch

if torch.cuda.is_available():
    print("CUDA is available")
    if torch.cuda.device_count() > 0:
        print("GPU is available")
        # Add your code here to test torch geometric with GPU
    else:
        print("No GPU available")
else:
    print("CUDA is not available")