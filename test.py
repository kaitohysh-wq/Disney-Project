import os
import torch

# Force Windows to see the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check again
print(f"Checking GPU one last time... {torch.cuda.is_available()}")