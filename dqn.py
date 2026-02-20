import torch
from torch import nn
import torch.nn.functional as F
print(torch.print("CUDA" if torch.cuda.is_available() else "CPU"))