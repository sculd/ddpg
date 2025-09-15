import torch

# Use cuda if available else use cpu
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
