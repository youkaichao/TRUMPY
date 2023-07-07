from trumpy.lib import get_tensors_saved_for_backward, memory_for_backward

import torch
from torchvision.models.resnet import resnet50
device = torch.device('cpu')
net = resnet50().to(device)
input = torch.rand(16, 3, 224, 224).to(device)

if device != torch.device('cpu'):
    current_memory = torch.cuda.memory_allocated()

loss = net(input).sum() # loss is the starting point of the backward computation graph
saved_memory = memory_for_backward(net, loss)
print(f'estimated memory usage for backward related tensors in cpu: {saved_memory / 1024 ** 3} GB')

if device != torch.device('cpu'):
    import gc
    gc.collect()
    saved_memory = torch.cuda.memory_allocated() - current_memory
    print(f'ground-truth memory usage for backward related tensors in gpu: {saved_memory / 1024 ** 3} GB')