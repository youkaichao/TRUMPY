# TRUMPY: Tracing and Reverse Understanding Memory in Pytorch

Analyze how much memory is used for calculating the backward propagation (excluding parameters and gradients). This function works by tracing the computation graph used for backward propagation.

The fascinating thing is that it works in both GPU and CPU. Therefore, you can **first run the code in CPU to estimate the memory usage, which will then be a decent estimation of memory consumption in GPU**!

# Install

`pip install trumpy` or clone this repository and build it yourself.

# Usage

The core function is just one-line: `from trumpy import memory_for_backward; saved_memory = memory_for_backward(net, loss)`. You just pass the network and the calculated loss to this function, and it will give you how much memory (in bytes) is needed for backward propagation.

Example usage is in `tests/test.py`.

```python
import torch
from torchvision.models.resnet import resnet50

# define device and models
device = torch.device('cpu')
net = resnet50().to(device)
input = torch.rand(16, 3, 224, 224).to(device)

# if gpu is available, record current memory
if device != torch.device('cpu'):
    current_memory = torch.cuda.memory_allocated()

# define loss function
loss = net(input).sum() # loss is the starting point of the backward computation graph
from trumpy import memory_for_backward
# calculate the memory used for backward
saved_memory = memory_for_backward(net, loss)
print(f'estimated memory usage for backward related tensors in {device}: {saved_memory / 1024 ** 3} GB')

# check how much memory pytorch holds
if device != torch.device('cpu'):
    import gc
    gc.collect()
    saved_memory = torch.cuda.memory_allocated() - current_memory
    print(f'ground-truth memory usage for backward related tensors in gpu: {saved_memory / 1024 ** 3} GB')
```

You should see something like:
```
estimated memory usage for backward related tensors in cpu: 1.2881765365600586 GB
```

After turning on the GPU (setting `device = torch.device('cuda:0')`):
```
estimated memory usage for backward related tensors in cuda:0: 1.2881765365600586 GB
ground-truth memory usage for backward related tensors in gpu: 1.28438138961792 GB
```

You see that the estimated memory usage in `CPU` is very similar with the actual memory usage in `GPU`!
