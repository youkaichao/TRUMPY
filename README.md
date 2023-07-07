# TRUMPY: Tracing and Reverse Understanding Memory in Pytorch

Analyze how much memory is used for calculating the backward propagation (excluding parameters and gradients). This function works by tracing the computation graph used for backward propagation.

Training neural networks in pytorch requires the following memory footprint:

- Pytorch cuda context 
- Model states (parameters and buffers)
- Optimizer states (momentum etc.)
- Intermediate tensors saved during forward calculation, in order to calculate backward propagation.

While the memory footprint of cuda context / model states / optimizer states are easy to analyze, the memory footprint of intermediate tensors saved during forward calculation is very vague.

This package gathers these tensors, and report their memory footprint.

# Install

`pip install trumpy` or clone this repository and build it yourself.

# Usage

The core function is just one-line: `from trumpy import memory_for_backward; saved_memory = memory_for_backward(net, loss)`. You just pass the network and the calculated loss to this function, and it will give you how much memory (in bytes) is needed for backward propagation.

Example usage is in `tests/test.py`.

After turning on the GPU (setting `device = torch.device('cuda:0')`):
```
estimated memory usage for backward related tensors in cuda:0: 1.2881765365600586 GB
ground-truth memory usage for backward related tensors in gpu: 1.28438138961792 GB
```

You see that the estimated memory usage in `CPU` is very similar with the actual memory usage in `GPU`!
