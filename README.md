# TRUMPY: Tracing and Reverse Understanding Memory in Pytorch

Analyze how much memory is used for calculating the backward propagation (excluding parameters and gradients).

The fascinating thing is that it works in both GPU and CPU. Therefore, you can first run the code in CPU to estimate the memory usage, which will then be a decent estimation of memory consumption in GPU!

Example usage is in `tests/test.py`.


