# Estimate GPU memory usage in CPU!

Run in cpu, and get a rough estimate for GPU memory usage!

```python
import torch

def get_tensors_saved_for_backward(net, output, duplicate=True):
    params = set(net.parameters())
    saved_tensors = dict() # saved_tensor --> operator name
    queue = [output.grad_fn]
    visited_grad_fn = set()
    while queue:
        new_queue = []
        for grad_fn in queue:
            if grad_fn in visited_grad_fn:
                continue
            visited_grad_fn.add(grad_fn)
            names =[k for k in dir(grad_fn) if k.startswith('_saved')] # names of saved tensors
            for k in names:
                v = getattr(grad_fn, k)
                if isinstance(v, torch.Tensor) and v not in params and v not in saved_tensors:
                    saved_tensors[v] = grad_fn.name()
            new_queue += [next_function for next_function, _ in grad_fn.next_functions if next_function is not None and next_function not in visited_grad_fn]
        queue = new_queue

    if not duplicate:
      return saved_tensors

    # remove duplicated tensors created by views/slicing/in-place operations
    deduplicated = dict() # ptr --> [tensor, operator_name]
    for tensor, operator_name in saved_tensors.items():
        ptr = tensor.storage().data_ptr()
        if ptr in deduplicated:
            value, old_operator_name = deduplicated[ptr]
            if value.numel() < tensor.numel():
                deduplicated[ptr] = [tensor, operator_name]
        else:
            deduplicated[ptr] = [tensor, operator_name]
    return deduplicated

from torchvision.models.resnet import resnet50

device = torch.device('cpu')

if device == torch.device('cpu'):
    net = resnet50().to(device)
    input = torch.rand(16, 3, 224, 224).to(device)
    output = net(input).sum() # output is the starting point of the backward computation graph
    deduplicated = get_tensors_saved_for_backward(net, output)
    saved_memory = sum([tensor.element_size() * tensor.numel() for ptr, [tensor, operator_name] in deduplicated.items()])
    print(f'estimated memory usage for backward related tensors in cpu: {saved_memory / 1024 ** 3} GB')
else:
    net = resnet50().to(device)
    input = torch.rand(16, 3, 224, 224).to(device)
    current_memory = torch.cuda.memory_allocated()
    output = net(input).sum() # output is the starting point of the backward computation graph
    import gc
    gc.collect()
    saved_memory = torch.cuda.memory_allocated() - current_memory
    print(f'ground-truth memory usage for backward related tensors in gpu: {saved_memory / 1024 ** 3} GB')
```
