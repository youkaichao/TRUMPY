import torch

def get_tensors_saved_for_backward(net, loss, duplicate=True):
    params = set(net.parameters())
    saved_tensors = dict() # saved_tensor --> operator name
    queue = [loss.grad_fn]
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

def memory_for_backward(net, loss):
    deduplicated = get_tensors_saved_for_backward(net, loss)
    saved_memory = sum([tensor.element_size() * tensor.numel() for ptr, [tensor, operator_name] in deduplicated.items()])
    return saved_memory
