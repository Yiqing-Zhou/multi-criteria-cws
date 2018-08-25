import torch


def to_cuda_if_available(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def tensor(data, dtype=None, device=None, requires_grad=False):
    t = torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
    return to_cuda_if_available(t)


def zeros(*sizes, out=None):
    z = torch.zeros(sizes, out=out)
    return to_cuda_if_available(z)


def ones(*sizes, out=None):
    o = torch.ones(sizes, out=out)
    return to_cuda_if_available(o)


def full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    f = torch.full(size, fill_value, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    return to_cuda_if_available(f)


def randn(*sizes, out=None):
    r = torch.randn(sizes, out=out)
    return to_cuda_if_available(r)
