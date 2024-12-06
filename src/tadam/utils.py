from tinygrad import Tensor


__all__ = ["norm", "normalize"]


def norm(x: Tensor, axis=-1) -> Tensor:
    """x (..., n) -> ||x|| (..., 1)"""
    return x.square().sum(-1, keepdim=True).sqrt()


def normalize(x: Tensor, axis=-1) -> Tensor:
    return x * x.square().sum(axis, keepdim=True).rsqrt()
