from tinygrad import Tensor, nn
from tadam.utils import normalize

__all__ = ["GenericAdam", "CayleyAdam"]


class SkewSymmetricRepresentantion:
    def __init__(self, x: Tensor, u: Tensor):
        self.x = x  # (..., n)
        self.u = u  # (..., n)

    def __mul__(self, other: float):
        return SkewSymmetricRepresentantion(self.x, self.u * other)

    __rmul__ = __mul__

    def mul(self, y: Tensor, is_x: bool = False):
        first = self.u if is_x else self.u * (self.x * self.y).sum(-1, keepdims=True)
        return first - self.x * (self.u * self.y).sum(-1, keepdims=True)

    def norm(self):
        raise NotImplementedError("TODO: implement norm")


def project(x: Tensor, u: Tensor):
    """Projects a vector u onto the tangent space of x.
    Complexity: ≈4n flops where n=x.numel()
    """
    return x - u * (x * u).sum(-1, keepdims=True)


class GenericAdam(nn.optim.Optimizer):
    """Very general Adam implementation that covers multiple simple cases.
    In particular, it supports:
    - Classical AdamW
    - Classical Adam (without weight decay)
    - Adam with normalization of weight matrices along their embedding dimension
      (as described in https://arxiv.org/abs/2410.01131)
    """

    def __init__(
        self,
        params: list[Tensor],
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, weight_decay
        # beta1^t and beta2^t
        self.b1_t, self.b2_t = (
            Tensor.ones(
                (1,),
                dtype=params[0].dtype,
                device=self.device,
                requires_grad=False,
            ).contiguous()
            for _ in [beta1, beta2]
        )
        # first moments
        self.m = [
            Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous()
            for t in self.params
        ]
        # second moments
        self.v = [
            Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous()
            for t in self.params
        ]

    def _step(self) -> list[Tensor]:
        self.b1_t *= self.b1
        self.b2_t *= self.b2
        for i, p in enumerate(self.params):
            assert p.grad is not None
            g = p.grad
            if self.wd != 0:
                g = g + self.wd * p.detach()
            self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * g)
            self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (g * g))
            up = self.m[i] / (self.v[i].sqrt() + self.eps)
            alpha = self.lr * (1.0 - self.b2_t).sqrt() / (1.0 - self.b1_t)
            new_p = p.detach() - alpha * up
            p.assign(normalize(new_p) if hasattr(p, "__normalized__") else new_p)

        return [self.b1_t, self.b2_t] + self.m + self.v


class IntermediateAdam(GenericAdam):
    def __init__(
        self,
        params: list[Tensor],
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        assert all([hasattr(p, "__normalized__") for p in params])
        super().__init__(params, lr)
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, weight_decay
        # beta1^t and beta2^t
        self.b1_t, self.b2_t = (
            Tensor.ones((1,), dtype=params[0].dtype, device=self.device, requires_grad=False).contiguous()
            for _ in [beta1, beta2]
        )
        # first moments
        self.m = [
            Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous()
            for t in self.params
        ]
        # second moments
        self.v = [
            Tensor.zeros(1, dtype=t.dtype, device=t.device, requires_grad=False).contiguous() for t in self.params
        ]

    def _step(self) -> list[Tensor]:
        self.b1_t *= self.b1
        self.b2_t *= self.b2
        for i, p in enumerate(self.params):
            x = p.detach()
            # compute euclidean and riemannian gradients
            euclidean_grad = p.grad.detach()
            if self.wd != 0:
                euclidean_grad = euclidean_grad + self.wd * x
            riemannian_grad = project(x, euclidean_grad)
            # accumulate first and second moments
            self.m[i].assign(self.b1 * project(x, self.m[i]) + (1.0 - self.b1) * riemannian_grad)
            self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * riemannian_grad.square().sum())
            # create descent direction
            step_size = self.lr * (1.0 - self.b2_t).sqrt() / (1.0 - self.b1_t) / (self.v[i].sqrt() + self.eps)
            # descent_direction = step_size * self.m[i]
            descent_direction = project(x, step_size * self.m[i])  # project for numerical stability
            # perform retraction (by simple normalization)
            p.assign((x - descent_direction) / (1 + descent_direction.square().sum()).sqrt())

        return [self.b1_t, self.b2_t] + self.m + self.v


class CayleyAdam(nn.optim.Optimizer):
    """Riemannian version of Adam based on Cayley retractions and vector transport.

    Adapted from the following paper describing the optimization scheme for the
    Stiefel manifold: https://arxiv.org/abs/2002.01113
    """

    def __init__(
        self,
        params: list[Tensor],
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        assert all([hasattr(p, "__normalized__") for p in params])
        super().__init__(params, lr)
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, weight_decay
        # beta1^t and beta2^t
        self.b1_t, self.b2_t = (
            Tensor.ones((1,), dtype=params[0].dtype, device=self.device, requires_grad=False).contiguous()
            for _ in [beta1, beta2]
        )
        # first moments
        self.m = [
            Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous()
            for t in self.params
        ]
        # second moments
        self.v = [
            Tensor.zeros(1, dtype=t.dtype, device=t.device, requires_grad=False).contiguous() for t in self.params
        ]

    def _step(self) -> list[Tensor]:
        self.b1_t *= self.b1
        self.b2_t *= self.b2
        for i, x in enumerate(self.params):
            assert x.grad is not None
            x.square()
            g = x.grad
            if self.wd != 0:
                g = g + self.wd * x.detach()
            # Update second moment
            # TODO: project gradient
            self.v[i].assign(self.b2 * self.v[i] + (1 - self.b2) * g.square().sum())
            s = (1 - self.b2_t) / (1 - self.b1_t) / (self.v[i].sqrt() + self.eps)
            # Update first moment and compute skew-symmetric representation
            M = SkewSymmetricRepresentantion(x, self.b1 * self.m[i] + (1 - self.b1) * g)
            self.m[i].assign(M.mul(x, is_x=True))
            # Compute final velocity (descent direction)
            W = s * M
            w = s * self.m[i]
            # Select step size
            # TODO: compute norm of W
            normW = W.square().sum().sqrt()
            alpha = min(self.lr, 1 / (normW + self.eps))
            # Approximate Cayley retraction
            y = x.detach() - alpha * w
            to_add = x.detach() - alpha / 2 * W.mul(x.detach(), is_x=True)
            for _ in range(2):
                y = to_add - alpha / 2 * W.mul(y)
            x.assign(y)

        return [self.b1_t, self.b2_t] + self.m + self.v
