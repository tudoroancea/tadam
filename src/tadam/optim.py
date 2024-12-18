from tinygrad import Tensor, nn
from tadam.utils import normalize


class SkewSymmetricRepresentantion:
    def __init__(self, x: Tensor, u: Tensor):
        self.x = x  # (..., n)
        self.u = u  # (..., n)

    def __mul__(self, other: float) -> "SkewSymmetricRepresentantion":
        return SkewSymmetricRepresentantion(self.x, self.u * other)

    __rmul__ = __mul__

    def __imul__(self, other: float) -> "SkewSymmetricRepresentantion":
        self.u.assign(self.u * other)
        return self

    __irmul__ = __imul__

    def mul(self, y: Tensor, is_x: bool = False) -> Tensor:
        first = self.u if is_x else self.u * (self.x * self.y).sum(-1, keepdim=True)
        return first - self.x * (self.u * self.y).sum(-1, keepdim=True)

    def norm(self):
        raise NotImplementedError("TODO: implement norm")


def project(x: Tensor, u: Tensor):
    """Projects a vector u onto the tangent space at x.
    Complexity: ≈4n flops where
    u has shape (..., n)
    x has shape (..., n)
    """
    return u - x * (u * x).sum(-1, keepdim=True)


class Adam(nn.optim.Optimizer):
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
            x = p.detach()
            egrad = p.grad
            if hasattr(p, "__wd__") and self.wd != 0:
                egrad = egrad + self.wd * x
            self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * egrad)
            self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * egrad.square())
            eps_hat = self.eps * (1 - self.b2_t).sqrt()
            descent_dir = self.m[i] / (self.v[i].sqrt() + eps_hat)
            step_size = self.lr * (1.0 - self.b2_t).sqrt() / (1.0 - self.b1_t)
            if hasattr(p, "__normalized__"):
                p.assign(normalize(x - project(x, step_size * descent_dir)))
            else:
                p.assign(x - step_size * descent_dir)

        return [self.b1_t, self.b2_t, descent_dir, step_size, eps_hat] + self.m + self.v


class TADAM(nn.optim.Optimizer):
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
            Tensor.ones((1,), dtype=params[0].dtype, device=self.device, requires_grad=False).contiguous()
            for _ in [beta1, beta2]
        )
        # first moments
        self.m = [
            Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous()
            for t in self.params
        ]
        # second moments
        # we assume the weight has shape (nout, nin) or (n)
        self.v = [
            # Tensor.zeros(t.shape[0], dtype=t.dtype, device=t.device, requires_grad=False).contiguous()
            Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous()
            for t in self.params
        ]

    def _step(self) -> list[Tensor]:
        self.b1_t *= self.b1
        self.b2_t *= self.b2
        for i, p in enumerate(self.params):
            x = p.detach()  # important because we can assign to p only tensors with requires_grad=False
            # compute euclidean and riemannian gradients
            egrad = p.grad
            if hasattr(p, "__wd__") and self.wd != 0:
                egrad = egrad + self.wd * x

            if hasattr(p, "__normalized__"):
                rgrad = project(x, egrad)
                self.m[i].assign(self.b1 * project(x, self.m[i]) + (1.0 - self.b1) * rgrad)
                self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * rgrad.square())
                step_size = self.lr * (1.0 - self.b2_t).sqrt() / (1.0 - self.b1_t)
                p.assign(normalize(x - step_size * self.m[i] * (self.v[i] + self.eps).rsqrt()))
            else:
                self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * egrad)
                self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * egrad.square())
                step_size = self.lr * (1.0 - self.b2_t).sqrt() / (1.0 - self.b1_t)
                p.assign(x - step_size * self.m[i] * (self.v[i] + self.eps).rsqrt())

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
