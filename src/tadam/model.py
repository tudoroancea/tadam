import functools
import math
from dataclasses import dataclass

from tinygrad import Tensor, dtypes, nn

from tadam.utils import normalize

__all__ = ["GPTConfig", "GPT"]


@dataclass
class GPTConfig:
    ngpt: bool = False
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    @property
    def padded_vocab_size(self):
        return 64 * (self.vocab_size // 64 + 1)

    @property
    def head_size(self):
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    @property
    def base_scale(self):
        return 1 / (self.n_embd**0.5)


class Scale:
    def __init__(self, dim: int, init: float, scale: float) -> None:
        self.scale = Tensor.full(dim, scale)
        self.forward_scale = init / scale

    def __call__(self) -> Tensor:
        return self.scale * self.forward_scale


class Linear:
    """Linear layer that is optionally normalized wrt"""

    def __init__(self, in_features: int, out_features: int, config: GPTConfig, init_std: float | None = None):
        if init_std is None:
            init_std = config.base_scale
        self.weight = Tensor.normal(out_features, in_features, mean=0.0, std=init_std)
        if config.ngpt:
            self.weight = normalize(self.weight)
            # Set special attribute to indicate the weights are supposed to be normalized after each optimization step.
            self.weight.__normalized__ = True
        else:
            # Set special attribute to indicate we want to weight decay this parameter
            self.weight.__wd__ = True

    def __call__(self, x: Tensor) -> Tensor:
        # (..., in_features) x (in_features, out_features) -> (..., out_features)
        return x.matmul(self.weight.transpose())


class Embedding:
    def __init__(self, vocab_size: int, embed_size: int, config: GPTConfig):
        self.vocab_sz = vocab_size
        self.embed_sz = embed_size
        self.weight = Tensor.normal(vocab_size, embed_size, mean=0.0, std=config.base_scale)
        if config.ngpt:
            self.weight = normalize(self.weight)
            # Set special attribute to indicate the weights are supposed to be normalized after each optimization step.
            self.weight.__normalized__ = True
        else:
            # to indicate we want to weight decay this parameter
            self.weight.__wd__ = True
        self.arange = Tensor.arange(self.vocab_sz, requires_grad=False).reshape(self.vocab_sz, 1)

    def __call__(self, idx: Tensor) -> Tensor:
        big_shp = idx.shape + (self.vocab_sz, self.embed_sz)
        arange = self.arange.expand(big_shp)
        idx = idx.reshape(idx.shape + (1, 1)).expand(big_shp)
        vals = self.weight.expand(big_shp)
        return (arange == idx).mul(vals).sum(-2)


#  https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def compute_rope_cache(pos: Tensor, dim: int, theta: int = 10000) -> Tensor:
    """Compute cos and sin freqs for RoPE."""
    assert dim % 2 == 0
    scale = Tensor.arange(0, dim, 2) / dim  # d = D/2
    omega = 1.0 / (theta**scale)
    # pos = Tensor.arange(C)
    out = Tensor.einsum("n,d->nd", pos, omega)  # could be accomplished with simple broadcasting
    return Tensor.stack(out.cos(), out.sin(), dim=-1)  # C,D/2,2


def apply_rope(q: Tensor, k: Tensor, rope_cache: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE rotation to q and k (which have shape (B,H,C,D))"""
    qshaped = q.reshape(*q.shape[:-1], -1, 2)  # B,H,C,D/2,2
    kshaped = k.reshape(*k.shape[:-1], -1, 2)  # B,H,C,D/2,2
    rope_cache = rope_cache.reshape(1, 1, *rope_cache.shape)  # 1,1,C,D/2,2
    q_out = Tensor.stack(
        qshaped[..., 0] * rope_cache[..., 0] - qshaped[..., 1] * rope_cache[..., 1],
        qshaped[..., 0] * rope_cache[..., 1] + qshaped[..., 1] * rope_cache[..., 0],
        dim=-1,
    )
    k_out = Tensor.stack(
        kshaped[..., 0] * rope_cache[..., 0] - kshaped[..., 1] * rope_cache[..., 1],
        kshaped[..., 0] * rope_cache[..., 1] + kshaped[..., 1] * rope_cache[..., 0],
        dim=-1,
    )
    return q_out.reshape(*q.shape), k_out.reshape(*k.shape)


class MultiHeadAttention:
    def __init__(self, config: GPTConfig):
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, config)
        # output projection
        self.c_proj = Linear(
            config.n_embd, config.n_embd, config, init_std=config.base_scale / math.sqrt(2 * config.n_layer)
        )
        # query and key scaling
        if config.ngpt:
            self.s_qk = Scale(config.n_embd, init=1.0, scale=config.base_scale)
        # attention causal mask
        self.causal_mask = Tensor.ones(config.block_size, config.block_size).triu(1)
        self.causal_mask.requires_grad = False

    def __call__(self, x: Tensor, rope_cache: Tensor):
        _, C, E, D = *x.shape, self.config.head_size  # batch, ctx_len, n_embd, head_size

        q, k, v = self.c_attn(x).split(E, dim=2)  # (B, C, E)
        k = k.rearrange("B C (H D) -> B H C D", D=D)
        q = q.rearrange("B C (H D) -> B H C D", D=D)
        v = v.rearrange("B C (H D) -> B H C D", D=D)

        q, k = apply_rope(q, k, rope_cache)
        if self.config.ngpt:
            s_qk = self.s_qk().rearrange("(H D) -> 1 H 1 D", D=D)
            # shouldn't change anything to remove them (cf. Table 6 in Annex 8)
            q = q * s_qk
            k = k * s_qk

        softmax_scale = math.sqrt(E * self.config.n_head) if self.config.ngpt else 1 / math.sqrt(D)
        att = (q @ k.transpose(-2, -1)) * softmax_scale  # (B, H, C, C)
        att = att.masked_fill(self.causal_mask[:C, :C], -float("inf"))
        att = att.softmax()
        y = att @ v
        y = y.rearrange("B H C D -> B C (H D)")

        return self.c_proj(y)


class MLP:
    def __init__(self, config: GPTConfig):
        self.config = config
        n_embd = config.n_embd
        self.c_fc = Linear(n_embd, 2 * 4 * n_embd, config)
        # apply special scaled init to the residual projections, per GPT-2 paper
        self.c_proj = Linear(4 * n_embd, n_embd, config, init_std=config.base_scale / math.sqrt(2 * config.n_layer))
        if self.config.ngpt:
            self.s_uv = Scale(2 * 4 * n_embd, init=1.0, scale=1.0)

    def __call__(self, x: Tensor) -> Tensor:
        uv = self.c_fc(x)
        if self.config.ngpt:
            uv = uv * self.s_uv().view(1, 1, -1)
        u, v = uv.split(uv.shape[-1] // 2, dim=-1)
        return self.c_proj(u * v.silu())


class Block:
    def __init__(self, config: GPTConfig):
        self.config = config
        # attention and MLP blocks
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        if config.ngpt:
            # eigen learning rates
            self.alpha_attn = Scale(config.n_embd, init=1 / config.n_layer, scale=config.base_scale)
            self.alpha_mlp = Scale(config.n_embd, init=1 / config.n_layer, scale=config.base_scale)
        else:
            # layer normalization
            self.ln_1 = nn.RMSNorm(config.n_embd)
            self.ln_2 = nn.RMSNorm(config.n_embd)

    def __call__(self, x: Tensor, rope_cache: Tensor):
        # x has shape (B, C, E)
        if self.config.ngpt:
            # LERP between x and attn(x):  x + alpha * (attn(x) - x) = (1 - alpha) * x + alpha * attn(x)
            x = normalize(x + self.alpha_attn() * (normalize(self.attn(x, rope_cache)) - x))
            x = normalize(x + self.alpha_mlp() * (normalize(self.mlp(x)) - x))
        else:
            x = x + self.attn(self.ln_1(x), rope_cache)
            x = x + self.mlp(self.ln_2(x))
        return x


class GPT:
    def __init__(self, config: GPTConfig, weights_path: str | None = None):
        """Normalized GPT model, as described in https://arxiv.org/abs/2410.01131

        Args:
            config: NGPTConfig object containing the model configuration
            weights_path: path to the weights file to load the model from
        """
        self.config = config
        # self.vocab_size, self.block_size, self.head_size = config.vocab_size, config.block_size, config.head_size
        self.wte = Embedding(config.padded_vocab_size, config.n_embd, config)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.lm_head = Linear(config.n_embd, config.padded_vocab_size, config)
        if config.ngpt:
            self.s_z = Scale(config.padded_vocab_size, init=1.0, scale=config.base_scale)
        else:
            self.ln_f = nn.RMSNorm(config.n_embd)
        # weight tying (https://paperswithcode.com/method/weight-tying)
        assert self.wte.weight.shape == self.lm_head.weight.shape
        self.wte.weight = self.lm_head.weight

        # pre-compute rope cache for block_size
        self.rope_cache = compute_rope_cache(Tensor.arange(config.block_size), config.head_size)
        self.rope_cache.requires_grad = False

        # load weights
        if weights_path is not None:
            nn.state.load_state_dict(self, nn.state.safe_load(weights_path))

    def __call__(self, idx: Tensor, eval: bool = False) -> Tensor:
        _, C = idx.shape
        # token embeddings
        tok_emb = self.wte(idx)  # B, C, E
        # crop RoPE cache to context length
        rope_cache = self.rope_cache[:C, ...]
        x = tok_emb.sequential(functools.partial(layer, rope_cache=rope_cache) for layer in self.h)  # B,C,E
        if not self.config.ngpt:
            x = self.ln_f(x)
        logits = self.lm_head(x) if not eval else self.lm_head(x[:, [-1], :])  # B,C,V or B,1,V
        if self.config.ngpt:
            logits = logits * self.s_z()
        logits = logits[:, :, : self.config.vocab_size]  # B,C,V or B,1,V
        return logits

    def generate(self, ctx: Tensor, max_new_tokens: int, temperature: float = 1.0):
        # ctx has shape (B, C)
        B, C = ctx.shape
        ctx = ctx.cat(Tensor.full((B, max_new_tokens), GPTConfig.vocab_size - 1, dtype=dtypes.uint8), dim=-1)
        probs = Tensor.rand(B, max_new_tokens, self.config.vocab_size, contiguous=True)
        for i in range(C, C + max_new_tokens):
            # run model to obtain logits
            logits = self(ctx[:, max(0, i - self.config.block_size) : i], True)  # B, C, V
            # compute probabilities for each possible token
            probs[:, i - C] = (logits[:, -1, :] / temperature).softmax()  # B, V
            # sample next token from the probabilities
            ctx[:, i : i + 1] = probs[:, i - C].multinomial()  # B, 1
        return ctx[:, C:], probs
