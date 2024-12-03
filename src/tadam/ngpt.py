import math
from dataclasses import dataclass

from tinygrad import Tensor, nn

from tadam.utils import apply_rope, compute_rope_cache, load_state_dict, norm, normalize

__all__ = ["NGPTConfig", "NGPT", "norm", "normalize"]


@dataclass
class NGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    @property
    def head_size(self):
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    alpha_init: float = 0.05  # ≈ 1/n_layer
    alpha_scale: float = 1.0 / math.sqrt(768)  # 1/sqrt(n_embd)
    s_qk_init: float = 1.0
    s_qk_scale: float = 1.0 / math.sqrt(768)  # 1/sqrt(n_embd)
    s_u_init: float = 1.0
    s_u_scale: float = 1.0
    s_z_init: float = 1.0
    s_z_scale: float = 1.0 / math.sqrt(768)  # 1/sqrt(n_embd)


class NormalizedLinear:
    def __init__(self, in_features: int, out_features: int):
        bound = 1 / math.sqrt(in_features)
        self.weight = normalize(Tensor.uniform(out_features, in_features, low=-bound, high=bound))
        # Set special attribute to indicate the weights are supposed to be normalized after each optimization step.
        self.weight.__normalized__ = True

    def __call__(self, x: Tensor) -> Tensor:
        # (B, T, Cin) x (Cin, Cout) -> (B, T, Cout)
        return x.matmul(self.weight.transpose())


class Scale:
    def __init__(self, dim: int, init: float, scale: float) -> None:
        self.scale = Tensor.full(dim, scale)
        self.forward_scale = init / scale

    def __call__(self) -> Tensor:
        return self.scale * self.forward_scale


class MultiHeadAttention:
    def __init__(self, config: NGPTConfig):
        self.head_size = config.head_size
        # key, query, value projections for all heads, but in a batch
        self.c_attn = NormalizedLinear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = NormalizedLinear(config.n_embd, config.n_embd)
        # query and key scaling
        # TODO: check again the scaling factors
        self.s_qk = Scale(config.head_size, init=1.0, scale=1.0 / math.sqrt(config.head_size))
        # causal mask
        self.causal_mask = Tensor.ones(config.block_size, config.block_size).triu(1)
        self.causal_mask.requires_grad = False

    def __call__(self, x: Tensor, rope_cache: Tensor):
        B, C, E, D = *x.shape, self.head_size  # batch, ctx_len, n_embd, head_size

        q, k, v = self.c_attn(x).split(E, dim=2)  # (B, C, E)
        k = k.rearrange("B C (H D) -> B H C D", D=self.head_size)
        q = q.rearrange("B C (H D) -> B H C D", D=self.head_size)
        v = v.rearrange("B C (H D) -> B H C D", D=self.head_size)

        q, k = apply_rope(q, k, rope_cache)
        q = normalize(q) * self.s_qk()
        k = normalize(k) * self.s_qk()

        att = (q @ k.transpose(-2, -1)) * math.sqrt(D)  # (B, H, C, C)
        att = att.masked_fill(self.causal_mask[:C, :C], -float("inf"))
        att = att.softmax()
        y = att @ v
        y = y.rearrange("B H C D -> B C (H D)")

        y = self.c_proj(y)
        return y


class MLP:
    def __init__(self, config: NGPTConfig):
        n_embd = config.n_embd
        self.c_fc = NormalizedLinear(n_embd, 4 * n_embd)
        self.s_u = Scale(4 * n_embd, init=1.0, scale=1.0 / math.sqrt(4 * n_embd))
        self.c_proj = NormalizedLinear(4 * n_embd, n_embd)

    def __call__(self, x: Tensor) -> Tensor:
        return self.c_proj((self.c_fc(x) * self.s_u()).gelu())


class Block:
    def __init__(self, config: NGPTConfig):
        # attention and MLP blocks
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        # eigen learning rates
        self.alpha_attn = Scale(config.n_embd, init=1.0, scale=1.0 / math.sqrt(config.n_embd))
        self.alpha_mlp = Scale(config.n_embd, init=1.0, scale=1.0 / math.sqrt(config.n_embd))

    def __call__(self, x: Tensor, rope_cache: Tensor):
        x = normalize(x + self.alpha_attn() * (normalize(self.attn(x, rope_cache)) - x))
        x = normalize(x + self.alpha_mlp() * (normalize(self.mlp(x)) - x))
        return x


class Embedding:
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_sz = vocab_size
        self.embed_sz = embed_size
        self.weight = normalize(Tensor.glorot_uniform(vocab_size, embed_size))
        # Set special attribute to indicate the weights are supposed to be normalized after each optimization step.
        self.weight.__normalized__ = True
        self.arange = Tensor.arange(self.vocab_sz, requires_grad=False).reshape(self.vocab_sz, 1)

    def __call__(self, idx: Tensor) -> Tensor:
        big_shp = idx.shape + (self.vocab_sz, self.embed_sz)
        arange = self.arange.expand(big_shp)
        idx = idx.reshape(idx.shape + (1, 1)).expand(big_shp)
        vals = self.weight.expand(big_shp)
        return (arange == idx).mul(vals).sum(-2)


class NGPT:
    def __init__(self, config: NGPTConfig, weights_path: str | None = None):
        """Normalized GPT model, as described in https://arxiv.org/abs/2410.01131

        Args:
            config: NGPTConfig object containing the model configuration
            weights_path: path to the weights file to load the model from
        """
        self.vocab_size, self.block_size, self.head_size = config.vocab_size, config.block_size, config.head_size
        self.wte = Embedding(config.padded_vocab_size, config.n_embd)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.lm_head = NormalizedLinear(config.n_embd, config.padded_vocab_size)
        self.s_z = Scale(
            config.padded_vocab_size,
            init=1.0,
            scale=1.0 / math.sqrt(config.padded_vocab_size),
        )
        # weight tying (https://paperswithcode.com/method/weight-tying)
        assert self.wte.weight.shape == self.lm_head.weight.shape
        self.wte.weight = self.lm_head.weight
        # load weights
        if weights_path is not None:
            load_state_dict(self, nn.state.safe_load(weights_path))

    def __call__(self, idx: Tensor):
        B, C, D = *idx.shape, self.head_size
        tok_emb = self.wte(idx)  # B, C, E
        rope_cache = compute_rope_cache(Tensor.arange(C), D)  # C, D/2, 2
        x = tok_emb
        for layer in self.h:
            x = layer(x, rope_cache)  # B, C, E
        logits = self.lm_head(x) * self.s_z()
        logits = logits[:, :, : self.vocab_size]  # B, C, V
        return logits

    def generate(self, ctx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -self.block_size :])  # B, C, V
            logits = logits[:, -1, :] / temperature
            next_tok = logits.softmax().multinomial()
            ctx = Tensor.cat(ctx, next_tok, dim=1)
        return ctx
