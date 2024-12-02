import math
from dataclasses import dataclass
from tinygrad import Tensor, nn
from tadam.utils import norm, normalize, load_state_dict

__all__ = ["NGPTConfig", "NGPT", "norm", "normalize"]


@dataclass
class NGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
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
        n_embd = config.n_embd
        self.n_head = n_head = config.n_head
        self.block_size = config.block_size
        assert n_embd % n_head == 0
        self.head_size = head_size = n_embd // n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = NormalizedLinear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = NormalizedLinear(n_embd, n_embd)
        # query and key scaling
        self.s_qk = Scale(head_size, init=1.0, scale=1.0 / math.sqrt(head_size))
        # causal mask
        self.causal_mask = Tensor.ones(self.block_size, self.block_size).triu(1)
        self.causal_mask.requires_grad = False

    def __call__(self, x: Tensor):
        B, T, C = x.shape  # batch, ctx_len, n_embd

        # query, key, value projections for all heads, but in a batch
        q, k, v = self.c_attn(x).split(C, dim=2)  # (B, T, C)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        # normalize and scale query and key
        q = normalize(q) * self.s_qk()
        k = normalize(k) * self.s_qk()

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * math.sqrt(C)  # (B, nh, T, T)
        att = att.masked_fill(self.causal_mask[:T, :T], -float("inf"))
        att = att.softmax()
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
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

    def __call__(self, x: Tensor):
        x = normalize(x + self.alpha_attn() * (normalize(self.attn(x)) - x))
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
        self.vocab_size, self.block_size = config.vocab_size, config.block_size
        self.wte = Embedding(config.padded_vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
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
        B, T = idx.shape
        pos = Tensor.arange(0, T)
        tok_emb = self.wte(idx)  # token embeddings of shape (B, T, n_embd)
        # TODO: implement RoPE
        pos_emb = self.wpe(pos)  # position embeddings of shape (T, n_embd)
        x: Tensor = tok_emb + pos_emb
        x = x.sequential(self.h)
        logits = self.lm_head(x) * self.s_z()
        logits = logits[:, :, : self.vocab_size]
        return logits

    def generate(self, ctx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -self.block_size :])
            logits = logits[:, -1, :] / temperature
            next_tok = logits.softmax().multinomial()
            ctx = Tensor.cat(ctx, next_tok, dim=1)
        return ctx
