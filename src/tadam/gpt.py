from dataclasses import dataclass

from tinygrad import Tensor, nn

from tadam.utils import compute_rope_cache, load_state_dict, apply_rope

__all__ = ["GPTConfig", "GPT"]


@dataclass
class GPTConfig:
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


class MultiHeadAttention:
    def __init__(self, config: GPTConfig):
        n_embd = config.n_embd
        self.n_head = config.n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x: Tensor, rope_cache: Tensor):
        B, C, E = x.shape  # batch, ctx_len, n_embd

        q, k, v = self.c_attn(x).split(E, dim=2)  # (B, C, E)
        k = k.rearrange("B C (H D) -> B H C D", H=self.n_head)
        q = q.rearrange("B C (H D) -> B H C D", H=self.n_head)
        v = v.rearrange("B C (H D) -> B H C D", H=self.n_head)

        q, k = apply_rope(q, k, rope_cache)
        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = y.rearrange("B H C D -> B C (H D)")

        y = self.c_proj(y)
        return y


class MLP:
    def __init__(self, config: GPTConfig):
        n_embd = config.n_embd
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x: Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())


class Block:
    def __init__(self, config: GPTConfig):
        n_embd = config.n_embd
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(config)

    def __call__(self, x: Tensor, rope_cache: Tensor):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT:
    def __init__(self, config: GPTConfig, weights_path: str | None = None):
        self.vocab_size, self.block_size, self.head_size = config.vocab_size, config.block_size, config.head_size
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        # self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        # weight tying (https://paperswithcode.com/method/weight-tying)
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
        x = self.ln_f(x)
        logits = self.lm_head(x)[:, :, : self.vocab_size]  # B, C, V
        return logits

    def generate(self, ctx: Tensor, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -self.block_size :])  # B, C, V
            logits = logits[:, -1, :] / temperature
            next_tok = logits.softmax().multinomial()
            ctx = Tensor.cat(ctx, next_tok, dim=1)
        return ctx
