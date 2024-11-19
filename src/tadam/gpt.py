from dataclasses import dataclass
from tinygrad import Tensor, nn
from tadam.utils import load_state_dict

__all__ = ["GPTConfig", "GPT"]


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MultiHeadAttention:
    def __init__(self, config: GPTConfig):
        n_embd = config.n_embd
        self.n_head = n_head = config.n_head
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x: Tensor):
        B, T, C = x.shape  # batch, ctx_len, n_embd
        q, k, v = self.c_attn(x).split(C, dim=2)  # (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # optimized implementation of attention
        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = y.view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
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

    def __call__(self, x: Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT:
    def __init__(self, config: GPTConfig, weights_path: str | None = None):
        self.vocab_size, self.block_size = config.vocab_size, config.block_size
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        # weight tying (https://paperswithcode.com/method/weight-tying)
        self.wte.weight = self.lm_head.weight
        # load weights
        if weights_path is not None:
            load_state_dict(self, nn.state.safe_load(weights_path))

    def __call__(self, idx: Tensor):
        B, T = idx.shape
        pos = Tensor.arange(0, T)
        tok_emb = self.wte(idx)  # token embeddings of shape (B, T, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (T, n_embd)
        x = tok_emb + pos_emb
        x = self.ln_f(x.sequential(self.h))
        logits = self.lm_head(x)[:, :, : self.vocab_size]
        return logits

    def generate(self, ctx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -self.block_size :])
            logits = logits[:, -1, :] / temperature
            next_tok = logits.softmax().multinomial()
            ctx = Tensor.cat(ctx, next_tok, dim=1)
        return ctx
