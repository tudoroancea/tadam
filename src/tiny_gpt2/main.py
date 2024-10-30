import os
import time
import tiktoken

import numpy as np
from tinygrad import Device, Tensor, TinyJit, nn, GlobalCounters

### hyper-parameters
# model
block_size: int = 1024
vocab_size: int = 50257
padded_vocab_size: int = 50304
n_layer: int = 12
n_head: int = 12
n_embd: int = 768
# training
ctx_len: int = 8
batch_size: int = 64
num_epochs: int = 1
lr = 1e-4


class MultiHeadAttention:
    def __init__(self):
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x: Tensor):
        B, T, C = x.shape  # batch, ctx_len, n_embd
        qkv = self.c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

        # manual implementation of attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = att.softmax()
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).view(B, T, C)  # re-assemble all head outputs side by side

        # optimized implementation of attention
        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = y.view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP:
    def __init__(self):
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x: Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())


class Block:
    def __init__(self):
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def __call__(self, x: Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT:
    def __init__(self):
        self.wte = nn.Embedding(padded_vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = [Block() for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, padded_vocab_size, bias=False)
        # weight tying (https://paperswithcode.com/method/weight-tying)
        self.wte.weight = self.lm_head.weight

    @staticmethod
    def from_pretrained(path: str):
        model = GPT()
        nn.state.load_state_dict(model, nn.state.safe_load(path))
        return model

    def __call__(self, idx: Tensor):
        B, T = idx.shape
        pos = Tensor.arange(0, T)
        tok_emb = self.wte(idx)  # token embeddings of shape (B, T, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (T, n_embd)
        x = tok_emb + pos_emb
        x = self.ln_f(x.sequential(self.h))
        logits = self.lm_head(x)[:, :, :vocab_size]
        return logits

    def generate(self, ctx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            next_tok = logits.softmax().multinomial()
            ctx = Tensor.cat(ctx, next_tok, dim=1)
        return ctx


def get_batches(toks):
    """Lightweight dataloader"""
    i = 0
    while i + batch_size * ctx_len + 1 < len(toks):
        x = toks[i : i + batch_size * ctx_len].view(batch_size, ctx_len)
        y = toks[i + 1 : i + batch_size * ctx_len + 1].view(batch_size, ctx_len)
        yield x, y
        i += batch_size * ctx_len


class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.bos = self.encode("<|endoftext|>")[0]
        self.eos = self.encode("<|endoftext|>")[0]

    def encode(self, text: str):
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int] | Tensor):
        return self.enc.decode(tokens if isinstance(tokens, list) else tokens.tolist())


def load_tokens(data_path: str):
    assert os.path.isfile(data_path)
    with open(data_path, "rb") as f:
        f.seek(0x400)
        tokens_np = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
    return Tensor(tokens_np)


def train():
    ### Load data
    train_tokens = load_tokens("data/tiny_shakespeare_train.bin")
    val_tokens = load_tokens("data/tiny_shakespeare_val.bin")
    print(
        f"Dataset size: {len(train_tokens)/1e3:.2f}K training tokens and {len(val_tokens)/1e3:.2f}K validation tokens."
    )

    ### Create model and optimizer
    assert 1 <= ctx_len <= block_size
    model = GPT()
    optimizer = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr, weight_decay=0)
    print(
        f"Total number of trainable parameters: {sum(p.numel() for p in optimizer.params) / 1e6:.2f}M"
    )
    print("Starting training...\n=================\n")

    ### Training loop
    @TinyJit
    def training_step(x, y):
        logits = model(x)
        loss = logits.sparse_categorical_crossentropy(y)
        optimizer.zero_grad()
        loss.backward()
        return loss.realize(*optimizer.schedule_step())

    @TinyJit
    def eval_step(x, y):
        logits = model(x)
        loss = logits.sparse_categorical_crossentropy(y)
        return loss.realize(*nn.state.get_parameters(model))  # ???

    best_val_loss = float("inf")
    for epoch in range(1, 1 + num_epochs):
        # Training step
        with Tensor.train():
            batch_cnt = 0
            running_loss = 0
            for x, y in get_batches(train_tokens):
                GlobalCounters.reset()
                batch_cnt += 1
                t0 = time.perf_counter()
                loss = training_step(x.contiguous(), y.contiguous())
                Device[Device.DEFAULT].synchronize()
                running_loss += loss.item()
                elapsed = time.perf_counter() - t0
                tflops = GlobalCounters.global_ops / elapsed / 1e12
                print(
                    f"\rStep {batch_cnt}, loss: {loss.item():.4f}, time: {elapsed*1000:.4f}ms, {batch_size*ctx_len/elapsed:.2f} tok/s, {tflops:.2f} TFLOPS",
                    end="",
                )
            print("")
            avg_train_loss = running_loss / batch_cnt

        # Evaluation step
        with Tensor.test():
            batch_cnt = 0
            running_loss = 0
            for x, y in get_batches(val_tokens):
                # for x, y in get_validation_batches():
                batch_cnt += 1
                loss = eval_step(x.contiguous(), y.contiguous())
                Device[Device.DEFAULT].synchronize()
                running_loss += loss.item()
            avg_eval_loss = running_loss / batch_cnt

        print(
            f"Epoch {epoch:2} | train loss: {avg_train_loss:.4f} | val loss: {avg_eval_loss:.4f}"
        )

        # Save checkpoint
        if avg_eval_loss < best_val_loss:
            best_val_loss = avg_eval_loss
            nn.state.safe_save(
                nn.state.get_state_dict(model), "checkpoints/best_gpt2.safetensors"
            )


def inference():
    model = GPT.from_pretrained("checkpoints/best_gpt2.safetensors")
    tokenizer = Tokenizer()
    val_tokens = load_tokens("data/tiny_shakespeare_val.bin")
    ### Inference on the first sentence of the validation set
    endoftext_positions = np.argwhere(
        val_tokens.numpy() == tokenizer.encode("<|endoftext|>")[0]
    ).ravel()
    input_sentence = val_tokens[
        int(endoftext_positions[0]) + 1 : int(endoftext_positions[1]) + 1
    ]
    expected_output_sentence = val_tokens[
        int(endoftext_positions[1]) + 1 : int(endoftext_positions[2]) + 1
    ]
    print(f"Input sentence: {tokenizer.decode(input_sentence)}")
    print(f"Expected output sentence: {tokenizer.decode(expected_output_sentence)}")
    with Tensor.test():
        output = model.generate(
            ctx=input_sentence.view(1, -1),
            max_new_tokens=len(expected_output_sentence),
            temperature=1.0,
        )[0]
        print(f"Actual output sentence: {tokenizer.decode(output)}")


if __name__ == "__main__":
    train()
