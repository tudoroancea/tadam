import os
import time
import tiktoken
from argparse import ArgumentParser

import numpy as np
from tinygrad import Device, Tensor, TinyJit, nn, GlobalCounters
from tiny_gpt2.gpt import GPT, GPTConfig
from tiny_gpt2.ngpt import NGPT, NGPTConfig
from tiny_gpt2.optim import GenericAdam

Tensor.manual_seed(127)

### hyper-parameters
# model
block_size: int = 128
vocab_size: int = 50257
padded_vocab_size: int = 50304
n_layer: int = 2
n_head: int = 4
n_embd: int = 128
# training
ctx_len: int = 128
batch_size: int = 64
num_epochs: int = 1
lr = 1e-3


def get_model(model_name: str, checkpoint: str | None = None):
    match model_name:
        case "gpt":
            return GPT(
                GPTConfig(
                    block_size, vocab_size, padded_vocab_size, n_layer, n_head, n_embd
                ),
                checkpoint,
            )
        case "ngpt":
            return NGPT(
                NGPTConfig(
                    block_size, vocab_size, padded_vocab_size, n_layer, n_head, n_embd
                ),
                checkpoint,
            )
        case _:
            raise ValueError(f"Unknown model name: {model_name}")


def split_parameters(params: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    # split parameters into two groups: those that need to be normalized and those that don't
    norm_params, non_norm_params = [], []
    for p in params:
        if hasattr(p, "__normalized__"):
            norm_params.append(p)
        else:
            non_norm_params.append(p)
    return norm_params, non_norm_params


def get_batches(toks: Tensor):
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


def model_name_parser() -> str:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    return parser.parse_args().model


def train():
    ### Load data
    train_tokens = load_tokens("data/tiny_shakespeare_train.bin")
    val_tokens = load_tokens("data/tiny_shakespeare_val.bin")
    print(
        f"Dataset size: {len(train_tokens)/1e3:.2f}K training tokens and {len(val_tokens)/1e3:.2f}K validation tokens."
    )

    ### Create model and optimizer
    assert 1 <= ctx_len <= block_size
    model_name = model_name_parser()
    model = get_model(model_name)
    # optimizer = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr, weight_decay=0)
    optimizer = GenericAdam(nn.state.get_parameters(model), lr=lr)
    # norm_params, non_norm_params = split_parameters(nn.state.get_parameters(model))
    # optimizer = nn.optim.OptimizerGroup(
    #     GenericAdam(norm_params, lr=lr, weight_decay=0),
    #     # CayleyAdam(norm_params, lr=lr, weight_decay=0),
    #     GenericAdam(non_norm_params, lr=lr, weight_decay=0),
    # )

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
                nn.state.get_state_dict(model),
                f"checkpoints/best_{model_name}.safetensors",
            )


def inference():
    model_name = model_name_parser()
    model = get_model(model_name, checkpoint="checkpoints/best_gpt2.safetensors")
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
