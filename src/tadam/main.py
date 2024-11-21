import time
from argparse import ArgumentParser
import os

import numpy as np
import tiktoken
import wandb
from icecream import ic
from tinygrad import Device, GlobalCounters, Tensor, TinyJit, nn  # type: ignore
from tinygrad.helpers import tqdm

from tadam.gpt import GPT, GPTConfig
from tadam.ngpt import NGPT, NGPTConfig
from tadam.optim import GenericAdam
from tadam.utils import get_state_dict

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
num_epochs: int = 50
lr = 1e-3


def get_model(model_name: str, checkpoint: str | None = None):
    match model_name:
        case "gpt":
            return GPT(
                GPTConfig(block_size, vocab_size, padded_vocab_size, n_layer, n_head, n_embd),
                checkpoint,
            )
        case "ngpt":
            return NGPT(
                NGPTConfig(block_size, vocab_size, padded_vocab_size, n_layer, n_head, n_embd),
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


def split_state_dict(state_dict: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    # split state_dict into two groups: those that need to be normalized and those that don't
    norm_state_dict, non_norm_state_dict = {}, {}
    for k, v in state_dict.items():
        if hasattr(v, "__normalized__"):
            norm_state_dict[k] = v
        else:
            non_norm_state_dict[k] = v
    return norm_state_dict, non_norm_state_dict


class Dataset:
    """Lightweight dataloader"""

    def __init__(self, toks: Tensor):
        self.toks = toks

    def __len__(self):
        return self.toks.shape[0] // (batch_size * ctx_len + 1)

    def __iter__(self):
        i = 0
        while i + batch_size * ctx_len + 1 < self.toks.shape[0]:
            x = self.toks[i : i + batch_size * ctx_len].view(batch_size, ctx_len)
            y = self.toks[i + 1 : i + batch_size * ctx_len + 1].view(batch_size, ctx_len)
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
    train_dataset = Dataset(train_tokens)
    val_tokens = load_tokens("data/tiny_shakespeare_val.bin")
    val_dataset = Dataset(val_tokens)
    print(
        f"Dataset size: {len(train_tokens)/1e3:.2f}K training tokens and {len(val_tokens)/1e3:.2f}K validation tokens."
    )
    print(len(train_dataset), len(val_dataset))

    ### Create model and optimizer
    assert 1 <= ctx_len <= block_size
    model_name = model_name_parser()
    model = get_model(model_name)
    state_dict = get_state_dict(model)
    norm_params, non_norm_params = split_state_dict(state_dict)
    ic(norm_params, non_norm_params)
    norm_params = list(norm_params.values())
    non_norm_params = list(non_norm_params.values())
    if model_name == "gpt":
        assert len(norm_params) == 0
        optimizer = GenericAdam(non_norm_params, lr=lr)
    elif model_name == "ngpt":
        assert len(norm_params) > 0
        optimizer = nn.optim.OptimizerGroup(
            GenericAdam(norm_params, lr=lr, weight_decay=0),
            # CayleyAdam(norm_params, lr=lr, weight_decay=0),
            GenericAdam(non_norm_params, lr=lr, weight_decay=0),
        )

    print(f"Total number of trainable parameters: {sum(p.numel() for p in optimizer.params) / 1e6:.2f}M")

    ### Create logging stuff
    os.makedirs("checkpoints", exist_ok=True)
    wandb.init(
        project="tadam",
        name=f"{model_name}-{lr}-{num_epochs}",
        config={
            "model": model_name,
            "optimizer": "adam",
            "epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "device": Device.DEFAULT,
            "beam": os.getenv("BEAM", 0),
        },
    )

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
        return loss.realize(*(norm_params + non_norm_params))  # TODO ???

    best_val_loss = float("inf")
    train_losses_per_step = []
    print("Starting training...\n=================\n")
    for epoch in range(1, 1 + num_epochs):
        # Training step
        with Tensor.train():
            batch_cnt = 0
            running_loss = 0
            for x, y in (pbar := tqdm(train_dataset, unit="steps")):
                GlobalCounters.reset()
                batch_cnt += 1
                t0 = time.perf_counter()
                loss = training_step(x.contiguous(), y.contiguous())
                Device[Device.DEFAULT].synchronize()
                running_loss += loss.item()
                step_runtime_s = time.perf_counter() - t0
                step_runtime_ms = step_runtime_s * 1000
                tflops = GlobalCounters.global_ops / step_runtime_s / 1e12
                train_losses_per_step.append(loss.item())
                ktok_per_s = batch_size * ctx_len / step_runtime_s / 1e3
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "epoch": epoch,
                        "performance/ktok_per_s": ktok_per_s,
                        "performance/step_runtime_ms": step_runtime_ms,
                        "performance/TFLOPS": tflops,
                    }
                )
                pbar.desc = (
                    f"train loss: {loss.item():.4f}, step time: {step_runtime_ms:.4f}ms, "
                    f"{ktok_per_s:.2f} tok/s, {tflops:.2f} TFLOPS "
                )
            avg_train_loss = running_loss / batch_cnt

        # Evaluation step
        with Tensor.test():
            batch_cnt = 0
            running_loss = 0
            for x, y in (pbar := tqdm(val_dataset, unit="steps")):
                batch_cnt += 1
                loss = eval_step(x.contiguous(), y.contiguous())
                Device[Device.DEFAULT].synchronize()
                running_loss += loss.item()
                pbar.desc = f"val_loss: {loss.item():.4f} "
            avg_eval_loss = running_loss / batch_cnt

        wandb.log({"val_loss": avg_eval_loss})
        print(f"Epoch {epoch:2} | train loss: {avg_train_loss:.4f} | val loss: {avg_eval_loss:.4f}")

        # Save checkpoint
        if avg_eval_loss < best_val_loss:
            best_val_loss = avg_eval_loss
            nn.state.safe_save(
                get_state_dict(model),
                f"checkpoints/best_{model_name}.safetensors",
            )


def inference():
    model_name = model_name_parser()
    model = get_model(model_name, checkpoint=f"checkpoints/best_{model_name}.safetensors")
    tokenizer = Tokenizer()
    val_tokens = load_tokens("data/tiny_shakespeare_val.bin")
    ### Inference on the first sentence of the validation set
    endoftext_positions = np.argwhere(val_tokens.numpy() == tokenizer.encode("<|endoftext|>")[0]).ravel()
    input_sentence = val_tokens[int(endoftext_positions[0]) + 1 : int(endoftext_positions[1]) + 1]
    expected_output_sentence = val_tokens[int(endoftext_positions[1]) + 1 : int(endoftext_positions[2]) + 1]
    print(f"Input sentence: {tokenizer.decode(input_sentence)}")
    print(f"Expected output sentence: {tokenizer.decode(expected_output_sentence)}")
    with Tensor.test():
        output = model.generate(
            ctx=input_sentence.view(1, -1),
            max_new_tokens=len(expected_output_sentence),
            temperature=1.0,
        )[0]
        print(f"Actual output sentence: {tokenizer.decode(output)}")


def download_dataset():
    import requests
    import os

    if not os.path.exists("data"):
        os.makedirs("data")
    for set in {"train", "val"}:
        url = f"https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_{set}.bin"
        filename = f"data/tiny_shakespeare_{set}.bin"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"Successfully downloaded {filename}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
