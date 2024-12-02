import math
import os
import time
from argparse import ArgumentParser

import numpy as np
import tiktoken
from icecream import ic
from tinygrad import Device, GlobalCounters, Tensor, TinyJit, nn  # type: ignore
from tinygrad.helpers import tqdm

import wandb
from tadam.gpt import GPT, GPTConfig
from tadam.ngpt import NGPT, NGPTConfig
from tadam.optim import CayleyAdam, GenericAdam, IntermediateAdam
from tadam.utils import get_state_dict

Tensor.manual_seed(127)

### hyper-parameters
skip_eval = False
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
                NGPTConfig(
                    block_size,
                    vocab_size,
                    padded_vocab_size,
                    n_layer,
                    n_head,
                    n_embd,
                    1 / n_layer,
                    1 / math.sqrt(n_embd),
                    1.0,
                    1 / math.sqrt(n_embd),
                    1.0,
                    1.0,
                    1.0,
                    1 / math.sqrt(n_embd),
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
    """
    one seq:


    """

    def __init__(self, toks: Tensor, random_order: bool = True):
        self.toks = toks
        self.random_order = random_order
        self.indices = self._gen_indices()

    def __len__(self):
        return self.toks.shape[0] // (batch_size * ctx_len + 1)

    def _gen_indices(self):
        return np.random.permutation(len(self)) if self.random_order else np.arange(len(self))

    def __iter__(self):
        Tensor.randint(0, len(self), out=self.indices)
        self.indices = self._gen_indices()
        for idx in self.indices:
            i = int(idx * (batch_size * ctx_len + 1))
            x = self.toks[i : i + batch_size * ctx_len].view(batch_size, ctx_len)
            y = self.toks[i + 1 : i + batch_size * ctx_len + 1].view(batch_size, ctx_len)
            yield x, y


def get_batch(toks: Tensor):
    idx = Tensor.randint(batch_size, low=0, high=len(toks) - ctx_len - 1).reshape(-1, 1)
    x = toks[idx + Tensor.arange(ctx_len)].contiguous()
    y = toks[idx + Tensor.arange(1, ctx_len + 1)].contiguous()
    return x, y


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
    ### Parse cli arguments
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    model_name = args.model
    optimizer_name = args.optimizer
    # TODO: implement lr scheduler
    lr = args.lr
    wd = args.wd
    train_steps = args.train_steps
    eval_steps = args.eval_steps
    eval_interval = args.eval_interval
    save_checkpoints = args.save_checkpoints
    silent = args.silent

    ### Create logging stuff
    os.makedirs("checkpoints", exist_ok=True)
    if not silent:
        wandb.init(
            project="tadam",
            name=f"{model_name}-{optimizer_name}-{lr}" + (f"-{wd}" if args.wd > 0 else ""),
            config={
                "model": model_name,
                "optimizer": optimizer_name,
                "epochs": num_epochs,
                "lr": lr,
                "wd": wd,
                "batch_size": batch_size,
                "device": Device.DEFAULT,
                "beam": os.getenv("BEAM", 0),
            },
        )

    ### Load data
    train_tokens = load_tokens("data/tiny_shakespeare_train.bin")
    # train_dataset = Dataset(train_tokens)
    eval_tokens = load_tokens("data/tiny_shakespeare_val.bin")
    # val_dataset = Dataset(val_tokens)
    if not silent:
        print(
            f"Dataset size: {len(train_tokens)/1e3:.2f}K training tokens and "
            f"{len(eval_tokens)/1e3:.2f}K validation tokens."
        )

    ### Create model and optimizer
    assert 1 <= ctx_len <= block_size
    model = get_model(model_name)
    state_dict = get_state_dict(model)
    norm_params, non_norm_params = split_state_dict(state_dict)
    norm_params = list(norm_params.values())
    non_norm_params = list(non_norm_params.values())
    if model_name == "gpt":
        assert len(norm_params) == 0
        assert optimizer_name == "adam"
        optimizer = GenericAdam(non_norm_params, lr=lr, weight_decay=wd)
    elif model_name == "ngpt":
        assert len(norm_params) > 0
        match optimizer_name:
            case "adam":
                first_optimizer = GenericAdam(norm_params, lr=lr, weight_decay=0)
            case "intermediate_adam":
                first_optimizer = IntermediateAdam(norm_params, lr=lr, weight_decay=0)
            case "cayley":
                first_optimizer = CayleyAdam(norm_params, lr=lr, weight_decay=0)
            case _:
                raise ValueError(f"Unknown optimizer name: {optimizer_name}")
        optimizer = nn.optim.OptimizerGroup(first_optimizer, GenericAdam(non_norm_params, lr=lr, weight_decay=0))

    if not silent:
        total_number_trainable_parameters = f"{sum(p.numel() for p in optimizer.params) / 1e6:.2f}M"
        ic(total_number_trainable_parameters, norm_params, non_norm_params)

    ### Training loop
    @TinyJit
    def training_step(x, y):
        loss = model(x).sparse_categorical_crossentropy(y)
        optimizer.zero_grad()  # TODO: move somewhere else to optimize memory?
        loss.backward()
        return loss.realize(*optimizer.schedule_step())

    @TinyJit
    def eval_step(x, y):
        return model(x).sparse_categorical_crossentropy(y).realize(*(norm_params + non_norm_params))

    best_eval_loss = float("inf")
    if not silent:
        print("Starting training...\n=================\n")
    for step in (pbar := tqdm(range(train_steps), unit="steps")):
        # TODO: implement lr scheduler

        # Evaluation step
        if step % eval_interval == 0 and not skip_eval:
            # compute loss over a few batches in the train and eval datasets
            t0 = time.perf_counter()
            with Tensor.test():
                eval_loss = sum(eval_step(*get_batch(eval_tokens)).item() for _ in range(eval_steps)) / eval_steps
                Device[Device.DEFAULT].synchronize()
            eval_runtime = time.perf_counter() - t0
            wandb.log(
                {
                    "performance/eval_time_ms": 1000 * eval_runtime,
                    "eval_loss": eval_loss,
                }
            )
            # Save checkpoint
            if save_checkpoints and eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                nn.state.safe_save(
                    get_state_dict(model),
                    f"checkpoints/best_{model_name}.safetensors",
                )

        # Training step
        x, y = get_batch(train_tokens)
        GlobalCounters.reset()
        t0 = time.perf_counter()
        with Tensor.train():
            loss = training_step(x, y).item()
            optimizer.zero_grad()
            Device[Device.DEFAULT].synchronize()
        step_runtime_s = time.perf_counter() - t0
        step_runtime_ms = 1000 * step_runtime_s
        tflops = GlobalCounters.global_ops / step_runtime_s / 1e12
        ktok_per_s = batch_size * ctx_len / step_runtime_s / 1e3
        wandb.log(
            {
                "train_loss": loss,
                "performance/ktok_per_s": ktok_per_s,
                "performance/step_runtime_ms": step_runtime_ms,
                "performance/TFLOPS": tflops,
            }
        )
        pbar.desc = (
            f"train loss: {loss:.4f}, step time: {step_runtime_ms:.4f}ms, "
            f"{ktok_per_s:.2f} tok/s, {tflops:.2f} TFLOPS "
        )


def inference():
    ### Parse cli arguments
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--save_checkpoints", action="store_true")
    model_name = parser.parse_args().model
    ### Load model
    model = get_model(model_name, checkpoint=f"checkpoints/best_{model_name}.safetensors")
    ### Load validation data and tokenizer
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
    import os

    import requests

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
