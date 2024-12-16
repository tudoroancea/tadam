import math
import os
import pickle
import time
from argparse import ArgumentParser

import numpy as np
from icecream import ic
from tinygrad import Device, GlobalCounters, Tensor, TinyJit, nn  # type: ignore
from tinygrad.helpers import tqdm

import wandb
from tadam.model import GPT, GPTConfig
from tadam.optim import Adam, CayleyAdam, IntermediateAdam

Tensor.manual_seed(127)
np.random.seed(127)


class Tokenizer:
    def __init__(self, itos: dict[int, str], stoi: dict[str, int]):
        self.itos = itos
        self.stoi = stoi

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(self, l: list[int]) -> str:
        return "".join([self.itos[i] for i in l])  # decoder: take a list of integers, output a string


def load_tokens(data_path: str):
    with open(data_path, "rb") as f:
        tokens_np = np.frombuffer(f.read(), dtype=np.uint8)
    return Tensor(tokens_np)


def get_batch(toks: Tensor, batch_size: int, ctx_len: int):
    idx = Tensor.randint(batch_size, low=0, high=len(toks) - ctx_len - 1).reshape(-1, 1)
    x = toks[idx + Tensor.arange(ctx_len)].contiguous()
    y = toks[idx + Tensor.arange(1, ctx_len + 1)].contiguous()
    return x, y


def train():
    ### Parse cli arguments
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--ctx_len", type=int, default=GPTConfig.block_size)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=1e-1)
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()
    model_name = args.model
    optimizer_name = args.optimizer
    ctx_len = args.ctx_len
    batch_size = args.batch_size
    train_steps = args.train_steps
    eval_steps = args.eval_steps
    eval_interval = args.eval_interval
    warmup_steps = args.warmup_steps
    max_lr = args.max_lr
    min_lr = args.min_lr
    if min_lr is None:
        min_lr = max_lr / 10  # as per Chinchilla
    wd = args.wd
    save_checkpoints = args.save_checkpoints
    silent = args.silent
    skip_eval = args.skip_eval

    ### Create logging stuff
    os.makedirs("checkpoints", exist_ok=True)
    if not silent:
        wandb.init(
            project="tadam",
            name=f"{model_name}-{optimizer_name}-{max_lr}",
            config={
                "model": model_name,
                "optimizer": optimizer_name,
                "train_steps": train_steps,
                "eval_steps": eval_steps,
                "eval_interval": eval_interval,
                "warmup_steps": warmup_steps,
                "max_lr": max_lr,
                "min_lr": min_lr,
                "wd": wd,
                "batch_size": batch_size,
                "device": Device.DEFAULT,
                "beam": os.getenv("BEAM", 0),
            },
        )

    ### Load data
    train_tokens = load_tokens("data/shakespeare_char_train.bin")
    eval_tokens = load_tokens("data/shakespeare_char_val.bin")
    with open("data/shakespeare_char_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    print(meta)
    if not silent:
        print(
            f"Dataset size: {len(train_tokens)/1e3:.2f}K training tokens and "
            f"{len(eval_tokens)/1e3:.2f}K validation tokens."
        )

    ### Create model and optimizer
    config = GPTConfig(ngpt=model_name == "ngpt", vocab_size=meta["vocab_size"], n_layer=6)
    assert 1 <= ctx_len <= config.block_size
    model = GPT(config)
    state_dict = nn.state.get_state_dict(model)
    params = list(state_dict.values())
    match optimizer_name:
        case "adam":
            optimizer = Adam(params, lr=0, weight_decay=wd)
        case "intermediate_adam":
            raise NotImplementedError
            optimizer = IntermediateAdam(params, lr=0, weight_decay=wd)
        case "cayley":
            raise NotImplementedError
            optimizer = CayleyAdam(params, lr=0, weight_decay=wd)
        case _:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    if not silent:
        trainable_params_dict = {k: v for k, v in state_dict.items() if v.requires_grad}
        total_number_trainable_parameters = f"{sum(p.numel() for p in optimizer.params) / 1e6:.2f}M"
        ic(trainable_params_dict, total_number_trainable_parameters)

    ### Training loop
    @TinyJit
    def training_step(x, y):
        loss = model(x).sparse_categorical_crossentropy(y)
        loss.backward()
        return loss.realize(*optimizer.schedule_step())

    @TinyJit
    def eval_step(x, y):
        return model(x).sparse_categorical_crossentropy(y).realize(*(optimizer.params + optimizer.buffers))

    best_eval_loss = float("inf")
    if not silent:
        print("Starting training...\n=================\n")
    for step in (pbar := range(train_steps) if silent else tqdm(range(train_steps), unit="steps")):
        # Evaluation step
        if step % eval_interval == 0 and not skip_eval:
            # compute loss over a few batches in the train and eval datasets
            t0 = time.perf_counter()
            with Tensor.test():
                eval_loss = (
                    sum(eval_step(*get_batch(eval_tokens, batch_size, ctx_len)).item() for _ in range(eval_steps))
                    / eval_steps
                )
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
                checkpoint_path = f"checkpoints/best_{model_name}.safetensors"
                nn.state.safe_save(nn.state.get_state_dict(model), checkpoint_path)
                wandb.save(checkpoint_path)

        # LR schedule
        if step < warmup_steps:
            # LR warmup
            lr = max_lr * (step / warmup_steps)
        else:
            # Cosine annealing
            decay_ratio = (step - warmup_steps) / (train_steps - warmup_steps)
            assert 0.0 <= decay_ratio <= 1.0
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            lr = min_lr + coeff * (max_lr - min_lr)
        optimizer.lr[0] = lr

        # Training step
        x, y = get_batch(train_tokens, batch_size, ctx_len)
        GlobalCounters.reset()
        t0 = time.perf_counter()
        with Tensor.train():
            train_loss = training_step(x, y).item()
            optimizer.zero_grad()
            Device[Device.DEFAULT].synchronize()
        step_runtime_s = time.perf_counter() - t0
        step_runtime_ms = 1000 * step_runtime_s
        tflops = GlobalCounters.global_ops / step_runtime_s / 1e12
        ktok_per_s = batch_size * ctx_len / step_runtime_s / 1e3
        memory_gb = GlobalCounters.mem_used / 1e9

        # Logging
        if not silent:
            wandb.log(
                {
                    "lr": lr,
                    "train_loss": train_loss,
                    "performance/ktok_per_s": ktok_per_s,
                    "performance/step_runtime_ms": step_runtime_ms,
                    "performance/TFLOPS": tflops,
                }
            )
            pbar.desc = (
                f"train loss: {train_loss:.4f}, step time: {step_runtime_ms:.4f}ms, "
                f"{ktok_per_s:.2f} Ktok/s, {tflops:.2f} TFLOPS, {memory_gb:.2f} GB "
            )

    if save_checkpoints:
        checkpoint_path = f"checkpoints/final_{model_name}.safetensors"
        nn.state.safe_save(nn.state.get_state_dict(model), checkpoint_path)
        wandb.save(checkpoint_path)


def inference():
    with open("data/shakespeare_char_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    ### Parse cli arguments
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()
    model_name = args.model
    ### Load model
    config = GPTConfig(ngpt=model_name == "ngpt", vocab_size=meta["vocab_size"], block_size=256, n_layer=1)
    model = GPT(config, "checkpoints/final_gpt.safetensors")
    ### Load validation data and tokenizer
    tokens = load_tokens("data/shakespeare_char_train.bin")
    # tokens = load_tokens("data/tiny_shakespeare_val.bin")
    tokenizer = Tokenizer(meta["itos"], meta["stoi"])
    ###
    max_new_tokens = 30
    input = tokens[: config.block_size]
    # input = Tensor(tokenizer.encode("\n"))
    expected_output = tokens[config.block_size : config.block_size + max_new_tokens]
    print("############# Input #############")
    print(tokenizer.decode(input.tolist()))
    print("############ Expected output #############")
    print(tokenizer.decode(expected_output.tolist()))
    print("############ Generated  output #############")
    with Tensor.test():
        output = model.generate(input.view(1, -1), max_new_tokens, args.temp)[0][0, -max_new_tokens:]
    print(tokenizer.decode(output.tolist()))


def naive_inference():
    with open("data/shakespeare_char_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()
    model_name = args.model
    temp = args.temp
    ### Load model
    config = GPTConfig(ngpt=model_name == "ngpt")
    model = GPT(config, "checkpoints/final_gpt.safetensors")
    ### Load validation data and tokenizer
    tokenizer = Tokenizer(meta["itos"], meta["stoi"])
    tokens = load_tokens("data/shakespeare_char_train.bin")
    ### Check that the entropy is similar to the one observed during the train
    with Tensor.test():
        x, y = get_batch(tokens, 64, config.block_size)
        entropy = model(x).sparse_categorical_crossentropy(y).numpy()
        ic(entropy)
    ### Take as many sentences as possible fitting in the block size, and make the model generate the next sentences
    i = np.random.randint(0, len(tokens) - config.block_size)
    input = tokens[i : i + config.block_size]
    expected_output = tokens[i + config.block_size : i + config.block_size + 50]
    print("############# Input #############")
    print(tokenizer.decode(input.tolist()))
    print("############ Expected output #############")
    print(tokenizer.decode(expected_output.tolist()))
    print("############ Generated  output #############")
    with Tensor.test():
        output, probs = model.generate(input.view(1, -1), len(expected_output), temp)
        output = output[0].numpy().tolist()
        probs = probs[0].numpy()
        print(tokenizer.decode(output))

    # plot distribution of logits
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    def update(val):
        ax.clear()
        time_step = int(slider.val)
        ax.step(np.arange(len(probs[time_step])), probs[time_step], where="post")
        ax.set_title(f"Token Distribution at Step {time_step}")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        # fig.canvas.draw_idle()

    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, "Time Step", 0, len(probs) - 1, valinit=0, valstep=1)
    slider.on_changed(update)

    update(0)
    plt.savefig("distribution.png", dpi=300, bbox_inches="tight")
    plt.show()


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
