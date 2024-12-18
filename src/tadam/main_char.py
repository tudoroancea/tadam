import math
import os
import pickle
import time
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from icecream import ic
from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, nn  # type: ignore
from tinygrad.helpers import tqdm

import wandb
from tadam.model import GPT, GPTConfig
from tadam.optim import TADAM, Adam

Tensor.manual_seed(127)
np.random.seed(127)

META_DATA_FILE = "data/shakespeare_char_meta.pkl"
TRAIN_DATA_FILE = "data/shakespeare_char_train.bin"
EVAL_DATA_FILE = "data/shakespeare_char_eval.bin"

DEFAULT_BATCH_SIZE = 64


def download_dataset():
    """
    length of dataset in characters: 1,115,394
    all the unique characters: !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    vocab size: 65
    train has 1,003,854 tokens
    eval has 111,540 tokens
    """
    import requests

    # download the tiny shakespeare dataset
    os.makedirs("data", exist_ok=True)
    input_file_path = os.path.join("data/input.txt")
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, "r") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(l):
        return "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # encode both to integers
    train_ids = encode(train_data)
    eval_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(eval_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint8)
    eval_ids = np.array(eval_ids, dtype=np.uint8)
    train_ids.tofile(TRAIN_DATA_FILE)
    eval_ids.tofile(EVAL_DATA_FILE)

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(META_DATA_FILE, "wb") as f:
        pickle.dump(meta, f)


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


def beam():
    ### Parse cli arguments
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--ctx_len", type=int, default=GPTConfig.block_size)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--beam", type=int, default=10)
    args = parser.parse_args()
    model_name = args.model
    optimizer_name = args.optimizer
    ctx_len = args.ctx_len
    batch_size = args.batch_size
    beam = args.beam

    ### Load data
    with open(META_DATA_FILE, "rb") as f:
        meta = pickle.load(f)

    # Create model, optimizer, and beam search for train and eval steps
    with Context(DEBUG=2, BEAM=beam):
        for n_layer in [1, 6]:
            config = GPTConfig(ngpt=model_name == "ngpt", vocab_size=meta["vocab_size"], n_layer=n_layer)
            assert 1 <= ctx_len <= config.block_size
            model = GPT(config)
            params = list(nn.state.get_state_dict(model).values())
            match optimizer_name:
                case "sgd":
                    if model_name == "ngpt":
                        raise NotImplementedError("SGD hasn't been implemented for NGPT yet.")
                    optimizer = nn.optim.SGD(params, lr=1.0, weight_decay=1.0)
                case "adam":
                    optimizer = Adam(params, lr=1.0, weight_decay=1.0)
                case "tadam":
                    raise NotImplementedError
                    optimizer = TADAM(params, lr=1.0, weight_decay=1.0)
                case _:
                    raise ValueError(f"Unknown optimizer name: {optimizer_name}")

            x = Tensor.randint(batch_size, ctx_len, low=0, high=GPTConfig.vocab_size)
            y = Tensor.randint(batch_size, ctx_len, low=0, high=GPTConfig.vocab_size)
            with Tensor.train():
                _ = model(x).sparse_categorical_crossentropy(y).backward().realize(*optimizer.schedule_step()).item()
                Device[Device.DEFAULT].synchronize()  # maybe useless
            with Tensor.test():
                _ = model(x).sparse_categorical_crossentropy(y).realize(*(optimizer.params + optimizer.buffers)).item()
                Device[Device.DEFAULT].synchronize()  # maybe useless


def analyze_grads():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("qtagg")

    ### Parse cli arguments
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--ctx_len", type=int, default=GPTConfig.block_size)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    args = parser.parse_args()
    model_name = args.model
    optimizer_name = args.optimizer
    ctx_len = args.ctx_len
    batch_size = args.batch_size
    warmup_steps = args.warmup_steps
    max_lr = args.max_lr
    min_lr = args.min_lr
    if min_lr is None:
        min_lr = max_lr / 10  # as per Chinchilla
    wd = args.wd
    beta1 = args.beta1
    beta2 = args.beta2

    ### Load data
    train_tokens = load_tokens(TRAIN_DATA_FILE)
    with open(META_DATA_FILE, "rb") as f:
        meta = pickle.load(f)

    ### Create model and optimizer
    config = GPTConfig(ngpt=model_name == "ngpt", vocab_size=meta["vocab_size"])
    assert 1 <= ctx_len <= config.block_size
    model = GPT(config)
    state_dict = nn.state.get_state_dict(model)
    params = list(state_dict.values())
    match optimizer_name:
        case "sgd":
            if model_name == "ngpt":
                raise NotImplementedError("SGD hasn't been implemented for NGPT yet.")
            optimizer = nn.optim.SGD(params, lr=0, weight_decay=wd)
        case "adam":
            optimizer = Adam(params, lr=0, weight_decay=wd, beta1=beta1, beta2=beta2)
        case "tadam":
            optimizer = TADAM(params, lr=0, weight_decay=wd, beta1=beta1, beta2=beta2)
        case _:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    ### Setup training and eval steps
    @TinyJit
    def training_step():
        x, y = get_batch(train_tokens, batch_size, ctx_len)
        loss = model(x).sparse_categorical_crossentropy(y)
        loss.backward()
        grad_normal_residuals = {
            k: (p * p.grad).sum(-1) for k, p in state_dict.items() if p.grad is not None and p.ndim == 2
        }
        grad_tangent_residuals = {
            k: (state_dict[k].grad.square().sum(-1) + grad_normal_residuals[k].square()).sqrt()
            for k in grad_normal_residuals
        }

        return (
            loss.realize(*optimizer.schedule_step(), *grad_normal_residuals.values(), *grad_tangent_residuals.values()),
            {k: v.numpy() for k, v in grad_normal_residuals.items()},
            {k: v.numpy() for k, v in grad_tangent_residuals.items()},
        )

    ### Run the loop
    n = defaultdict(list)
    t = defaultdict(list)
    for step in range(10):
        # LR schedule
        if step < warmup_steps:
            # LR warmup
            lr = max_lr * (step / warmup_steps)
        else:
            # Cosine annealing
            decay_ratio = (step - warmup_steps) / (1000 - warmup_steps)
            assert 0.0 <= decay_ratio <= 1.0
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            lr = min_lr + coeff * (max_lr - min_lr)
        optimizer.lr[0] = lr

        # Training step
        with Tensor.train():
            train_loss, grad_normal_residuals, grad_tangent_residuals = training_step()
            optimizer.zero_grad()
            Device[Device.DEFAULT].synchronize()  # maybe useless

        # n.append(list(grad_normal_residuals.values()))
        # t.append(list(grad_tangent_residuals.values()))
        for k in grad_normal_residuals:
            n[k].append(grad_normal_residuals[k])
            t[k].append(grad_tangent_residuals[k])

        # analyze gradients
        # grad_normal_residuals_mean = {k: v.mean() for k, v in grad_normal_residuals.items()}
        # grad_normal_residuals_std = {k: v.std() for k, v in grad_normal_residuals.items()}
        # grad_tangent_residuals_mean = {k: v.mean() for k, v in grad_tangent_residuals.items()}
        # grad_tangent_residuals_std = {k: v.std() for k, v in grad_tangent_residuals.items()}
        # ic(
        #     grad_normal_residuals_mean,
        #     grad_normal_residuals_std,
        #     grad_tangent_residuals_mean,
        #     grad_tangent_residuals_std,
        # )
        # breakpoint()
    # plot...
    # log y axis
    # x axis = step
    # scatter plot in red points for the normal residuals, in blue points for the tangent residuals
    # breakpoint()
    n = {k: np.array(v) for k, v in n.items()}
    t = {k: np.array(v) for k, v in t.items()}
    for k in n:
        plt.figure()
        plt.scatter(np.repeat(np.arange(n[k].shape[0]), n[k].shape[1]), np.ravel(n[k]), c="r", label="normal residuals")
        plt.scatter(
            np.repeat(np.arange(t[k].shape[0]), t[k].shape[1]), np.ravel(t[k]), c="b", label="tangent residuals"
        )
        plt.legend(
            loc="best",
        )
        plt.yscale("log")
        plt.title(f"Gradient residuals for weight {k}")
        plt.xlabel("Step")
        plt.ylabel("residuals")
    plt.show()


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
    parser.add_argument("--lr_decay_steps", type=int, default=None)
    parser.add_argument("--wd", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
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
    lr_decay_steps = args.lr_decay_steps
    if lr_decay_steps is None:
        lr_decay_steps = train_steps
    wd = args.wd
    beta1 = args.beta1
    beta2 = args.beta2
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
                "lr_decay_steps": lr_decay_steps,
                "wd": wd,
                "beta1": beta1,
                "beta2": beta2,
                "batch_size": batch_size,
                "device": Device.DEFAULT,
                "beam": os.getenv("BEAM", 0),
            },
        )

    ### Load data
    train_tokens = load_tokens(TRAIN_DATA_FILE)
    eval_tokens = load_tokens(EVAL_DATA_FILE)
    with open(META_DATA_FILE, "rb") as f:
        meta = pickle.load(f)
    if not silent:
        print(
            f"Dataset size: {len(train_tokens)/1e3:.2f}K training tokens and "
            f"{len(eval_tokens)/1e3:.2f}K validation tokens."
        )

    ### Create model and optimizer
    config = GPTConfig(ngpt=model_name == "ngpt", vocab_size=meta["vocab_size"])
    assert 1 <= ctx_len <= config.block_size
    model = GPT(config)
    state_dict = nn.state.get_state_dict(model)
    params = list(state_dict.values())
    match optimizer_name:
        case "sgd":
            if model_name == "ngpt":
                raise NotImplementedError("SGD hasn't been implemented for NGPT yet.")
            optimizer = nn.optim.SGD(params, lr=0, weight_decay=wd)
        case "adam":
            optimizer = Adam(params, lr=0, weight_decay=wd, beta1=beta1, beta2=beta2)
        case "tadam":
            optimizer = TADAM(params, lr=0, weight_decay=wd, beta1=beta1, beta2=beta2)
        case _:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    if not silent:
        trainable_params_dict = {k: v for k, v in state_dict.items() if v.requires_grad}
        total_number_trainable_parameters = f"{sum(p.numel() for p in optimizer.params) / 1e6:.2f}M"
        ic(trainable_params_dict, total_number_trainable_parameters)

    ### Setup training and eval steps
    @TinyJit
    def training_step():
        x, y = get_batch(train_tokens, batch_size, ctx_len)
        return model(x).sparse_categorical_crossentropy(y).backward().realize(*optimizer.schedule_step())

    @TinyJit
    def eval_step():
        x, y = get_batch(eval_tokens, batch_size, ctx_len)
        return model(x).sparse_categorical_crossentropy(y).realize(*(optimizer.params + optimizer.buffers))

    def perform_eval(best_eval_loss: float, eval_loss: float):
        # compute loss over a few batches in the train and eval datasets
        t0 = time.perf_counter()
        with Tensor.test():
            eval_loss = sum(eval_step().item() for _ in range(eval_steps)) / eval_steps
            Device[Device.DEFAULT].synchronize()  # maybe useless
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

        return best_eval_loss, eval_loss

    ### Run the loop
    best_eval_loss = float("inf")
    eval_loss = float("inf")
    if not silent:
        print("Starting training...\n=================\n")
    for step in (pbar := range(train_steps) if silent else tqdm(range(train_steps), unit="steps")):
        # Evaluation step
        if step % eval_interval == 0 and not skip_eval:
            best_eval_loss, eval_loss = perform_eval(best_eval_loss, eval_loss)

        # LR schedule
        if step < warmup_steps:
            # LR warmup
            lr = max_lr * (step / warmup_steps)
        else:
            # Cosine annealing
            decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
            assert 0.0 <= decay_ratio <= 1.0
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            lr = min_lr + coeff * (max_lr - min_lr)
        optimizer.lr[0] = lr

        # Training step
        GlobalCounters.reset()
        t0 = time.perf_counter()
        with Tensor.train():
            train_loss = training_step().item()
            optimizer.zero_grad()
            Device[Device.DEFAULT].synchronize()  # maybe useless
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
                f"train loss: {train_loss:.4f}, eval loss: {eval_loss:.4f}, "
                f"step time: {step_runtime_ms:.4f}ms, {ktok_per_s:.2f} Ktok/s, {tflops:.2f} TFLOPS, {memory_gb:.2f} GB "
            )

    best_eval_loss, eval_loss = perform_eval(best_eval_loss, eval_loss)
    if save_checkpoints:
        checkpoint_path = f"checkpoints/final_{model_name}.safetensors"
        nn.state.safe_save(nn.state.get_state_dict(model), checkpoint_path)
        wandb.save(checkpoint_path)


def inference():
    with open(META_DATA_FILE, "rb") as f:
        meta = pickle.load(f)
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--data", type=str, default="eval", choices=["train", "eval"])
    args = parser.parse_args()
    model_name = args.model
    temp = args.temp
    ### Load model
    config = GPTConfig(ngpt=model_name == "ngpt")
    model = GPT(config, f"checkpoints/final_{model_name}.safetensors")
    ### Load validation data and tokenizer
    tokenizer = Tokenizer(meta["itos"], meta["stoi"])
    tokens = load_tokens(TRAIN_DATA_FILE if args.data == "train" else EVAL_DATA_FILE)
    ### Generate based on the first sentence
    i = np.random.randint(0, len(tokens) - config.block_size)
    input = tokens[i : i + config.block_size]
    expected_output = tokens[i + config.block_size : i + config.block_size + args.max_new_tokens]
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
