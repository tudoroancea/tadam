from os import getenv
from typing import OrderedDict
from tinygrad import Tensor, GlobalCounters
from tinygrad.helpers import tqdm, Timing
from tinygrad.multi import MultiLazyBuffer

DEBUG = int(getenv("DEBUG", 0))

__all__ = ["norm", "normalize"]


def norm(x: Tensor) -> Tensor:
    """x (..., n) -> ||x|| (..., 1)"""
    return x.square().sum(-1, keepdim=True).sqrt()


def normalize(x: Tensor) -> Tensor:
    return x / norm(x)


def get_state_dict(obj, prefix: str = "", tensor_type=Tensor) -> dict[str, Tensor]:
    """
    Returns a state_dict of the object, with optional prefix.

    ```python exec="true" source="above" session="tensor" result="python"
    class Net:
      def __init__(self):
        self.l1 = nn.Linear(4, 5)
        self.l2 = nn.Linear(5, 6)

    net = Net()
    print(nn.state.get_state_dict(net).keys())
    ```
    """
    if isinstance(obj, tensor_type) and (obj.requires_grad is None or obj.requires_grad):  # exclude requires_grad=False
        return {prefix.strip("."): obj}
    if hasattr(obj, "_asdict"):
        return get_state_dict(obj._asdict(), prefix, tensor_type)  # namedtuple
    if isinstance(obj, OrderedDict):
        return get_state_dict(dict(obj), prefix, tensor_type)
    if hasattr(obj, "__dict__"):
        return get_state_dict(obj.__dict__, prefix, tensor_type)
    state_dict = {}
    if isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            state_dict.update(get_state_dict(x, f"{prefix}{str(i)}.", tensor_type))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            state_dict.update(get_state_dict(v, f"{prefix}{str(k)}.", tensor_type))
    return state_dict


def load_state_dict(model, state_dict: dict[str, Tensor], strict=True, verbose=True, consume=False) -> None:
    """
    Loads a state_dict into a model.

    ```python
    class Net:
      def __init__(self):
        self.l1 = nn.Linear(4, 5)
        self.l2 = nn.Linear(5, 6)

    net = Net()
    state_dict = nn.state.get_state_dict(net)
    nn.state.load_state_dict(net, state_dict)
    ```
    """
    start_mem_used = GlobalCounters.mem_used
    with Timing(
        "loaded weights in ",
        lambda et_ns: f", {(GlobalCounters.mem_used-start_mem_used)/1e9:.2f} GB loaded at "
        f"{(GlobalCounters.mem_used-start_mem_used)/et_ns:.2f} GB/s",
    ):  # noqa: E501
        model_state_dict = get_state_dict(model)
        if DEBUG >= 1 and len(state_dict) > len(model_state_dict):
            print("WARNING: unused weights in state_dict", sorted(list(state_dict.keys() - model_state_dict.keys())))
        for k, v in (t := tqdm(model_state_dict.items())):
            t.desc = f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, {k:50s}: "
            if k not in state_dict and not strict:
                if DEBUG >= 1:
                    print(f"WARNING: not loading {k}")
                continue
            if isinstance((mlb := v.lazydata), MultiLazyBuffer):
                if isinstance(state_dict[k].lazydata, MultiLazyBuffer):
                    v.replace(state_dict[k]).realize()
                else:
                    v.replace(state_dict[k].shard(mlb.device, mlb.axis)).realize()
            else:
                v.replace(state_dict[k].to(v.device)).realize()
            if consume:
                del state_dict[k]
