import numpy as np
from icecream import ic
from tinygrad import Tensor, nn
from tadam.utils import norm
from tadam.model import GPT, GPTConfig

# load models
ngpt_model = GPT(GPTConfig(ngpt=True, vocab_size=65), "checkpoints/best_ngpt.safetensors")
gpt_model = GPT(GPTConfig(ngpt=False, vocab_size=65), "checkpoints/best_gpt.safetensors")
gpt_state_dict = nn.state.get_state_dict(gpt_model)
ngpt_state_dict = nn.state.get_state_dict(ngpt_model)
ic(gpt_state_dict.keys())
print("GPT:")
ic(
    {
        k: norm(v).mean().item()
        for k, v in gpt_state_dict.items()
        if v.requires_grad is None and len(v.shape) == 2
    },
    {k: v.mean().item() for k, v in gpt_state_dict.items() if "ln_" in k},
)
print("NGPT:")
ic(
    {
        k: norm(v).mean().item()
        for k, v in ngpt_state_dict.items()
        if v.requires_grad is None and len(v.shape) == 2
    },
    [b.alpha_attn().mean().item() for b in ngpt_model.h],
    [b.attn.s_qk().mean().item() for b in ngpt_model.h],
    [b.mlp.s_uv().mean().item() for b in ngpt_model.h],
    ngpt_model.s_z()[:65].mean().item(),
)
