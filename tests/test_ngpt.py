import numpy as np
import pytest
from icecream import ic
from tinygrad import Tensor, nn

from tadam.ngpt import NGPT, NGPTConfig, norm


@pytest.fixture
def config() -> NGPTConfig:
    return NGPTConfig(
        block_size=64,
        vocab_size=512,
        padded_vocab_size=512,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )


def test_forward_pass(config):
    model = NGPT(config)
    ctx_len = 8
    batch_size = 4
    x = Tensor.randint(
        batch_size,
        ctx_len,
        low=0,
        high=config.vocab_size,
    )
    state_dict = nn.state.get_state_dict(model)
    ic(state_dict)
    params = list(state_dict.values())
    y = model(x).numpy()
    assert y.shape == (batch_size, ctx_len, config.padded_vocab_size)
    # check that all the weights are normalized
    for p in params:
        if hasattr(p, "__normalized__"):
            assert np.allclose(norm(p).numpy(), 1.0)

    # check no NaNs
    assert not np.any(np.isnan(y))
