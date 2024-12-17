# `tadam`: Tangent space ADAM
```bash
uv run download_dataset_char
uv run train_char --model gpt --save_checkpoints
uv run inference_char --model gpt --temp 1.0
```
> _Isn't that just wonderful? The simplicity of it all._

If you want some more performance, you can also specify an env var `BEAM` to enable tinygrad's beam search:
```bash
BEAM=10 uv run train_char --model gpt --save_checkpoints
BEAM=10 uv run inference_char --model gpt --temp 1.0
```
