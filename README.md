# `tiny-gpt2`

## setup
```bash
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_train.bin
wget https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_val.bin
uv run train
```

## todo
- [x] finish implementing `NGPT`
  - [x] mark weights with `__normalized__`
- [x] implement `DumbRiemannianAdam`
  - [x] add weight decay to use the same optimizer for `NGPT` and `GPT`
  - [x] understand how the scaling learning rate works
- [ ] implement `CayleyAdam`
  - [x] implement updates for normalized weights
  - [ ] implement updates for non-normalized weights
- [ ] compare over 1 epoch:
  - [ ] `NGPT` with `DumbRiemannianAdam` vs `GPT` with `Adam`
  - [ ] `NGPT` with `CayleyAdam` vs `NGPT` with `DumbRiemannianAdam`
