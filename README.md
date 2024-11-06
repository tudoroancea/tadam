# `tiny-gpt2`

## setup
```bash
uv run train
```
> _Isn't that just wonderful? The simplicity of it all._

## todo
- [x] finish implementing `NGPT`
  - [x] mark weights with `__normalized__`
- [x] implement `DumbRiemannianAdam`
  - [x] add weight decay to use the same optimizer for `NGPT` and `GPT`
  - [x] understand how the scaling learning rate works
- [x] implement `CayleyAdam`
  - [x] implement updates for normalized weights
- [ ] add tqdm in training
- [ ] add better logging for experiments (simple CSV?)

## optimizer versions

- v0: we don't do anything fancy, just renormalize the weights after each optimization step.
- v1: we at least project the descent direction onto the tangent space at the current point, such
  that the normalization corresponds to a proper retraction.
- v2: we always reproject the accumulated 1st order moment onto the tangent space at the current point,
  and naively compute the 2nd order moment accumulation.
- v3: same as v2, but we accumulate the squared norm as the 2nd order moment.
- v4: Cayley Adam (represent tangent vectors with skew symmetric matrices) and use Cayley retraction.
