from einops import rearrange

from flax import linen as nn

from jax import numpy as jnp

def attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, pe: jnp.ndarray) -> jnp.ndarray:
    q, k = apply_rope(q, k, pe)

    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")

    x = nn.dot_product_attention(q, k, v)
    x = rearrange(x, "B L H D -> B L (H D)")

    return x

def apply_rope(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    print(xq_.shape, freqs_cis.shape)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape), xk_out.reshape(*xk.shape)

def rope(pos: jnp.ndarray, dim: int, theta: int) -> jnp.ndarray:
    assert dim % 2 == 0
    scale = jnp.arange(0, dim, 2, dtype=jnp.float64) / dim
    omega = 1.0 / (theta**scale)
    out = jnp.einsum("...n,d->...nd", pos, omega)
    out = jnp.stack([jnp.cos(out), -jnp.sin(out), jnp.sin(out), jnp.cos(out)], axis=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out