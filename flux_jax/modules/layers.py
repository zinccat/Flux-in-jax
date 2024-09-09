import math
from dataclasses import dataclass

from einops import rearrange

from flax import linen as nn
from flax.linen import compact

from jax import numpy as jnp

from typing import Tuple

from flux_jax.math import rope, attention

class EmbedND(nn.Module):
    dim: int
    theta: int
    axes_dim: list[int]

    @compact
    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        n_axes = ids.shape[-1]
        emb = jnp.concatenate(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )

        return jnp.expand_dims(emb, axis=1)

def timestep_embedding(t: jnp.ndarray, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)

    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    if jnp.issubdtype(t.dtype, jnp.floating):
        embedding = embedding.astype(t.dtype)
    return embedding

class MLPEmbedder(nn.Module):
    hidden_dim: int
    def setup(self):
        self.in_layer = nn.Dense(self.hidden_dim, use_bias=True)
        self.out_layer = nn.Dense(self.hidden_dim, use_bias=True)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.out_layer(nn.silu(self.in_layer(x)))
    
class QKNorm(nn.Module):
    @compact
    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        q = nn.RMSNorm(name='query_norm')(q)
        k = nn.RMSNorm(name='key_norm')(k)
        return q.astype(v.dtype), k.astype(v.dtype)
class SelfAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False

    def setup(self):
        self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name='qkv')
        self.norm = QKNorm(name='norm')
        self.proj = nn.Dense(self.dim, name='proj')
    
    def __call__(self, x: jnp.ndarray, pe: jnp.ndarray) -> jnp.ndarray:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x
    
    def qkv(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.qkv(x)
    
    def norm(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.norm(q, k, v)
    
@dataclass
class ModulationOut:
    shift: jnp.ndarray
    scale: jnp.ndarray
    gate: jnp.ndarray

class Modulation(nn.Module):
    dim: int
    is_double: bool

    def setup(self):
        self.multiplier = 6 if self.is_double else 3
        self.lin = nn.Dense(self.multiplier * self.dim, use_bias=True)

    def __call__(self, vec: jnp.ndarray) -> tuple[ModulationOut, ModulationOut | None]:
        out = jnp.split(self.lin(nn.silu(vec))[:, None, :], self.multiplier, axis=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class DoubleStreamBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool = False

    def setup(self):
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.img_mod = Modulation(self.hidden_size, is_double=True, name='img_mod')
        self.img_norm1 = nn.LayerNorm(epsilon=1e-6, name='img_norm1')
        self.img_attn = SelfAttention(self.hidden_size, self.num_heads, qkv_bias=self.qkv_bias, name='img_attn')

        self.img_norm2 = nn.LayerNorm(epsilon=1e-6, name='img_norm2')
        self.img_mlp = nn.Sequential([
            nn.Dense(mlp_hidden_dim),
            nn.gelu,
            nn.Dense(self.hidden_size)
        ], name='img_mlp')

        self.txt_mod = Modulation(self.hidden_size, is_double=True, name='txt_mod')
        self.txt_norm1 = nn.LayerNorm(epsilon=1e-6, name='txt_norm1')
        self.txt_attn = SelfAttention(self.hidden_size, self.num_heads, qkv_bias=self.qkv_bias, name='txt_attn')

        self.txt_norm2 = nn.LayerNorm(epsilon=1e-6, name='txt_norm2')
        self.txt_mlp = nn.Sequential([
            nn.Dense(mlp_hidden_dim),
            nn.gelu,
            nn.Dense(self.hidden_size)
        ], name='txt_mlp')

    def __call__(self, img: jnp.ndarray, txt: jnp.ndarray, vec: jnp.ndarray, pe: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Image stream
        img_mod1, img_mod2 = self.img_mod(vec)
        img_norm1 = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_norm1 + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Text stream
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        txt_norm1 = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_norm1 + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # Attention calculation
        q = jnp.concatenate([txt_q, img_q], axis=2)
        k = jnp.concatenate([txt_k, img_k], axis=2)
        v = jnp.concatenate([txt_v, img_v], axis=2)

        attn = attention(q, k, v, pe)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
    
class SingleStreamBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    qk_scale: float | None = None

    def setup(self):
        head_dim = self.hidden_size // self.num_heads
        self.scale = self.qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        self.linear1 = nn.Dense(self.hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Dense(self.hidden_size)

        self.norm = QKNorm()

        self.prenorm = nn.LayerNorm(epsilon=1e-6)

        self.mlp_act = nn.gelu
        self.modulation = Modulation(self.hidden_size, is_double=False)

    def __call__(self, x: jnp.ndarray, vec: jnp.ndarray, pe: jnp.ndarray) -> jnp.ndarray:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.prenorm(x) + mod.shift
        print(self.linear1(x_mod).shape)
        print([self.hidden_size * 3, self.mlp_hidden_dim])
        qkv, mlp = jnp.split(self.linear1(x_mod), [self.hidden_size * 3], axis=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe)
        output = self.linear2(jnp.concatenate([attn, self.mlp_act(mlp)], axis=2))
        return x + mod.gate * output

class LastLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    def setup(self):
        super().__init__()
        self.norm_final = nn.LayerNorm(epsilon=1e-6)
        self.linear = nn.Dense(self.patch_size * self.patch_size * self.out_channels, use_bias=True)
        self.adaLN_modulation = nn.Sequential([nn.silu, nn.Dense(2 * self.hidden_size, use_bias=True)])

    def __call__(self, x: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
        shift, scale = jnp.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x