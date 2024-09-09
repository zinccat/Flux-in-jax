from dataclasses import dataclass

import torch
from einops import rearrange
# from torch import Tensor, nn

from flax import linen as nn
from flax.linen import Module, compact
from flax.linen.initializers import normal, zeros
from jax import numpy as jnp
import jax

from typing import Any, Callable, Optional, Tuple


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


# def swish(x: Tensor) -> Tensor:
#     return x * torch.sigmoid(x)

class AttnBlock(nn.Module):
    in_channels: int

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-6)

        self.q = nn.Conv(self.in_channels, kernel_size=1)
        self.k = nn.Conv(self.in_channels, kernel_size=1)
        self.v = nn.Conv(self.in_channels, kernel_size=1)
        self.proj_out = nn.Conv(self.in_channels, kernel_size=1)

    def attention(self, h_: jnp.ndarray) -> jnp.ndarray:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) 1 c")
        k = rearrange(k, "b c h w -> b (h w) 1 c")
        v = rearrange(v, "b c h w -> b (h w) 1 c")

        h_ = nn.dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b (h w) 1 c -> b c h w", h=h, w=w)

        return h_
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.attention(x)
        h = self.proj_out(h)
        return h
    
class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int

    def setup(self):
        if self.out_channels is None:
            self.out_channels = self.in_channels
        self.norm1 = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv1 = nn.Conv(self.out_channels, kernel_size=(3, 3))
        self.norm2 = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv2 = nn.Conv(self.out_channels, kernel_size=(3, 3))
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv(self.out_channels, kernel_size=1, padding=0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        print(self.name, x.shape)
        h = self.norm1(x)
        h = nn.swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.swish(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        print(self.name, x.shape, h.shape)
        return x + h
    
class Downsample(nn.Module):
    in_channels: int

    def setup(self):
        self.conv = nn.Conv(self.in_channels, kernel_size=(3,3), strides=(2,2), padding=0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))
        x = jnp.pad(x, pad, mode='constant', constant_values=0)
        x = self.conv(x)
        return x
    
class Upsample(nn.Module):
    in_channels: int

    def setup(self):
        self.conv = nn.Conv(self.in_channels, kernel_size=(3, 3), padding=(1, 1))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, height, width, channels = x.shape
        x = jax.image.resize(
            x, (batch, height * 2, width * 2, channels), method='nearest'
        )
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    resolution: int
    in_channels: int
    ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int

    def setup(self):
        self.num_resolutions = len(self.ch_mult)
        self.conv_in = nn.Conv(self.ch, kernel_size=(3,3), strides=(1, 1), padding=1)

        curr_res = self.resolution
        self.in_ch_mult = (1, ) + tuple(self.ch_mult)
        down = []
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = self.ch * self.in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down_dict = {'block': block, 'attn': attn}
            if i_level != self.num_resolutions - 1:
                down_dict['downsample'] = Downsample(block_in)
                curr_res = curr_res // 2
            down.append(down_dict)
        self.down = down

        # middle
        self.mid_block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid_attn_1 = AttnBlock(block_in)
        self.mid_block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv_out = nn.Conv(2 * self.z_channels, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))

    def __call__(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level]['block'][i_block](hs[-1])
                if len(self.down[i_level]['attn']) > 0:
                    h = self.down[i_level]['attn'][i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level]['downsample'](hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)

        # end
        h = self.norm_out(h)
        h = nn.swish(h)
        h = self.conv_out(h)
        return h
    
class Decoder(nn.Module):
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    in_channels: int
    resolution: int
    z_channels: int

    def setup(self):
        self.num_resolutions = len(self.ch_mult)
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = self.ch * self.ch_mult[self.num_resolutions - 1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, self.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv(block_in, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
        # middle
        self.mid_block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid_attn_1 = AttnBlock(block_in)
        self.mid_block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up_dict = {'block': block, 'attn': attn}
            if i_level != 0:
                up_dict['upsample'] = Upsample(block_in)
                curr_res = curr_res * 2
            up.insert(0, up_dict)
        self.up = up

        self.norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6)
        self.conv_out = nn.Conv(self.out_ch, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))

    def __call__(self, x):
        # z to block_in
        h = self.conv_in(x)

        # middle
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level]['block'][i_block](h)
                if len(self.up[i_level]['attn']) > 0:
                    h = self.up[i_level]['attn'][i_block](h)
            if i_level != 0:
                h = self.up[i_level]['upsample'](h)
        
        h = self.norm_out(h)
        h = nn.swish(h)
        h = self.conv_out(h)
        return h
    
class DiagonalGaussian(nn.Module):
    sample: bool = True
    chunk_dim: int = -1 # why original code is 1?

    @compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # TODO: fix RNG
        mean, logvar = jnp.split(z, 2, axis=self.chunk_dim)
        if self.sample:
            return mean + jnp.exp(0.5 * logvar) * jax.random.normal(jax.random.PRNGKey(0), mean.shape)
        else:
            return mean

class AutoEncoder(nn.Module):
    params: AutoEncoderParams

    def setup(self):
        self.encoder = Encoder(
            resolution=self.params.resolution,
            in_channels=self.params.in_channels,
            ch=self.params.ch,
            ch_mult=self.params.ch_mult,
            num_res_blocks=self.params.num_res_blocks,
            z_channels=self.params.z_channels
        )
        self.decoder = Decoder(
            ch=self.params.ch,
            out_ch=self.params.out_ch,
            ch_mult=self.params.ch_mult,
            num_res_blocks=self.params.num_res_blocks,
            in_channels=self.params.in_channels,
            resolution=self.params.resolution,
            z_channels=self.params.z_channels
        )
        self.reg = DiagonalGaussian()
        self.scale_factor = self.params.scale_factor
        self.shift_factor = self.params.shift_factor

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.encoder(x)
        z = self.reg(h)
        z = self.scale_factor * (z - self.shift_factor)
        return z
    
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = self.encode(x)
        return self.decode(z)
    
if __name__ == '__main__':
    # block = ResnetBlock(in_channels, out_channels)
    # x = jnp.ones((1, 4, 32, 32))
    # params = block.init(jax.random.PRNGKey(0), x)
    # y = block.apply(params, x)
    # print(y.shape)

    # enc = Encoder(128, 3, 32, [1, 2, 4, 8], 2, 256)
    # x = jnp.ones((1, 128, 128, 3))
    # params = enc.init(jax.random.PRNGKey(0), x)
    # y = enc.apply(params, x)
    # print(y.shape)

    # dec = Decoder(4, 3, [1, 2, 4, 8], 2, 64, 256, 32)
    # x = jnp.ones((1, 256, 256, 64))
    # params = dec.init(jax.random.PRNGKey(0), x)
    # y = dec.apply(params, x)
    # print(y.shape)

    ae = AutoEncoder(AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ))
    x = jnp.ones((1, 256, 256, 3))
    params = ae.init(jax.random.PRNGKey(0), x)
    y = ae.apply(params, x)
    print(y.shape)