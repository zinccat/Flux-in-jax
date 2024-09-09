import torch
import jax
from jax import numpy as jnp
from torch import Tensor

def jax_to_torch(x: jnp.ndarray) -> Tensor:
    return torch.from_dlpack(jax.dlpack.to_dlpack(x))

def torch_to_jax(x: Tensor) -> jnp.ndarray:
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))

def isclose(a: jnp.ndarray, b: Tensor) -> bool:
    b = b.detach().cpu().numpy()
    return jnp.allclose(a, b, atol=1e-3).all()

if __name__ == "__main__":
    x = jax.numpy.array([1, 2, 3])
    y = jax_to_torch(x)
    z = torch_to_jax(y)
    print(x, y, z)