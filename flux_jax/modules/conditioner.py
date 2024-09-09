# from torch import Tensor, nn
from flax import linen as nn
import jax
from jax import numpy as jnp
from transformers import (FlaxCLIPTextModel, CLIPTokenizer, FlaxT5EncoderModel,
                          T5Tokenizer)
from flax.core.frozen_dict import FrozenDict


class HFEmbedder(nn.Module):
    version: str
    max_length: int
    hf_kwargs: FrozenDict = FrozenDict()

    def setup(self):
        self.is_clip = self.version.startswith("openai")
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(self.version, max_length=self.max_length)
            self.hf_module: FlaxCLIPTextModel = FlaxCLIPTextModel.from_pretrained(self.version, **self.hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(self.version, max_length=self.max_length)
            self.hf_module: FlaxT5EncoderModel = FlaxT5EncoderModel.from_pretrained(self.version, **self.hf_kwargs)

        # self.hf_module = self.hf_module.eval().requires_grad_(False)

    def __call__(self, text: list[str]) -> jnp.ndarray:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="jax",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"],
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
    
if __name__ == "__main__":
    emb = HFEmbedder(version="openai/clip-vit-base-patch32", max_length=77)
    text = ["a photo of a cat", "a photo of a dog"]
    params = emb.init(jax.random.PRNGKey(0), text)
    print(emb.apply(params, text).shape)