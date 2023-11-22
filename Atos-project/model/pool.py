from torch import Tensor, max, mean
from torch.nn import Module

from typing import Optional


class PoolArgs:
    """Arguments necessary to build a pooling module for the model"""
    def __init__(self, emb_out: int, use_max: Optional[bool]=True):
        """
        :param emb_out: a memory of the output size of the pooling operation
        :param use_max: if :code:`False` will use mean pooling instead of max pooling
        """
        self.use_max = use_max
        self.out_size = emb_out


class Pool(Module):
    """Module doing a max pooling or a mean pooling on embeddings to produce a single summary pooling"""
    def __init__(self, emb_out: int, use_max: Optional[bool]=True):
        """Build a pooling module

        :param emb_out: a memory of the output size of the pooling operation
        :param use_max: if :code:`False` will use mean pooling instead of max pooling
        """
        super(Pool, self).__init__()
        self.emb_out = emb_out
        self.use_max = use_max

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass in the pooling module"""
        if self.use_max:
            return max(input, dim=-2)[0]

        else:
            return mean(input, dim=-2)
