from typing import Optional, List, Union

from torch import Tensor, LongTensor
from torch.nn import Module, Embedding

from . import Output, build_linear, Pool


class Model(Module):
    """Model composed of an embedding module, a pooling module, a linear module, and an output module"""
    def __init__(self,
                 dict_len: int,
                 chars_per_line: int,
                 emb_out: int,
                 hidden_layers: Optional[Union[List[Union[int, None]], None]]=None):
        """
        Build a model composed of:

        - an embedding layer
        - a pooling layer
        - a linear layer with or without hidden layers
        - an output layer to rearrange the data

        :param dict_len: length of the dictionary (necessary to produce a correctly shaped output)
        :param chars_per_line: number of characters in the lines of data
        :param emb_out: size of the embedding of each character
        :param hidden_layers: list of values defining the sizes of the hidden layers of the module;
            the number of hidden layers is defined by the length of the list;
            for :code:`Null` or a value inferior to :code:`1`, the output size of the previous layer will be used
        """
        super(Model, self).__init__()

        # Create embeddings layer
        emb_in = dict_len
        self.emb = Embedding(emb_in, emb_out)

        # Create max-pooling layer
        pool_out = emb_out
        self.pool = Pool(pool_out)

        # Create linear treatment layer
        self.lin = build_linear(in_size=pool_out,  # input size is the output size of the pooling layer
                                out_size=chars_per_line * dict_len,  # output size is the number of characters in the
                                # output times the size of the dictionary; what is produced is a probability-like
                                # distribution over the dictionary for each character
                                hidden_sizes=hidden_layers)

        # Create output layer
        self.out = Output(chars_per_line=chars_per_line,
                          dict_len=dict_len)

    def forward(self, input: LongTensor) -> Tensor:
        """Forward pass in the whole model"""
        emb = self.emb(input)
        pooled = self.pool(emb)
        processed = self.lin(pooled)
        output = self.out(processed)

        return output
