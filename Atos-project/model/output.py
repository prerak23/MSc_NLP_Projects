from typing import Optional

from torch import Tensor
from torch.nn import Module


class Output(Module):
    """Module transforming the output of a linear layer (1 dimension) into the expected output of the model (2
    dimensions)"""
    def __init__(self, chars_per_line: int, dict_len: int):
        """Build an output module

        :param chars_per_line: size of the first dimension of the output
        :param dict_len: size of the second dimension of the output
        """
        super(Output, self).__init__()
        self.chars_per_line = chars_per_line
        self.dict_len = dict_len

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass in the output module, "view"-based reshaping"""
        if input.dim() == 2:
            view = input.view(input.size()[0], self.chars_per_line, self.dict_len)
        else:
            view = input.view(self.chars_per_line, self.dict_len)
        return view
