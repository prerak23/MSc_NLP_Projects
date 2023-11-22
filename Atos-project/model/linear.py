from torch.nn import Linear, Sequential, Module, ReLU

from typing import Optional, Union, List


class Linear1Hidden(Sequential):
    """Fully connected module, with an input layer and an output layer, and a single hidden layer"""

    def __init__(self, in_size: int, out_size: int, hidden_size: Optional[int] = -1):
        """Build a fully connected module, with an input layer and an output layer, and a single hidden layer
        (so 2 fully connected input to output layers).

        The activation function is ReLU.

        :param in_size: Input size of the input module
        :param out_size: Output size of the input module
        :param hidden_size: Number of neurons in the hidden layer
        (any number under 1 is considered as a wildcard and will be replaced by the size of the input)

        :shape:
            :input: (N, *, in_size) where * means any number of additional dimensions
            :output: (N, *, out_size) where all but the last dimension are the same shape as the input.
        """

        # If a wildcard is given for the hidden layer's size, we use input size as hidden size
        if hidden_size <= 0:
            hidden_size = in_size

        # Initialise the sequence with two fully connected layers with a ReLU in-between
        super(Linear1Hidden, self).__init__(
            Linear(in_size, hidden_size),
            ReLU(),
            Linear(hidden_size, out_size)
        )


class Linear2Hidden(Sequential):
    """Fully connected module, with an input layer and an output layer, and two hidden layers"""
    def __init__(self, in_size: int, out_size: int,
                 first_hidden_size: Optional[int] = -1,
                 second_hidden_size: Optional[int] = -1):
        """Build a fully connected module, with an input layer and an output layer, and two hidden layers
        (so 3 fully connected input to output layers).

        The activation function is ReLU.

        :param in_size: Input size of the input module
        :param out_size: Output size of the input module
        :param first_hidden_size: Number of neurons in the first hidden layer
        (any number under 1 is considered as a wildcard and will be replaced by the size of the input)
        :param second_hidden_size: Number of neurons in the second hidden layer
        (any number under 1 is considered as a wildcard and will be replaced by the number of neurons in the first
        hidden layer)

        :shape:
            :input: (N, *, in_size) where * means any number of additional dimensions
            :output: (N, *, out_size) where all but the last dimension are the same shape as the input.
        """

        # If a wildcard is given for the first hidden layer's size, we use input size instead
        if first_hidden_size <= 0:
            first_hidden_size = in_size

        # If a wildcard is given for the second hidden layer's size, we use fist hidden size instead
        if second_hidden_size <= 0:
            second_hidden_size = first_hidden_size

        # Initialise the sequence with three fully connected layers with ReLUs in-between
        super(Linear2Hidden, self).__init__(
            Linear(in_size, first_hidden_size),
            ReLU(),
            Linear(first_hidden_size, second_hidden_size),
            ReLU(),
            Linear(second_hidden_size, out_size)
        )


class LinearNHidden(Sequential):
    """Generic fully connected module, with an input layer and an output layer, and any number of hidden layers"""

    def __init__(self, in_size: int, out_size: int,
                 hidden_sizes: Optional[List[int]] = []):
        """Build a fully connected module, with an input layer and an output layer, and any number of hidden layers
        (for H hidden layers, it has H + 1 fully connected input to output layers).

        The activation function is ReLU.

        :param in_size: Input size of the input module
        :param out_size: Output size of the input module
        :param hidden_sizes: List of the number of neurons in the hidden layers, from the closest to the input to the
        closest to the output (any number under 1 is considered as a wildcard and will be replaced by the size of the
        previous layer)

        :shape:
            :input: (N, *, in_size) where * means any number of additional dimensions
            :output: (N, *, out_size) where all but the last dimension are the same shape as the input.
        """

        # If a wildcard is given for a hidden layer's size, we use the previous layer's size instead
        # For the first hidden layer, we use the input layer's size as a replacement
        hidden_sizes = [
            hidden_size if hidden_size > 0 else  # we use the given value if it is valid (not wildcard)
            hidden_sizes[i - 1] if i > 0 else    # we use the previous previous layer's size if given value is not valid
            in_size                              # we use the input layer's size as the first hidden layer's replacement
            for i, hidden_size in enumerate(hidden_sizes)
        ]

        # We create a sequence of modules, with fully connected layers and ReLUs in-between
        modules = []

        # We store the length because it will be used multiple times
        length = len(hidden_sizes)

        # If we have no hidden layer, we use a simple fully connected layer
        if length < 1:

            # We add the fully connected layer
            modules += [Linear(in_size, out_size)]

        # If we have hidden layers, we use an alternation of linear layers and ReLU activations
        else:

            # We add the input layer
            modules += [Linear(in_size, hidden_sizes[0])]

            # We add all the hidden layers (including the hidden to output layer)
            for i in range(length):
                modules += [ReLU(),  # we add the ReLU
                            Linear(hidden_sizes[i],
                                   hidden_sizes[i + 1] if i < length else out_size)]

        # We use the sequence of modules to build the linear module
        super(LinearNHidden, self).__init__(*modules)


def build_linear(in_size: int, out_size: int, hidden_sizes: Optional[Union[List[Union[int]], None]] = None) -> Module:
    """Build and return a fully connected module, with an input layer and an output layer, and any number of hidden
    layers.

    If :code:`hidden_sizes` is :code:`None` or an empty list,
    a simple fully connected layer will be used (see torch.nn.Linear).

    If :code:`hidden_sizes` is 1 element long,
    a fully connected layer with a hidden layer will be used (see Linear1Hidden).

    If :code:`hidden_sizes` is 2 element long,
    a fully connected layer with two hidden layers will be used (see Linear2Hidden).

    If :code:`hidden_sizes` is more than 2 element long,
    a generic fully connected layer with any number of hidden layers will be used (see LinearNHidden).

    In every case, the activation function is ReLU.

    :param in_size: Input size of the input module
    :param out_size: Output size of the input module
    :param hidden_sizes: None or list of the number of neurons in the hidden layers, from the closest to the input to
    the closest to the output (any number under 1 is considered as a wildcard and will be replaced by the size of the
    previous layer)

    :shape:
        :input: (N, *, in_size) where * means any number of additional dimensions
        :output: (N, *, out_size) where all but the last dimension are the same shape as the input.
    """

    if hidden_sizes is None or len(hidden_sizes) < 1:
        # We use the linear class without hidden layer
        return Linear(in_size, out_size)

    elif len(hidden_sizes) == 1:
        # We use the linear class for 1 hidden layer
        return Linear1Hidden(in_size, out_size, hidden_sizes[0])

    elif len(hidden_sizes) == 2:
        # We use the linear class for 2 hidden layer
        return Linear2Hidden(in_size, out_size, hidden_sizes[0], hidden_sizes[1])

    else:
        # In any other case, we use the generic linear class
        return LinearNHidden(in_size, out_size, hidden_sizes)
