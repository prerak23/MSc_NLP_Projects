from typing import Optional

from torch import Tensor, argmax, eq, sum
from torch.nn import CrossEntropyLoss

from data.corpus.dictionary import Dictionary

"""CrossEntropy
"If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes"

Using dictionary counter as weight could be a good idea (at leas as initialisation)

------------------------------------------------------------------------------------------------------------------------
"The input is expected to contain scores for each class.

input has to be a Tensor of size either (minibatch,C)
or (minibatch,C,d1,d2,...,dK) with K≥2 for the K-dimensional case (described later)."

with embeddings, a line is either (1 (1 line), c (c characters), d1, ... dK (K possible dictionary entries)) or 
(c (c characters), d1, ... dK (K possible dictionary entries))

"This criterion expects a class index (0 to C-1) as the target for each value of a 1D tensor of size minibatch"

minibatch should be c, the number of characters by line
------------------------------------------------------------------------------------------------------------------------
"""

"""NLL
reduce (bool, optional) – By default, the losses are averaged or summed for each minibatch. When reduce is False, the 
loss function returns a loss per batch instead and ignores size_average. Default: True
"""


class LossArgs:
    def __init__(self, dictionary: Optional[Dictionary] = None):
        if dictionary is not None:
            sorted_by_index_counter = sorted(dictionary.counter.items(), key=lambda key_value: key_value[0])
            self.weight_init = Tensor(sorted_by_index_counter)
        else:
            self.weight_init = None


# CrossEntropyLoss
class Loss(CrossEntropyLoss):
    def __init__(self, loss_args: LossArgs):
        if loss_args.weight_init is None or True:  # TODO
            super(Loss, self).__init__()
        else:
            super(Loss, self).__init__(weight=loss_args.weight_init)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.size(0) == target.size(0), \
            "minibatch/character dimension do not match: expected {} got {}".format(target.size(0), input.size(0))
        if target.dim() > 1:
            loss = super(Loss, self).forward(input[0], target[0])
            for i in range(1, input.size(0)):
                loss += super(Loss, self).forward(input[i], target[i])

            loss /= input.size(0)
        else:
            loss = super(Loss, self).forward(input, target)
        return loss


def accuracy(output: Tensor, target: Tensor) -> float:
    """Evaluates accuracy

    accuracy = number_chars_correctly_predicted / total_number_of_chars
    """
    chars_predicted = argmax(output, dim=-1)
    chars_correctly_predicted = eq(chars_predicted, target)
    number_chars_correctly_predicted = int(sum(chars_correctly_predicted))
    total_number_of_chars = chars_correctly_predicted.numel()
    accuracy = number_chars_correctly_predicted / total_number_of_chars

    return accuracy
