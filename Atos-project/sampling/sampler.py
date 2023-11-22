from typing import Tuple, List, Union

from torch import multinomial
from torch import Tensor
from model import Model
from data.corpus.dictionary import Dictionary


"""Tentative sampler for the model"""


class Sampler(object):
    def __init__(self, model: Model, dictionary: Dictionary):
        self.model = model
        self.dictionary = dictionary

    def __call__(self, sample_number: int, initial_input: Tensor=None) -> List[Tuple[Tensor, str]]:
        if initial_input is None:
            initial_input = self.generate_initial_input()

        samples = [(initial_input, self.to_text(initial_input))]

        for i in range(sample_number):
            samples.append(self.sample(samples[-1][0]))

        return samples

    def sample(self, input: Tensor) -> Tuple[Tensor, str]:
        output = self.model(input)

        # Getting a tensor/list of ids for result
        if self.model.out.output_args.is_linear:
            result = output

        else:
            # Sampling from output probabilities
            result = []
            for probas in output:
                id = multinomial(probas, 1)
                result.append(id)

        text_result = self.to_text(result)

        return Tensor(result), text_result

    def to_text(self, ids: Union[Tensor, List[int]]) -> str:
        """Transform a 1D tensor of ids into a string of characters

        :param ids: 1D tensor or list to transform
        :return: 2D list of characters
        """
        chars = []
        for id in ids:
            chars.append(self.dictionary.get_value(id))

        return ''.join(chars)

    def generate_initial_input(self) -> Tensor:
        return None  # TODO
