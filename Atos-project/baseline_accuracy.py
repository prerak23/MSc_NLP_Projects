from typing import Tuple, List, Dict

from torch import no_grad, Tensor

from data_bkp.firsttest import corpus as corpus_
from trainer import accuracy

"""Scan provided corpus to find baseline accuracy for specific characters"""


def _baseline_accuracy(corpus, value) -> Tuple[float, float, float]:
    with no_grad():
        id = corpus.dictionary.get_id(value)
        fake_proba_distrib = [(0 if i != id else 1) for i in range(len(corpus.dictionary))]
        fake_line = [fake_proba_distrib for i in range(corpus.line_length)]

        fake_output = Tensor(fake_line)

        accuracies = []
        for line in corpus.train:
            accuracies.append(accuracy(fake_output, line))

        mean_accuracy_train = sum(accuracies) / len(accuracies)

        accuracies = []
        for line in corpus.valid:
            accuracies.append(accuracy(fake_output, line))

        mean_accuracy_valid = sum(accuracies) / len(accuracies)

        accuracies = []
        for line in corpus.test:
            accuracies.append(accuracy(fake_output, line))

        mean_accuracy_test = sum(accuracies) / len(accuracies)

    return mean_accuracy_train, mean_accuracy_valid, mean_accuracy_test


def baseline_accuracy(corpus, extra_chars: List[str]=[]) -> Dict[str, Tuple[float, float, float]]:
    chars = [corpus.dictionary.get_value(0), corpus.dictionary.padding_value]
    chars += extra_chars

    accuracies_chars = {}
    for char in chars:
        accuracies = _baseline_accuracy(corpus, char)
        print("Baseline accuracy for char `{}`: training {}; valid {}; test {}".format(
            char, *map(lambda f: round(f,3), accuracies)))
        accuracies_chars[char] = accuracies

    return accuracies_chars


baseline_accuracy(corpus_, [' ', 'a', '0'])

print(corpus_.dictionary.discarded - set(corpus_.dictionary.id_to_value))
