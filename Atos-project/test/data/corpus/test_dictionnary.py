from logging import info
from unittest import TestCase

from data.corpus.dictionary import Dictionary


class DictionaryTest(TestCase):
    def test_char(self):
        chars = """no it was n't black monday
but while the new york stock exchange did n't fall apart friday as the dow jones industrial average plunged N
points most of it in the final hour it barely managed to stay this side of chaos
some circuit breakers installed after the october N crash failed their first test traders say unable to cool the
selling panic in both stocks and futures
the N stock specialist firms on the big board floor the buyers and sellers of last resort who were criticized
after the N crash once again could n't handle the selling pressure
big investment banks refused to step up to the plate to support the beleaguered floor traders by buying big blocks
of stock traders say
heavy selling of standard & poor 's 500-stock index futures in chicago"""

        dict = Dictionary(replace_value="<unk>")
        for line in chars.split('\n'):
            for char in line:
                dict.add(char)
            info(line)
            info({char: chars.count(char) for char in set(line)})

        dict.crop(minimal_occurrences=5, max_length=9)
        self.assertLessEqual(len(dict), 9 + 1)


if __name__ == "__main__":
    main()
