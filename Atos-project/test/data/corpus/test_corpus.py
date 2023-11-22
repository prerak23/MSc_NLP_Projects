from os.path import realpath, join, dirname, split
from unittest import TestCase, main

import time

from data.corpus.corpus import Corpus
from data.corpus.loaders import MultiFile
from data.corpus.multiqueues import MultiQueues
from data.corpus.processors import EndlineHexDictCropPad, Dictionary, HEX3, CropPad, EndLine
from data.corpus.tensorisers import Batch

_ROOT = dirname(realpath(__file__))
_TEST_FILE = join(*split(_ROOT)[:-1], "raw")


class TestMultiqueues(TestCase):
    def setUp(self):
        self.corpus = Corpus(["1.gz", "2.gz", "3.gz", "4.gz"] * 30,
                             _TEST_FILE,
                             slice(None, -100),
                             slice(-100, -50),
                             slice(-50, None),
                             "<unk>",
                             "─",
                             5,
                             200,
                             slice(None, None),
                             0,
                             200,
                             128,
                             1024,
                             32)

    def test_corpus(self):
        i = 0
        for data, target, status in self.corpus.train(True):
            i += len(data)
            print("-" * 80)
            print(data)
            print(status)


if __name__ == "__main__":
    main()
