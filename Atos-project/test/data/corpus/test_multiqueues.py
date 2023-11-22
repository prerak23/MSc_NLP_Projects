from os.path import realpath, join, dirname, split
from unittest import TestCase, main

import time

from data.corpus.loaders import MultiFile
from data.corpus.multiqueues import MultiQueues
from data.corpus.processors import EndlineHexDictCropPad, Dictionary, HEX3, CropPad, EndLine
from data.corpus.tensorisers import Batch

_ROOT = dirname(realpath(__file__))
_TEST_FILE = join(*split(_ROOT)[:-1], "raw")


class TestMultiqueues(TestCase):
    def setUp(self):
        self.file = MultiFile([
            join(_TEST_FILE, "1.gz"),
            join(_TEST_FILE, "2.gz"),
            join(_TEST_FILE, "3.gz"),
            join(_TEST_FILE, "4.gz"),
        ] * 30)

        self.endline = EndLine()
        self.hex = HEX3
        self.dict = Dictionary("<unk>", "─", 1, 200)
        padding_id = self.dict.prepare_dict(self.file.iter_slice(0, 200), [self.endline, self.hex])
        self.crop_pad = CropPad(0, 100, padding_id)

        print(self.dict.dictionary.id_to_value)

        self.processors = [self.endline, self.hex, self.dict, self.crop_pad]

        self.batch = Batch(128, cuda=False)

    def test_multiqueues(self):
        multiqueues = MultiQueues(self.file, self.processors, self.batch, 0, len(self.file), 256, 64)

        i = 0
        iterator = iter(multiqueues)
        for tensor in iterator:
            i += self.batch.batch_size - 1
            print(tensor)
            print(iterator.status())
            print("-" * 80)
            print(i, "/", len(self.file))
            print("-" * 80)

        i = 0
        iterator = iter(multiqueues)
        for tensor in iterator:
            i += self.batch.batch_size - 1
            print(tensor)
            print(iterator.status())
            print("-" * 80)
            print(i, "/", len(self.file))
            print("-" * 80)



if __name__ == "__main__":
    main()
