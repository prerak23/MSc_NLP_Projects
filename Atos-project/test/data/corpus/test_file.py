from unittest import TestCase, main

from data.corpus.loaders.file import *
from os.path import realpath, join, dirname, split

_ROOT = dirname(realpath(__file__))
_TEST_FILE = join(*split(_ROOT)[:-1], "raw")


class TestFile(TestCase):
    def setUp(self):
        self.file_txt = File(join(_TEST_FILE, "1.txt"))
        self.file_gz = File(join(_TEST_FILE, "1.gz"))
        self.file_gz_pad = File(join(_TEST_FILE, "1.gz"), padding=10)

    def test_existence(self):
        self.assertIsNotNone(self.file_txt)
        self.assertIsNotNone(self.file_gz)
        self.assertIsNotNone(self.file_gz_pad)

    def test_len(self):
        self.assertEqual(len(self.file_txt), 20)
        self.assertEqual(len(self.file_gz), 20)
        self.assertEqual(len(self.file_gz_pad), 20)

    def test_contains(self):
        self.assertIn(0, self.file_gz)
        self.assertIn(5, self.file_gz)
        self.assertIn(19, self.file_gz)
        self.assertNotIn(20, self.file_gz)

        self.assertNotIn(5, self.file_gz_pad)
        self.assertIn(10, self.file_gz_pad)
        self.assertIn(29, self.file_gz_pad)
        self.assertNotIn(31, self.file_gz_pad)

    def test_iter(self):
        reference = [line for line in self.file_txt]

        i = 0
        for line in self.file_gz:
            self.assertEqual(reference[i], line)
            i += 1

        i = 0
        for line in self.file_gz_pad:
            self.assertEqual(reference[i], line)
            i += 1

    def test_iter_slice(self):
        reference = [line for line in self.file_txt]
        reference = reference[5:15]

        self.assertEqual(len(reference), len([line for line in self.file_gz.iter_slice(5, 15)]))
        self.assertEqual(5, len([line for line in self.file_gz_pad.iter_slice(5, 15)]))

        i = 0
        for line in self.file_gz.iter_slice(5, 15):
            self.assertEqual(reference[i], line)
            i += 1

        i = 0
        for line in self.file_gz_pad.iter_slice(5 + 10, 15 + 10):
            self.assertEqual(reference[i], line)
            i += 1


if __name__ == "__main__":
    main()
