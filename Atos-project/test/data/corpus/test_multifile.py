from unittest import TestCase, main

from data.corpus.loaders.multifile import *
from os.path import realpath, join, dirname, split

_ROOT = dirname(realpath(__file__))
_TEST_FILE = join(*split(_ROOT)[:-1], "raw")


class TestMultiFile(TestCase):
    def setUp(self):
        self.file = MultiFile([
            join(_TEST_FILE, "1.gz"),
            join(_TEST_FILE, "2.gz"),
            join(_TEST_FILE, "3.gz"),
            join(_TEST_FILE, "4.gz"),
        ])

    def test_existence(self):
        self.assertIsNotNone(self.file)

    def test_len(self):
        self.assertEqual(len(self.file), 100)

    def test_contains(self):

        print( self.file.files)
        self.assertNotIn(-5, self.file)
        self.assertIn(0, self.file)
        self.assertIn(5, self.file)
        self.assertIn(19, self.file)
        self.assertIn(34, self.file)
        self.assertIn(64, self.file)
        self.assertIn(99, self.file)
        self.assertNotIn(200, self.file)

    def test_iter(self):
        reference_1 = [line for line in File(join(_TEST_FILE, "1.gz"))]
        reference_2 = [line for line in File(join(_TEST_FILE, "2.gz"))]
        reference_3 = [line for line in File(join(_TEST_FILE, "3.gz"))]
        reference_4 = [line for line in File(join(_TEST_FILE, "4.gz"))]

        reference = reference_1 + reference_2 + reference_3 + reference_4

        i = 0
        for line in self.file:
            if i in range(0, 20):
                self.assertEqual(reference_1[i], line)
            elif i in range(20, 40):
                self.assertEqual(reference_2[i - 20], line)
            elif i in range(40, 60):
                self.assertEqual(reference_3[i - 40], line)
            elif i in range(60, 100):
                self.assertEqual(reference_4[i - 100], line)

            self.assertEqual(reference[i], line)
            i += 1

    def test_iter_slice(self):
        reference_1 = [line for line in File(join(_TEST_FILE, "1.gz"))]
        reference_2 = [line for line in File(join(_TEST_FILE, "2.gz"))]
        reference_3 = [line for line in File(join(_TEST_FILE, "3.gz"))]
        reference_4 = [line for line in File(join(_TEST_FILE, "4.gz"))]

        reference = reference_1 + reference_2 + reference_3 + reference_4
        reference = reference[5:75]

        self.assertEqual(len(reference), len([line for line in self.file.iter_slice(5, 75)]))

        i = 0  # relative index
        j = 5  # absolute index
        for line in self.file.iter_slice(5, 75):
            if j in range(0, 20):
                self.assertEqual(reference_1[j], line)
            elif j in range(20, 40):
                self.assertEqual(reference_2[j - 20], line)
            elif j in range(40, 60):
                self.assertEqual(reference_3[j - 40], line)
            elif j in range(60, 100):
                self.assertEqual(reference_4[j - 100], line)

            self.assertEqual(reference[i], line)
            i += 1
            j += 1


if __name__ == "__main__":
    main()
