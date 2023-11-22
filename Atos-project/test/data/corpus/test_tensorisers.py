from data.corpus.tensorisers import AbstractTensoriser, Batch
from unittest import TestCase, main
from torch import equal


class TestTensorisers(TestCase):
    def setUp(self):
        self.size = [40, 10]

        self.data = [
            list(range(n, n + self.size[1])) for n in range(self.size[0])
        ]

        print(self.data)

    def test_batch(self):
        batch = Batch(15, overlap=1)

        batches = []
        end_batch = None
        for line in self.data:
            batch.put(line)
            if batch.is_full() and batch.can_produce():
                batches.append(batch.produce())

        if batch.can_produce():
            end_batch = batch.produce()

        end_batch_size = len(self.data) % (batch.batch_size - batch.overlap)
        if end_batch_size > 0:
            self.assertIsNotNone(end_batch)
            self.assertTrue(equal(batches[-1][-1], end_batch[0]))
            self.assertEqual(end_batch_size, len(end_batch))
        else:
            self.assertIsNone(end_batch)

        self.assertTrue(equal(batches[0][-1], batches[1][0]))
        self.assertEqual(15, len(batches[0]))



if __name__ == "__main__":
    main()
