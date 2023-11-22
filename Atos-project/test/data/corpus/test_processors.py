from unittest import TestCase, main
from data.corpus.processors import DEFAULT_PROCESSORS, CropPad


class TestProcessors(TestCase):
    def setUp(self):
        self.sentence_hex = 'this is a test x0153 sentence 50x6988 for 0x0000005f hexadecimal 0256f840 + 0x8593'
        self.word_hex = '0x6988'
        self.word_no_hex = '4x6988'
        self.sentence_no_hex = 'this is a test 4x0153 sentence 51xg988 without 0y000005f hexadecimal 0256g840 + 0x8*93'
        self.sentence_line_end = 'this is a test sentence with end of line char\n'
        self.sentence_no_line_end = 'this is a test sentence with end of line char'

    def test_regex_hex(self):
        processor = DEFAULT_PROCESSORS["hex3"]

        # Check with a sentence that should be containing hexadecimal codes
        processed_sentence_hex = processor.process(self.sentence_hex)
        present = False
        for tag in processor.tags:
            present = present or tag in processed_sentence_hex
        self.assertTrue(present)

        # Check with a sentence that should not be containing hexadecimal codes
        processed_sentence_no_hex = processor.process(self.sentence_no_hex)
        for tag in processor.tags:
            self.assertNotIn(tag, processed_sentence_no_hex)

        # Check with a word that should be an hexadecimal code
        processed_word_hex = processor.process(self.word_hex)
        present = False
        for tag in processor.tags:
            present = present or tag in processed_word_hex
        self.assertTrue(present)

        # Check with a word that should not be an hexadecimal code
        processed_word_no_hex = processor.process(self.word_no_hex)
        for tag in processor.tags:
            self.assertNotIn(tag, processed_word_no_hex)

    def test_endline(self):
        processor = DEFAULT_PROCESSORS["endline"]

        self.assertEqual(self.sentence_no_line_end, processor.process(self.sentence_line_end))

    def test_crop_pad(self):
        processor = CropPad(10, 0)

        to_pad = [1, 2, 3, 4, 5, 6]
        padded = [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]

        to_crop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        cropped = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        no_operation = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.assertEqual(padded, processor(to_pad))
        self.assertEqual(cropped, processor(to_crop))
        self.assertEqual(no_operation, processor(no_operation))


if __name__ == "__main__":
    main()
