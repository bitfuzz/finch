import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transcriber import DictationPostProcessor, StreamingTranscriber


class DictationQualityTests(unittest.TestCase):
    def test_filler_custom_words_and_question_cleanup(self):
        p = DictationPostProcessor()

        self.assertEqual(
            p.clean("you whitelist those ips or header based rules? mm."),
            "You whitelist those IPs or header based rules?",
        )
        self.assertEqual(
            p.clean("can you whitelist those ips or header based rules mm"),
            "Can you whitelist those IPs or header based rules?",
        )
        self.assertEqual(p.clean("hello comma world question mark"), "Hello, world?")

    def test_fuzzy_custom_words(self):
        p = DictationPostProcessor()
        self.assertEqual(p.clean("the gpuu is hot"), "The GPU is hot.")

    def test_trim_to_speech_removes_edges(self):
        t = StreamingTranscriber()
        silence = np.zeros(4000, dtype=np.float32)
        speech = np.ones(16000, dtype=np.float32) * 0.02
        trimmed = t._trim_to_speech(np.concatenate([silence, speech, silence]))

        self.assertLess(len(trimmed), 24000)
        self.assertGreater(len(trimmed), 15000)


if __name__ == "__main__":
    unittest.main()
