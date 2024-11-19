import unittest
from unittest.mock import mock_open, patch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_database.data_loader import load_word_embeddings

sample_content = (
            "word1 0.1 0.2 0.3\n"
            "word2 0.4 0.5 0.6\n"
            "word3 0.7 0.8 0.9\n"
        )

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        
        self.expected_embeddings = {
            'word1': np.array([0.1, 0.2, 0.3], dtype=np.float32),
            'word2': np.array([0.4, 0.5, 0.6], dtype=np.float32),
            'word3': np.array([0.7, 0.8, 0.9], dtype=np.float32),
        }

    @patch('builtins.open', new_callable=mock_open, read_data="")
    def test_load_empty_file(self, mock_file):
        embeddings = load_word_embeddings("empty_file.txt")
        self.assertEqual(embeddings, {})

    @patch('builtins.open', new_callable=mock_open, read_data="invalid line\n")
    def test_load_invalid_line(self, mock_file):
        with self.assertRaises(ValueError):
            load_word_embeddings("invalid_file.txt")

    @patch('builtins.open', new_callable=mock_open, read_data=sample_content)
    def test_load_embeddings_all_lines(self, mock_file):
        embeddings = load_word_embeddings("embeddings.txt")
        for word, vector in self.expected_embeddings.items():
            self.assertIn(word, embeddings)
            np.testing.assert_array_almost_equal(embeddings[word], vector)

    @patch('builtins.open', new_callable=mock_open, read_data=sample_content)
    def test_load_embeddings_with_lines_desired(self, mock_file):
        embeddings = load_word_embeddings("embeddings.txt", lines_desired=2)
        expected_words = list(self.expected_embeddings.keys())[:2]
        for word in expected_words:
            self.assertIn(word, embeddings)
        self.assertEqual(len(embeddings), 2)

    @patch('os.path.join', return_value='data/nonexistent.txt')
    def test_file_not_found(self, mock_join):
        with self.assertRaises(FileNotFoundError):
            load_word_embeddings("nonexistent.txt")

    @patch('builtins.open', new_callable=mock_open, read_data=sample_content + "\n\n")
    def test_load_embeddings_with_empty_lines(self, mock_file):
        embeddings = load_word_embeddings("embeddings.txt")
        self.assertEqual(len(embeddings), 3)
        for word, vector in self.expected_embeddings.items():
            self.assertIn(word, embeddings)
            np.testing.assert_array_almost_equal(embeddings[word], vector)

if __name__ == '__main__':
    unittest.main()
