# tests/test_computation.py

import unittest
import numpy as np
import tenseal as ts

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to be tested
from vector_database.computation import (
    encrypted_cosine_similarity,
    compute_encrypted_cosine_similarities
)

class TestComputation(unittest.TestCase):

    def setUp(self):
        # Create a TenSEAL context for testing
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree = 32768, 
            coeff_mod_bit_sizes = [60] + [40]*4 + [60]
        )
        
        self.context.global_scale = 2 ** 40
        
        self.context.generate_galois_keys()
        # Do not make context public in tests, as we need the secret key for decryption

        # Sample plaintext vectors
        self.plain_vector_A = np.array([0.1, 0.2, 0.3])
        self.plain_vector_B = np.array([0.4, 0.5, 0.6])
        self.plain_query_vector = np.array([0.2, 0.3, 0.4])
        self.plain_embeddings = {
            'word1': np.array([0.1, 0.2, 0.3]),
            'word2': np.array([0.4, 0.5, 0.6]),
            'word3': np.array([0.7, 0.8, 0.9])
        }

        # Compute norms and inverse norms
        self.plain_norm_A = np.linalg.norm(self.plain_vector_A)
        self.plain_inv_norm_A = 1.0 / self.plain_norm_A

        self.plain_norm_B = np.linalg.norm(self.plain_vector_B)
        self.plain_inv_norm_B = 1.0 / self.plain_norm_B

        self.plain_query_norm = np.linalg.norm(self.plain_query_vector)
        self.plain_query_inv_norm = 1.0 / self.plain_query_norm

        # Encrypt vectors and inverse norms
        self.encrypted_vector_A = ts.ckks_vector(self.context, self.plain_vector_A)
        self.encrypted_vector_B = ts.ckks_vector(self.context, self.plain_vector_B)
        self.encrypted_inv_norm_A = ts.ckks_vector(self.context, [self.plain_inv_norm_A])
        self.encrypted_inv_norm_B = ts.ckks_vector(self.context, [self.plain_inv_norm_B])

        self.encrypted_query_vector = ts.ckks_vector(self.context, self.plain_query_vector)
        self.encrypted_query_inv_norm = ts.ckks_vector(self.context, [self.plain_query_inv_norm])

        self.encrypted_embeddings = {}
        for word, vector in self.plain_embeddings.items():
            norm = np.linalg.norm(vector)
            inv_norm = 1.0 / norm
            enc_vector = ts.ckks_vector(self.context, vector)
            enc_inv_norm = ts.ckks_vector(self.context, [inv_norm])
            self.encrypted_embeddings[word] = {
                'encrypted_vector': enc_vector,
                'encrypted_inv_norm': enc_inv_norm
            }

    def test_encrypted_cosine_similarity(self):
        # Compute encrypted cosine similarity
        encrypted_cos_sim = encrypted_cosine_similarity(
            self.encrypted_vector_A,
            self.encrypted_vector_B,
            self.encrypted_inv_norm_A,
            self.encrypted_inv_norm_B
        )

        # Decrypt the result
        decrypted_cos_sim = encrypted_cos_sim.decrypt()[0]

        # Compute expected cosine similarity
        dot_product = np.dot(self.plain_vector_A, self.plain_vector_B)
        expected_cos_sim = dot_product * self.plain_inv_norm_A * self.plain_inv_norm_B

        # Assert that the decrypted result is close to the expected value
        self.assertAlmostEqual(decrypted_cos_sim, expected_cos_sim, places=4)

    def test_compute_encrypted_cosine_similarities(self):
        # Compute encrypted cosine similarities
        encrypted_results = compute_encrypted_cosine_similarities(
            self.encrypted_query_vector,
            self.encrypted_query_inv_norm,
            self.encrypted_embeddings
        )

        # Decrypt results and compute expected similarities
        decrypted_results = {}
        for word, enc_cos_sim in encrypted_results.items():
            decrypted_cos_sim = enc_cos_sim.decrypt()[0]
            decrypted_results[word] = decrypted_cos_sim

        # Compute expected results
        expected_results = {}
        for word, vector in self.plain_embeddings.items():
            dot_product = np.dot(self.plain_query_vector, vector)
            norm_vector = np.linalg.norm(vector)
            inv_norm_vector = 1.0 / norm_vector
            expected_cos_sim = dot_product * self.plain_query_inv_norm * inv_norm_vector
            expected_results[word] = expected_cos_sim

        # Compare decrypted results with expected results
        for word in self.plain_embeddings:
            self.assertAlmostEqual(
                decrypted_results[word],
                expected_results[word],
                places=4,
                msg=f"Mismatch in cosine similarity for {word}"
            )

if __name__ == '__main__':
    unittest.main()
