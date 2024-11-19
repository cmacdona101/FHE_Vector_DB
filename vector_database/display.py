# vector_database/display.py

import tenseal as ts
import os
import pickle
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = 'data'


def load_encrypted_results(results_path='data/encrypted_results.bin', context_private_path='data/context_private.bin'):
    """
    Loads the encrypted results from file.

    Args:
        results_path (str): Path to the encrypted results file.
        context_private_path (str): Path to the private context file.

    Returns:
        dict: Dictionary of words to encrypted cosine similarity values.
        ts.Context: The private TenSEAL context.
    """

    # Set default paths if not provided
    if results_path is None:
        results_path = os.path.join(script_dir, data_dir, 'encrypted_results.bin')
    if context_private_path is None:
        context_private_path = os.path.join(script_dir, data_dir, 'context_private.bin')

    
    # Load private context
    with open(context_private_path, 'rb') as f:
        context = ts.context_from(f.read())

    # Load encrypted results
    with open(results_path, 'rb') as f:
        encrypted_results_bytes = pickle.load(f)

    # Deserialize encrypted results
    encrypted_results = {}
    for word, enc_bytes in encrypted_results_bytes.items():
        encrypted_value = ts.ckks_vector_from(context, enc_bytes)
        encrypted_results[word] = encrypted_value

    return encrypted_results, context


def decrypt_results(encrypted_results):
    """
    Decrypts the encrypted results.

    Args:
        encrypted_results (dict): Dictionary of words to encrypted cosine similarity values.

    Returns:
        dict: Dictionary of words to decrypted cosine similarity values.
    """
    decrypted_results = {}
    for word, enc_value in encrypted_results.items():
        decrypted_value = enc_value.decrypt()[0]
        decrypted_results[word] = decrypted_value
    return decrypted_results


def compute_plaintext_similarities(embeddings, query_vector):
    """
    Computes the plaintext cosine similarities between the query vector and embeddings.

    Args:
        embeddings (dict): Dictionary of word embeddings.
        query_vector (numpy.array): The query vector.

    Returns:
        dict: Dictionary of words to plaintext cosine similarity values.
    """
    query_norm = np.linalg.norm(query_vector)
    plaintext_results = {}
    for word, vector in embeddings.items():
        dot_product = np.dot(vector, query_vector)
        vector_norm = np.linalg.norm(vector)
        cosine_similarity = dot_product / (vector_norm * query_norm)
        plaintext_results[word] = cosine_similarity
    return plaintext_results


def display_results(decrypted_results, plaintext_results):
    """
    Displays the decrypted and plaintext results, along with their differences.

    Args:
        decrypted_results (dict): Dictionary of words to decrypted cosine similarity values.
        plaintext_results (dict): Dictionary of words to plaintext cosine similarity values.

    Returns:
        None
    """
    print("\nResults:")
    for word in decrypted_results:
        decrypted_value = decrypted_results[word]
        plaintext_value = plaintext_results[word]
        difference = abs(decrypted_value - plaintext_value)
        print(f"Word: {word}")
        print(f"  Decrypted Cosine Similarity^2: {decrypted_value}")
        print(f"  Plaintext Cosine Similarity^2: {plaintext_value}")
        print(f"  Difference: {difference:.8f}\n")
