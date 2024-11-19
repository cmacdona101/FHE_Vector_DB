# vector_database/computation.py

import tenseal as ts
import os
import pickle


def load_encrypted_embeddings(encrypted_data_path='data/encrypted_vectors.bin', context_public_path='data/context_public.bin'):
    """
    Loads the encrypted embeddings and inverse norms from file.

    Args:
        encrypted_data_path (str): Path to the encrypted embeddings file.
        context_public_path (str): Path to the public context file.

    Returns:
        dict: A dictionary mapping words to encrypted vectors and inverse norms.
        ts.Context: The public TenSEAL context.
    """
    # Load public context
    with open(context_public_path, 'rb') as f:
        context = ts.context_from(f.read())

    # Load encrypted embeddings
    with open(encrypted_data_path, 'rb') as f:
        encrypted_embeddings_bytes = pickle.load(f)

    # Deserialize encrypted embeddings and inverse norms
    encrypted_embeddings = {}
    for word, enc_data in encrypted_embeddings_bytes.items():
        encrypted_vector = ts.ckks_vector_from(context, enc_data['encrypted_vector'])
        encrypted_inv_norm = ts.ckks_vector_from(context, enc_data['encrypted_inv_norm'])
        encrypted_embeddings[word] = {
            'encrypted_vector': encrypted_vector,
            'encrypted_inv_norm': encrypted_inv_norm
        }

    return encrypted_embeddings, context


def load_encrypted_query(encrypted_query_path='data/encrypted_query.bin', context_public_path='data/context_public.bin'):
    """
    Loads the encrypted query vector and inverse norm from file.

    Args:
        encrypted_query_path (str): Path to the encrypted query file.
        context_public_path (str): Path to the public context file.

    Returns:
        tuple: The encrypted query vector and encrypted inverse norm.
    """
    # Load public context
    with open(context_public_path, 'rb') as f:
        context = ts.context_from(f.read())

    # Load encrypted query
    with open(encrypted_query_path, 'rb') as f:
        encrypted_query_data = pickle.load(f)

    # Deserialize encrypted query vector and inverse norm
    encrypted_query_vector = ts.ckks_vector_from(context, encrypted_query_data['encrypted_query_vector'])
    encrypted_query_inv_norm = ts.ckks_vector_from(context, encrypted_query_data['encrypted_query_inv_norm'])

    return encrypted_query_vector, encrypted_query_inv_norm


def encrypted_cosine_similarity(encrypted_vector_A, encrypted_vector_B, encrypted_inv_norm_A, encrypted_inv_norm_B):
    """
    Computes the cosine similarity between two encrypted vectors using homomorphic encryption.

    Args:
        encrypted_vector_A (CKKSVector): The first encrypted vector.
        encrypted_vector_B (CKKSVector): The second encrypted vector.
        encrypted_inv_norm_A (CKKSVector): The encrypted inverse norm of vector A.
        encrypted_inv_norm_B (CKKSVector): The encrypted inverse norm of vector B.

    Returns:
        CKKSVector: The encrypted cosine similarity.
    """
    # Compute the dot product
    encrypted_dot_product = (encrypted_vector_A * encrypted_vector_B).sum()

    # Calculate the cosine similarity
    encrypted_cosine_similarity = encrypted_dot_product * encrypted_inv_norm_A * encrypted_inv_norm_B

    return encrypted_cosine_similarity


def compute_encrypted_cosine_similarities(encrypted_query_vector, encrypted_query_inv_norm, encrypted_embeddings):
    """
    Computes cosine similarities between the encrypted query and encrypted embeddings using pre-encrypted inverse norms.

    Args:
        encrypted_query_vector (CKKSVector): The encrypted query vector.
        encrypted_query_inv_norm (CKKSVector): The encrypted inverse norm of the query vector.
        encrypted_embeddings (dict): Dictionary of encrypted embeddings and inverse norms.

    Returns:
        dict: A dictionary mapping words to encrypted cosine similarity values.
    """
    encrypted_cosine_similarities = {}

    for word, enc_data in encrypted_embeddings.items():
        enc_vector = enc_data['encrypted_vector']
        enc_inv_norm = enc_data['encrypted_inv_norm']

        # Compute encrypted cosine similarity using pre-encrypted inverse norms
        encrypted_cos_sim = encrypted_cosine_similarity(
            encrypted_query_vector, enc_vector, encrypted_query_inv_norm, enc_inv_norm
        )

        encrypted_cosine_similarities[word] = encrypted_cos_sim

    return encrypted_cosine_similarities


def save_encrypted_results(encrypted_results, results_path='data/encrypted_results.bin'):
    """
    Saves the encrypted results to a file.

    Args:
        encrypted_results (dict): Dictionary of words to encrypted cosine similarity values.
        results_path (str): Path to save the encrypted results.

    Returns:
        None
    """
    # Serialize encrypted results
    encrypted_results_bytes = {}
    for word, enc_value in encrypted_results.items():
        enc_value_bytes = enc_value.serialize()
        encrypted_results_bytes[word] = enc_value_bytes

    # Save to file
    with open(results_path, 'wb') as f:
        pickle.dump(encrypted_results_bytes, f)

    print(f"Encrypted results saved to {results_path}")
