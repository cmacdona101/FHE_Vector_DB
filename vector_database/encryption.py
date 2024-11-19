# vector_database/encryption.py

import tenseal as ts
import pickle
import numpy as np

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
#print(script_dir)

def create_contexts(poly_modulus_degree=32768, coeff_mod_bit_sizes=None, global_scale=2**40, context_dir='data'):
    """
    Creates TenSEAL contexts with and without private keys and saves them to files.

    Args:
        poly_modulus_degree (int): The degree of the polynomial modulus.
        coeff_mod_bit_sizes (list): List of coefficient modulus sizes.
        global_scale (float): The global scale parameter.
        context_dir (str): Directory to save the contexts.

    Returns:
        None
    """
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60] + [40] * 4 + [60]

    # Create context with secret key
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale
    context.generate_galois_keys()
    context.generate_relin_keys()
    
    
    # If context_dir is not an absolute path, make it relative to script_dir
    if not os.path.isabs(context_dir):
        context_dir = os.path.join(script_dir, '..', context_dir)
    #print("context_public_path", context_dir)
        
    # Ensure the context directory exists
    os.makedirs(context_dir, exist_ok=True)
        
    
    # Serialize context with secret key
    context_private = context.serialize(save_secret_key=True)

    # Save context with secret key
    context_private_path = os.path.join(context_dir, 'context_private.bin')
    with open(context_private_path, 'wb') as f:
        f.write(context_private)

    # Remove secret key
    context.make_context_public()

    # Serialize context without secret key
    context_public = context.serialize()

    # Save context without secret key
    context_public_path = os.path.join(context_dir, 'context_public.bin')
    with open(context_public_path, 'wb') as f:
        f.write(context_public)

    print(f"Contexts saved to {context_dir}")


def encrypt_embeddings(embeddings, context_public_path='data/context_public.bin', encrypted_data_path='data/encrypted_vectors.bin'):
    """
    Encrypts embeddings and their inverse norms using the public context and saves them to a file.

    Args:
        embeddings (dict): Dictionary of word embeddings.
        context_public_path (str): Path to the public context file.
        encrypted_data_path (str): Path to save the encrypted embeddings.

    Returns:
        None
    """
    # Load public context
    
    if not os.path.isabs(context_public_path):
        context_public_path = os.path.join(script_dir, '..', context_public_path)
        
    if not os.path.isabs(encrypted_data_path):
        encrypted_data_path = os.path.join(script_dir, '..', encrypted_data_path)
    
    print(context_public_path)
    with open(context_public_path, 'rb') as f:
        context = ts.context_from(f.read())

    # Prepare data for encryption
    encrypted_embeddings = {}

    for word, vector in embeddings.items():
        # Compute the inverse norm
        norm = np.linalg.norm(vector)
        inv_norm = 1.0 / norm

        # Encrypt vector
        encrypted_vector = ts.ckks_vector(context, vector)
        # Encrypt inverse norm
        encrypted_inv_norm = ts.ckks_vector(context, [inv_norm])

        # Serialize encrypted vector and inverse norm
        encrypted_vector_bytes = encrypted_vector.serialize()
        encrypted_inv_norm_bytes = encrypted_inv_norm.serialize()

        # Store in dictionary
        encrypted_embeddings[word] = {
            'encrypted_vector': encrypted_vector_bytes,
            'encrypted_inv_norm': encrypted_inv_norm_bytes
        }

    # Save encrypted embeddings to file
    with open(encrypted_data_path, 'wb') as f:
        pickle.dump(encrypted_embeddings, f)

    print(f"Encrypted embeddings and inverse norms saved to {encrypted_data_path}")


def encrypt_query(query_word, embeddings, context_public_path='data/context_public.bin', encrypted_query_path='data/encrypted_query.bin'):
    """
    Encrypts the query vector and its inverse norm corresponding to the query word using the public context and saves it to a file.

    Args:
        query_word (str): The query word to encrypt.
        embeddings (dict): Dictionary of word embeddings.
        context_public_path (str): Path to the public context file.
        encrypted_query_path (str): Path to save the encrypted query.

    Returns:
        None
    """
    
    if not os.path.isabs(context_public_path):
        context_public_path = os.path.join(script_dir, '..', context_public_path)
    if not os.path.isabs(encrypted_query_path):
        encrypted_query_path = os.path.join(script_dir, '..', encrypted_query_path)
    
    # Ensure the directory exists
    encrypted_query_dir = os.path.dirname(encrypted_query_path)
    os.makedirs(encrypted_query_dir, exist_ok=True)
    
    # Load public context
    with open(context_public_path, 'rb') as f:
        context = ts.context_from(f.read())

    if query_word not in embeddings:
        raise ValueError(f"Query word '{query_word}' not found in embeddings.")

    vector = embeddings[query_word]
    # Compute the inverse norm
    norm = np.linalg.norm(vector)
    inv_norm = 1.0 / norm

    # Encrypt query vector
    encrypted_query = ts.ckks_vector(context, vector)
    # Encrypt inverse norm
    encrypted_inv_norm = ts.ckks_vector(context, [inv_norm])

    # Serialize encrypted query vector and inverse norm
    encrypted_query_bytes = encrypted_query.serialize()
    encrypted_inv_norm_bytes = encrypted_inv_norm.serialize()

    # Save encrypted query vector and inverse norm to file
    with open(encrypted_query_path, 'wb') as f:
        pickle.dump({
            'encrypted_query_vector': encrypted_query_bytes,
            'encrypted_query_inv_norm': encrypted_inv_norm_bytes
        }, f)

    print(f"Encrypted query vector and inverse norm saved to {encrypted_query_path}")
