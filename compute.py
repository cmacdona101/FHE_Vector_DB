# compute.py
# This step2 takes in  fully encrypted data: the vector embeddings and norms,
# as well as the public context (including the public key), 
#  then performs a FHE cosine similarity calculation, 
# and finally saves the resulting (still encrypted) value to disk for later 
# decryption by a user with the private key.  


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = 'data'

from vector_database.computation import (
    load_encrypted_embeddings,
    load_encrypted_query,
    compute_encrypted_cosine_similarities,
    save_encrypted_results,
)
import time


def main():
    # Step 2: Computation

    # Start timer
    start_time = time.time()

    # Load encrypted embeddings
    print("Loading encrypted embeddings...")
    encrypted_embeddings, context = load_encrypted_embeddings(
        encrypted_data_path=os.path.join(script_dir, data_dir, 'encrypted_vectors.bin'),
        context_public_path=os.path.join(script_dir, data_dir, 'context_public.bin')
    )

    # Load encrypted query vector and inverse norm
    print("Loading encrypted query vector and inverse norm...")
    encrypted_query_vector, encrypted_query_inv_norm = load_encrypted_query(
        encrypted_query_path=os.path.join(script_dir, data_dir, 'encrypted_query.bin'),
        context_public_path=os.path.join(script_dir, data_dir, 'context_public.bin')
    )

    # Compute encrypted cosine similarities
    print("Computing encrypted cosine similarities...")
    encrypted_results = compute_encrypted_cosine_similarities(
        encrypted_query_vector,
        encrypted_query_inv_norm,
        encrypted_embeddings
    )

    # Save encrypted results
    print("Saving encrypted results...")
    save_encrypted_results(
        encrypted_results,
        results_path=os.path.join(script_dir, data_dir, 'encrypted_results.bin')
    )

    # End timer
    end_time = time.time()
    print(f"Computation completed in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
