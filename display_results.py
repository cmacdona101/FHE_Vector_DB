# display_results.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_database.display import (
    load_encrypted_results,
    decrypt_results,
    compute_plaintext_similarities,
    display_results,
)

from vector_database.data_loader import load_word_embeddings
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = 'data'

def main():
    # Step 3: Decryption and Display

    # Start timer
    start_time = time.time()

    # Load encrypted results and private context
    print("Loading encrypted results and private context...")
    encrypted_results, context = load_encrypted_results(
        results_path=os.path.join(script_dir, data_dir, 'encrypted_results.bin'),
        context_private_path=os.path.join(script_dir, data_dir, 'context_private.bin')
    )

    # Decrypt results
    print("Decrypting results...")
    decrypted_results = decrypt_results(encrypted_results)

    # Load embeddings and query vector
    print("Loading embeddings and query vector...")
    embeddings_path = os.path.join(script_dir, data_dir, 'word_embeddings.txt')
    embeddings = load_word_embeddings(embeddings_path)
    query_word = 'king'
    query_vector = embeddings[query_word]

    # Compute plaintext cosine similarities
    print("Computing plaintext cosine similarities...")
    plaintext_results = compute_plaintext_similarities(embeddings, query_vector)

    # Display results
    print("Displaying results...")
    display_results(decrypted_results, plaintext_results)

    # End timer
    end_time = time.time()
    print(f"Decryption and display completed in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
