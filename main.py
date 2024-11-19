# main.py

import time

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_database.data_loader import load_word_embeddings
from vector_database.encryption import create_contexts, encrypt_embeddings, encrypt_query


def main():
    # Step 1: Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir='data'

    # Start timer
    start_time = time.time()

    # Create contexts
    print("Creating contexts...")
    create_contexts()

    # Load embeddings
    print("Loading embeddings...")
    
    
    file_name = 'word_embeddings.txt'
    file_dir = os.path.join(script_dir,  data_dir)
    file_dir = os.path.join(file_dir,  file_name)
    embeddings = load_word_embeddings(file_dir)

    # Encrypt embeddings
    print("Encrypting embeddings...")
    encrypt_embeddings(embeddings)

    # Encrypt query word
    query_word = 'king'
    print(f"Encrypting query word '{query_word}'...")
    encrypt_query(query_word, embeddings)

    # End timer
    end_time = time.time()
    print(f"Setup completed in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
