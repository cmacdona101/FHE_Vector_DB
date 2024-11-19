# vector_database/data_loader.py

import os
import numpy as np


def load_word_embeddings(filename, lines_desired=None):
    """
    Load word embeddings from a file.

    Args:
        filename (str): The filename of the embeddings.
        lines_desired (int, optional): Number of lines to read. Reads all lines if None.

    Returns:
        dict: A dictionary mapping words to their embedding vectors.
    """
    
    
    
    file_path = os.path.join('data', filename)
    #print(file_path)

    embeddings = {}
    with open(file_path, 'r') as file:
        if lines_desired is not None:
            lines = [next(file) for _ in range(lines_desired)]
        else:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            vector_components = parts[1:]
            vector = np.array([float(val) for val in vector_components], dtype=np.float32)
            embeddings[word] = vector
    return embeddings
