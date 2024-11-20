# FHE_Vector_DB

An open-source vector database implemented using homomorphic encryption with TenSEAL. This project demonstrates how to securely perform computations on encrypted data, calculating cosine similarities between encrypted word embeddings without revealing the underlying data.  

As a proof of concept, a vector database of word embeddings is encrypted using a public key.  The user then "queries" the database with a word ("king", for example), which is converted to vector embeddings using Google's Word2Vec engine and likewise encrypted with the public key.  A cosine similarity search is run on the encrypted vector database using the encrypted word embedding, all without decrypting either sets of information.  

This is done via fully homomorphic encryption, which allows some mathematical operations to operate on encrypted data directly without first needing the decryption -> operation -> re-encryption stages.  The final result is passed back to the user as a list of top k results (as well as several low-ranking words for control comparisons in this proof of concept test). 

Please take a look at example_output.txt for the full example.  

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Step 1: Encryption Setup](#step-1-encryption-setup)
  - [Step 2: Encrypted Computation](#step-2-encrypted-computation)
  - [Step 3: Decryption and Display](#step-3-decryption-and-display)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Notes](#notes)

---

## Features

- **Homomorphic Encryption with TenSEAL**: Utilizes the TenSEAL library to encrypt data and perform computations on encrypted data.
- **Encrypted Cosine Similarity Computation**: Computes cosine similarities between encrypted word embeddings and an encrypted query vector.
- **Secure Querying**: Allows querying the database without exposing the underlying data or query contents.
- **Modular Design**: The project is organized into modules for data loading, encryption, computation, and display.
- **Unit Tests**: Includes unit tests to ensure the correctness of computations.

---

## Getting Started

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/cmacdona101/FHE_Vector_DB.git
   cd FHE_Vector_DB
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. **Obtain Word Embeddings:**
   - Prepare a word embeddings file named `word_embeddings.txt`
   - Each line should contain a word followed by its embedding vector components separated by spaces
   - Example:
     ```
     king 0.1 0.2 0.3 0.4 0.5
     queen 0.2 0.1 0.4 0.3 0.6
     ```

2. **Place the File:**
   - Put `word_embeddings.txt` in the `data/` directory at the root of the project

### Step 1: Encryption Setup

Run the `main.py` script to:
- Load word embeddings
- Create encryption contexts
- Encrypt the embeddings and a query vector

```bash
python main.py
```

**Output:**
```
Creating contexts...
Contexts saved to data
Loading embeddings...
Encrypting embeddings...
Encrypted embeddings and inverse norms saved to data/encrypted_vectors.bin
Encrypting query word 'king'...
Encrypted query vector and inverse norm saved to data/encrypted_query.bin
Setup completed in X.XX seconds.
```

### Step 2: Encrypted Computation

Run the `compute.py` script to:
- Load encrypted embeddings and the encrypted query vector
- Compute encrypted cosine similarities
- Save the encrypted results

```bash
python compute.py
```

**Output:**
```
Loading encrypted embeddings...
Loading encrypted query vector and inverse norm...
Computing encrypted cosine similarities...
Saving encrypted results...
Encrypted results saved to data/encrypted_results.bin
Computation completed in X.XX seconds.
```

### Step 3: Decryption and Display

Run the `display_results.py` script to:
- Load encrypted results and the private context
- Decrypt the results
- Compute plaintext cosine similarities for comparison
- Display the results

```bash
python display_results.py
```

**Output:**
```
Loading encrypted results and private context...
Decrypting results...
Loading embeddings and query vector...
Computing plaintext cosine similarities...
Displaying results...

Results:
Word: king
  Decrypted Cosine Similarity: 1.000000
  Plaintext Cosine Similarity: 1.000000
  Difference: 0.00000000

Word: queen
  Decrypted Cosine Similarity: 0.987654
  Plaintext Cosine Similarity: 0.987654
  Difference: 0.00000012

...

Decryption and display completed in X.XX seconds.
```

## Project Structure

```
FHE_Vector_DB/
├── data/
│   ├── word_embeddings.txt          # Your embeddings file
│   ├── encrypted_vectors.bin        # Encrypted embeddings
│   ├── encrypted_query.bin          # Encrypted query vector
│   ├── encrypted_results.bin        # Encrypted computation results
│   ├── context_public.bin           # Public encryption context
│   └── context_private.bin          # Private encryption context
├── vector_database/
│   ├── __init__.py
│   ├── data_loader.py               # Module for loading embeddings
│   ├── encryption.py                # Module for encryption operations
│   ├── computation.py               # Module for encrypted computations
│   └── display.py                   # Module for decryption and display
├── main.py                          # Script for encryption setup
├── compute.py                       # Script for encrypted computation
├── display_results.py               # Script for decryption and displaying results
├── requirements.txt                 # Project dependencies
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py          # Unit tests for data_loader.py
│   └── test_computation.py          # Unit tests for computation.py
├── README.md                        # Project documentation
└── LICENSE                          # Project license
└── example_output.txt               # Example output showing cosine similarities
```

## Testing

To run the unit tests:

1. Navigate to the project root directory
2. Run the tests using unittest:
   ```bash
   python -m unittest discover tests
   ```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Repository
2. Create a Feature Branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit Your Changes:
   ```bash
   git commit -am 'Add new feature'
   ```
4. Push to the Branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

> **Disclaimer**: This project is for educational purposes to demonstrate homomorphic encryption techniques. It should not be used in production environments without proper security assessments.

## Acknowledgments

- **TenSEAL Library**: An open-source library for homomorphic encryption using Microsoft SEAL and PyTorch/TensorFlow integration
- **Microsoft SEAL**: A cryptographic library that provides implementations of homomorphic encryption schemes

## Contact

For questions or suggestions, please open an issue on GitHub or contact me at -TBD-.

## Notes

- **Performance Considerations**: Homomorphic encryption operations are computationally intensive. Ensure that your environment has sufficient resources.
- **Security Considerations**: Always handle cryptographic keys securely. Do not expose the private context (`context_private.bin`) in insecure environments.
