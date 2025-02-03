import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vortexllm.custom_tokenizer import CustomTokenizer

def test_tokenizer():
    print("Starting CustomTokenizer tests...")
    
    # Initialize with smaller dimensions for testing
    tokenizer = CustomTokenizer(vector_dim=32, nearby_size=2)
    
    # Test 1: Basic tokenization and training
    print("\nTest 1: Basic tokenization and training")
    test_text = "the cat sat on the mat"
    tokenizer.train_tokens(test_text)
    assert len(tokenizer.token_to_id) == 5, f"Expected 5 unique tokens, got {len(tokenizer.token_to_id)}"
    print("[PASS] Successfully trained tokens")
    print(f"Vocabulary: {tokenizer.token_to_id}")
    
    # Test 2: Token encoding
    print("\nTest 2: Token encoding")
    token_ids, relations = tokenizer.encode_tokens(test_text)
    assert len(token_ids) == 6, f"Expected 6 tokens, got {len(token_ids)}"
    print("[PASS] Successfully encoded tokens")
    print(f"Token IDs: {token_ids}")
    
    # Test 3: Relation matrix properties
    print("\nTest 3: Relation matrix properties")
    assert relations.shape == (len(token_ids), len(token_ids)), "Incorrect relation matrix shape"
    assert np.all(relations.data >= 0) and np.all(relations.data <= 1), "Relations outside [0,1] range"
    print("[PASS] Relation matrix properties verified")
    print(f"Relation matrix shape: {relations.shape}")
    print("Non-zero relations:")
    print(relations.todense())
    
    # Test 4: Memory usage
    print("\nTest 4: Memory usage")
    num_tokens, memory_mb = tokenizer.get_stats()
    print(f"Number of tokens: {num_tokens}")
    print(f"Approximate memory usage: {memory_mb:.2f} MB")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_tokenizer()
