import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vortexllm.brain_memory import BrainMemory

def test_brain_memory():
    print("Starting BrainMemory tests...")
    
    # Initialize with smaller dimension for testing
    brain = BrainMemory(vector_dim=64)
    
    # Test 1: Adding new concept
    print("\nTest 1: Adding new concept")
    concept_name = "test_concept"
    success = brain.add_concept(concept_name)
    assert success, "Failed to add new concept"
    print("[PASS] Successfully added new concept")
    
    # Test 2: Vector range verification
    print("\nTest 2: Vector range verification")
    vector = brain.get_concept_vector(concept_name)
    assert vector is not None, "Failed to retrieve concept vector"
    assert np.all(vector >= -1) and np.all(vector <= 1), "Vector values outside [-1, 1] range"
    print("[PASS] Vector values within expected range")
    
    # Test 3: Duplicate concept addition
    print("\nTest 3: Duplicate concept test")
    success = brain.add_concept(concept_name)
    assert not success, "Should not allow duplicate concept addition"
    print("[PASS] Successfully prevented duplicate concept addition")
    
    # Test 4: Memory stats
    print("\nTest 4: Memory usage stats")
    num_concepts, memory_mb = brain.get_memory_stats()
    assert num_concepts == 1, "Incorrect concept count"
    print(f"Number of concepts: {num_concepts}")
    print(f"Approximate memory usage: {memory_mb:.2f} MB")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_brain_memory()
