import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy import sparse
from vortexllm.ahamov_net import AhamovNet

def test_ahamov():
    print("Starting AhamovNet tests...")
    
    # Initialize with smaller dimensions for testing
    net = AhamovNet(vector_dim=32)
    
    # Test 1: Basic processing
    print("\nTest 1: Basic processing")
    token_ids = [1, 2, 3]
    concept_vectors = np.random.uniform(-1, 1, (3, 32))
    relation_weights = sparse.csr_matrix([[0, 0.5, 0.3],
                                        [0.5, 0, 0.4],
                                        [0.3, 0.4, 0]])
    vortex_influences = np.array([0.8, 0.9, 0.7])
    
    activation_scores = net.process(token_ids, concept_vectors, 
                                  relation_weights, vortex_influences, 
                                  time=0.0)
    
    assert len(activation_scores) == len(token_ids), "Incorrect number of activation scores"
    assert np.all(activation_scores >= -1) and np.all(activation_scores <= 1), \
        "Activation scores outside [-1, 1] range"
    print("[PASS] Successfully processed inputs")
    print("Activation scores:", activation_scores)
    
    # Test 2: Adaptation
    print("\nTest 2: Adaptation")
    feedback_scores = np.array([0.9, 0.5, 0.7])
    net.adapt(feedback_scores)
    
    # Get some connection weights for verification
    connection_key = (token_ids[0], token_ids[1])
    weights = net.connection_weights[connection_key]
    assert np.all(weights >= -1) and np.all(weights <= 1), \
        "Weights outside [-1, 1] range after adaptation"
    print("[PASS] Successfully adapted weights")
    
    # Test 3: Memory usage
    print("\nTest 3: Memory usage")
    stats = net.get_stats()
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test 4: Multiple processing steps
    print("\nTest 4: Multiple processing steps")
    for i in range(3):
        activation_scores = net.process(token_ids, concept_vectors,
                                     relation_weights, vortex_influences,
                                     time=float(i))
        print(f"Step {i} activation scores:", activation_scores)
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_ahamov()
