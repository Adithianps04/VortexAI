import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vortexllm.vortex_llm import VortexLLM

def test_vortex_llm():
    print("Starting VortexLLM integration tests...")
    
    # Initialize with smaller dimensions for testing
    llm = VortexLLM(vector_dim=32, nearby_size=2)
    
    # Test 1: Training
    print("\nTest 1: Training")
    train_text = "the cat sat on the mat"
    llm.train(train_text)
    stats = llm.get_memory_stats()
    print("Memory stats after training:")
    print(f"Brain concepts: {stats['brain_memory']['concepts']}")
    print(f"Tokenizer vocabulary: {stats['tokenizer']['tokens']}")
    print(f"Total memory usage: {stats['total_memory_mb']:.2f} MB")
    
    # Test 2: Processing
    print("\nTest 2: Processing")
    process_text = "the cat sat"
    token_ids, activation_scores = llm.process(process_text)
    assert len(token_ids) == 3, f"Expected 3 tokens, got {len(token_ids)}"
    assert len(activation_scores) == 3, f"Expected 3 activation scores, got {len(activation_scores)}"
    print("[PASS] Successfully processed input")
    print("Token IDs:", token_ids)
    print("Activation scores:", activation_scores)
    
    # Test 3: Feedback
    print("\nTest 3: Feedback")
    feedback_scores = np.array([0.8, 0.6, 0.7])
    llm.feedback(feedback_scores)
    print("[PASS] Successfully provided feedback")
    
    # Test 4: Memory efficiency
    print("\nTest 4: Memory efficiency")
    # Process longer text to test memory usage
    long_text = "the quick brown fox jumps over the lazy dog"
    llm.train(long_text)
    stats = llm.get_memory_stats()
    print("\nFinal memory stats:")
    for component, data in stats.items():
        if component != 'total_memory_mb':
            print(f"\n{component.replace('_', ' ').title()}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
    print(f"\nTotal memory usage: {stats['total_memory_mb']:.2f} MB")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_vortex_llm()
