import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy import sparse
import psutil
import time
from vortexllm.vortex_llm import VortexLLM

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_memory_efficiency():
    print("\nTesting Memory Efficiency...")
    
    # Record initial memory usage
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Initialize with smaller dimensions for testing
    llm = VortexLLM(vector_dim=64, nearby_size=2)
    
    # Test with increasing data size
    test_sizes = [100, 500, 1000]
    for size in test_sizes:
        print(f"\nProcessing {size} tokens...")
        
        # Generate test data
        text = " ".join([f"word{i}" for i in range(size)])
        
        # Train and process
        start_time = time.time()
        llm.train(text)
        train_time = time.time() - start_time
        
        # Process subset
        process_text = " ".join([f"word{i}" for i in range(min(10, size))])
        start_time = time.time()
        token_ids, scores = llm.process(process_text)
        process_time = time.time() - start_time
        
        # Get memory stats
        current_memory = get_memory_usage()
        memory_increase = current_memory - initial_memory
        
        print(f"Memory usage: {current_memory:.2f} MB (increase: {memory_increase:.2f} MB)")
        print(f"Training time: {train_time:.3f}s")
        print(f"Processing time: {process_time:.3f}s")
        
        # Get component stats
        stats = llm.get_memory_stats()
        print("\nComponent Statistics:")
        for component, data in stats.items():
            if component != 'total_memory_mb':
                print(f"{component}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")

def test_vortex_behavior():
    print("\nTesting Vortex Behavior...")
    
    # Initialize system
    llm = VortexLLM(vector_dim=64, nearby_size=2)
    
    # Test 1: Concept relationship learning
    print("\nTest 1: Concept Relationship Learning")
    train_text = "the quick brown fox jumps over the lazy dog"
    llm.train(train_text)
    
    # Process related concepts
    related_pairs = [
        ("quick brown fox", "jumps"),
        ("lazy dog", "sleeps"),
        ("the quick", "brown")
    ]
    
    print("\nTesting concept relationships:")
    for input_text, target in related_pairs:
        token_ids, scores = llm.process(input_text)
        print(f"\nInput: {input_text}")
        print(f"Activation scores: {scores}")
        
        # Provide feedback
        feedback = np.zeros_like(scores)
        feedback[-1] = 1.0  # Encourage the target connection
        llm.feedback(feedback)
        print("[PASS] Successfully processed and provided feedback")
    
    # Test 2: Path strength adaptation
    print("\nTest 2: Path Strength Adaptation")
    
    # Process same input multiple times
    test_text = "quick brown fox"
    print("\nProcessing same input multiple times:")
    for i in range(3):
        token_ids, scores = llm.process(test_text)
        print(f"Iteration {i+1} scores: {scores}")
        
        # Provide consistent feedback
        feedback = np.array([0.2, 0.5, 0.8])
        llm.feedback(feedback)
        print(f"[PASS] Successfully completed iteration {i+1}")
    
    # Get final statistics
    stats = llm.get_memory_stats()
    print("\nFinal Statistics:")
    for component, data in stats.items():
        if isinstance(data, dict):
            print(f"\n{component}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"{component}: {data}")

if __name__ == "__main__":
    print("Starting VortexLLM Behavior Tests...")
    test_memory_efficiency()
    test_vortex_behavior()
    print("\nAll tests completed successfully!")
