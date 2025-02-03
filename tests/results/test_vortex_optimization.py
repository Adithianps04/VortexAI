import os
import sys
import time
import psutil
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vortexllm.vortex_llm import VortexLLM

def test_optimization():
    """Test VortexLLM optimizations including pruning, checkpointing, and memory management."""
    print("\nVortexLLM Optimization Test Suite")
    print("=" * 50)
    
    # Initialize system
    llm = VortexLLM(vector_dim=64, nearby_size=2, checkpoint_interval=5)
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"\nInitial Memory Usage: {initial_memory:.2f} MB")
    
    # Test 1: Concept Pruning
    print("\nTest 1: Concept Pruning")
    print("-" * 30)
    
    # Add concepts and measure memory
    test_text = "the quick brown fox jumps over the lazy dog " * 20
    print("Adding concepts...")
    start_time = time.time()
    llm.train(test_text)
    train_time = time.time() - start_time
    
    memory_after_train = process.memory_info().rss / 1024 / 1024
    print(f"Training Time: {train_time:.3f}s")
    print(f"Memory After Training: {memory_after_train:.2f} MB")
    print(f"Memory Increase: {memory_after_train - initial_memory:.2f} MB")
    
    # Get brain stats
    stats = llm.get_memory_stats()
    print("\nBrain Memory Stats:")
    for component, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{component}:")
            for metric, metric_value in value.items():
                print(f"  {metric}: {metric_value}")
        else:
            print(f"{component}: {value}")
    
    # Test 2: Auto-Checkpointing
    print("\nTest 2: Auto-Checkpointing")
    print("-" * 30)
    
    # Process text multiple times to trigger checkpoints
    print("Processing text and creating checkpoints...")
    checkpoint_times = []
    for i in range(3):
        start_time = time.time()
        token_ids, scores = llm.process(test_text)
        llm.feedback(np.random.uniform(-1, 1, len(scores)))
        checkpoint_times.append(time.time() - start_time)
        print(f"Iteration {i+1}:")
        print(f"  Processing Time: {checkpoint_times[-1]:.3f}s")
        print(f"  Tokens Processed: {len(token_ids)}")
        print(f"  Average Score: {np.mean(scores):.3f}")
    
    # Check for checkpoint files
    checkpoint_files = os.listdir(llm.checkpoint_dir)
    print(f"\nCheckpoints Created: {len(checkpoint_files)}")
    print(f"Average Checkpoint Time: {np.mean(checkpoint_times):.3f}s")
    
    # Test 3: Memory Management
    print("\nTest 3: Memory Management")
    print("-" * 30)
    
    # Process large text to test memory management
    large_text = "the quick brown fox jumps over the lazy dog " * 100
    print("Processing large text...")
    
    start_time = time.time()
    token_ids, scores = llm.process(large_text)
    process_time = time.time() - start_time
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Processing Time: {process_time:.3f}s")
    print(f"Final Memory Usage: {final_memory:.2f} MB")
    print(f"Total Memory Increase: {final_memory - initial_memory:.2f} MB")
    
    # Get final stats
    final_stats = llm.get_memory_stats()
    print("\nFinal System Stats:")
    for component, value in final_stats.items():
        if isinstance(value, dict):
            print(f"\n{component}:")
            for metric, metric_value in value.items():
                print(f"  {metric}: {metric_value}")
        else:
            print(f"{component}: {value}")
    
    print("\nOptimization Test Summary")
    print("=" * 50)
    print(f"Initial Memory: {initial_memory:.2f} MB")
    print(f"Final Memory: {final_memory:.2f} MB")
    print(f"Memory Efficiency: {(final_memory - initial_memory) / len(token_ids):.4f} MB/token")
    print(f"Processing Speed: {len(token_ids) / process_time:.1f} tokens/second")
    print(f"Checkpoint Overhead: {np.mean(checkpoint_times):.3f}s per operation")
    
    # Cleanup
    for checkpoint in checkpoint_files:
        os.remove(os.path.join(llm.checkpoint_dir, checkpoint))
    os.rmdir(llm.checkpoint_dir)

if __name__ == "__main__":
    test_optimization()
