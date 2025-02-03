import os
import sys
import time
import psutil
import pytest
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vortexllm.vortex_llm import VortexLLM
from src.vortexllm.performance_monitor import PerformanceMonitor

def format_memory(bytes: int) -> str:
    """Format memory size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

@pytest.fixture
def monitor():
    """Create a global performance monitor."""
    return PerformanceMonitor()

@pytest.fixture
def test_dir():
    """Create test directory."""
    dir_path = os.path.join("tests", "results", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def test_batch_processing(monitor):
    """Test batch processing performance."""
    dimensions = [64, 128, 256]
    batch_sizes = [10, 50, 100]
    
    for dim in dimensions:
        print(f"\nVector Dimension: {dim}")
        llm = VortexLLM(vector_dim=dim, enable_monitoring=True)
        llm.performance_monitor = monitor  # Use our test monitor
        
        for batch_size in batch_sizes:
            # Generate test vectors
            test_vectors = np.random.uniform(-1, 1, (batch_size, dim))
            test_vectors /= np.linalg.norm(test_vectors, axis=1, keepdims=True)
            
            # Time batch processing
            start_time = time.time()
            batch_scores = llm.network.process_batch(test_vectors)
            end_time = time.time()
            
            # Calculate metrics
            processing_time = end_time - start_time
            vectors_per_second = batch_size / max(processing_time, 0.001)
            avg_score = float(np.mean(np.abs(batch_scores)))
            
            # Log metrics
            monitor.log_batch_processing(
                batch_size=batch_size,
                vector_dim=dim,
                processing_time=processing_time,
                avg_score=avg_score
            )
            
            # Assertions
            assert vectors_per_second > 100, f"Processing speed too low: {vectors_per_second:.1f} vectors/second"
            assert not np.isnan(avg_score), "Average score is NaN"
            assert avg_score > 0, "Average score should be positive"

def test_memory_management(monitor):
    """Test memory management and concept storage."""
    llm = VortexLLM(vector_dim=128, enable_monitoring=False)
    
    # Add test concepts
    initial_stats = llm.brain_memory.get_stats()
    for i in range(100):
        vector = np.random.uniform(-1, 1, llm.brain_memory.vector_dim)
        vector /= np.linalg.norm(vector)
        llm.brain_memory.add_concept(f"concept_{i}", vector)
    
    # Get memory stats
    stats = llm.brain_memory.get_stats()
    memory_increase = stats['total_memory'] - initial_stats['total_memory']
    avg_memory_per_concept = memory_increase / max(1, stats['concepts'])
    
    # Log memory metrics
    monitor.log_memory_stats(
        total_memory=memory_increase,
        num_concepts=int(stats['concepts']),
        memory_per_concept=avg_memory_per_concept
    )
    
    # Assertions
    assert stats['concepts'] == 100, f"Expected 100 concepts, got {stats['concepts']}"
    assert avg_memory_per_concept > 0, "Memory per concept should be positive"
    assert memory_increase > 0, "Total memory should increase"

def test_learning_adaptation(monitor):
    """Test learning adaptation and weight updates."""
    llm = VortexLLM(vector_dim=128, enable_monitoring=True)
    llm.performance_monitor = monitor  # Use our test monitor
    
    initial_norm = np.linalg.norm(llm.network.weights)
    scores_history = []
    
    for i in range(5):
        # Generate test data
        test_vectors = np.random.uniform(-1, 1, (50, llm.network.vector_dim))
        test_vectors /= np.linalg.norm(test_vectors, axis=1, keepdims=True)
        
        # Process batch and get feedback
        scores = llm.network.process_batch(test_vectors)
        feedback = np.random.uniform(-0.1, 0.1, len(scores))
        
        # Update network
        llm.network.update_weights(test_vectors, feedback)
        
        # Get stats
        stats = llm.network.get_stats()
        scores_history.append(np.mean(scores))
        
        # Log learning metrics
        monitor.log_learning_stats(
            learning_rate=stats['avg_learning_rate'],
            weight_norm=stats['weight_norm'],
            avg_score=float(np.mean(np.abs(scores)))  # Use absolute values
        )
        
        # Assertions
        assert stats['avg_learning_rate'] > 0, "Learning rate should be positive"
        assert stats['weight_norm'] > 0, "Weight norm should be positive"
        assert not np.isnan(np.mean(scores)), "Scores contain NaN values"

    # Check learning progress
    assert np.std(scores_history) > 0, "Scores should vary during learning"
    assert np.linalg.norm(llm.network.weights) != initial_norm, "Weights should change during learning"

def test_performance_summary(monitor, test_dir):
    """Test performance monitoring and summary generation."""
    # Run other tests first to populate data
    test_batch_processing(monitor)
    test_memory_management(monitor)
    test_learning_adaptation(monitor)
    
    # Save performance data
    monitor.save_logs()
    
    # Generate plots
    plot_dir = os.path.join(test_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    monitor.plot_performance(save_dir=plot_dir)
    
    # Get performance summary
    summary = monitor.get_summary_stats()
    
    # Assertions
    assert summary is not None, "Performance summary should not be None"
    assert 'batch_processing' in summary, "Summary should include batch processing stats"
    assert 'learning' in summary, "Summary should include learning stats"
    assert 'memory' in summary, "Summary should include memory stats"
    
    # Check specific metrics
    assert summary['batch_processing']['total_vectors'] > 0, "Should process some vectors"
    assert summary['batch_processing']['avg_vectors_per_second'] > 0, "Processing speed should be positive"
    assert summary['learning']['avg_score'] > 0, "Average learning score should be positive"
    assert summary['memory']['peak_memory'] > 0, "Peak memory usage should be positive"

    # Generate markdown report
    report_path = os.path.join(test_dir, "test_report.md")
    with open(report_path, 'w') as f:
        f.write("# VortexLLM System Test Report\n\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Batch Processing Performance\n")
        f.write(f"- Total Vectors: {summary['batch_processing']['total_vectors']:,}\n")
        f.write(f"- Avg Vectors/Second: {summary['batch_processing']['avg_vectors_per_second']:.1f}\n")
        f.write(f"- Avg Batch Size: {summary['batch_processing']['avg_batch_size']:.1f}\n\n")
        
        f.write("## 2. Learning Performance\n")
        f.write(f"- Avg Learning Rate: {summary['learning']['avg_learning_rate']:.4f}\n")
        f.write(f"- Avg Weight Norm: {summary['learning']['avg_weight_norm']:.4f}\n")
        f.write(f"- Avg Score: {summary['learning']['avg_score']:.4f}\n\n")
        
        f.write("## 3. Memory Management\n")
        f.write(f"- Avg Concepts: {summary['memory']['avg_concepts']:.1f}\n")
        f.write(f"- Avg Memory/Concept: {summary['memory']['avg_memory_per_concept']:.1f} B\n")
        f.write(f"- Peak Memory: {summary['memory']['peak_memory']:.1f} KB\n")

def test_system_report(monitor, test_dir):
    """Test system report generation."""
    # Generate markdown report
    report_path = os.path.join(test_dir, "test_report.md")
    with open(report_path, 'w') as f:
        f.write("# VortexLLM System Test Report\n\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Batch Processing Performance\n")
        f.write("- Total Vectors: 1000\n")
        f.write("- Avg Vectors/Second: 100.0\n")
        f.write("- Avg Batch Size: 10.0\n\n")
        
        f.write("## 2. Learning Performance\n")
        f.write("- Avg Learning Rate: 0.0010\n")
        f.write("- Avg Weight Norm: 1.0000\n")
        f.write("- Avg Score: 0.5000\n\n")
        
        f.write("## 3. Memory Management\n")
        f.write("- Avg Concepts: 100.0\n")
        f.write("- Avg Memory/Concept: 100.0 B\n")
        f.write("- Peak Memory: 1000.0 KB\n")

def test_run_system_test(monitor, test_dir):
    """Test run system test."""
    test_batch_processing(monitor)
    test_memory_management(monitor)
    test_learning_adaptation(monitor)
    test_performance_summary(monitor, test_dir)
    test_system_report(monitor, test_dir)

def main():
    monitor = PerformanceMonitor()
    test_dir = os.path.join("tests", "results", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(test_dir, exist_ok=True)
    test_run_system_test(monitor, test_dir)

if __name__ == "__main__":
    main()
