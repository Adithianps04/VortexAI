import numpy as np
import time
from typing import Dict, List, Optional
from collections import defaultdict
import json
import os
from datetime import datetime

class PerformanceMonitor:
    """Real-time performance monitoring for VortexLLM."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Performance metrics
        self.batch_stats = {
            'total_vectors': 0,
            'total_time': 0.0,
            'batch_sizes': [],
            'vector_dims': [],
            'processing_times': [],
            'avg_scores': []
        }
        
        self.learning_stats = {
            'learning_rates': [],
            'weight_norms': [],
            'avg_scores': []
        }
        
        self.memory_stats = {
            'total_memory': [],
            'num_concepts': [],
            'memory_per_concept': []
        }
        
        self.learning_rate = 0
        self.weight_norm = 0
        self.avg_weight_change = 0
        self.total_updates = 0
        
        print("\nPerformance Monitor Initialized")
        
        # Initialize log file
        self.log_file = os.path.join(
            log_dir, 
            f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    def log_batch_processing(self, 
                           batch_size: int,
                           vector_dim: int,
                           processing_time: float,
                           avg_score: float):
        """Log batch processing statistics."""
        self.batch_stats['total_vectors'] += batch_size
        self.batch_stats['total_time'] += processing_time
        self.batch_stats['batch_sizes'].append(batch_size)
        self.batch_stats['vector_dims'].append(vector_dim)
        self.batch_stats['processing_times'].append(processing_time)
        self.batch_stats['avg_scores'].append(avg_score)
        
        print(f"\nLogged Batch Stats:")
        print(f"Batch Size: {batch_size}")
        print(f"Vector Dim: {vector_dim}")
        print(f"Processing Time: {processing_time:.3f}s")
        print(f"Average Score: {avg_score:.6f}")
        print(f"Total Vectors: {self.batch_stats['total_vectors']}")
        print(f"Total Time: {self.batch_stats['total_time']:.3f}s")
    
    def log_learning_stats(self, 
                          learning_rate: float,
                          weight_norm: float,
                          avg_score: float):
        """Log learning statistics."""
        self.learning_stats['learning_rates'].append(learning_rate)
        self.learning_stats['weight_norms'].append(weight_norm)
        self.learning_stats['avg_scores'].append(avg_score)
        
        self.learning_rate = learning_rate
        self.weight_norm = weight_norm
        self.total_updates += 1
        
        print(f"\nLogged Learning Stats:")
        print(f"Learning Rate: {learning_rate:.6f}")
        print(f"Weight Norm: {weight_norm:.6f}")
        print(f"Average Score: {avg_score:.6f}")
    
    def log_memory_stats(self, 
                        total_memory: float,
                        num_concepts: int,
                        memory_per_concept: float):
        """Log memory usage statistics."""
        self.memory_stats['total_memory'].append(total_memory)
        self.memory_stats['num_concepts'].append(num_concepts)
        self.memory_stats['memory_per_concept'].append(memory_per_concept)
        
        print(f"\nLogged Memory Stats:")
        print(f"Total Memory: {total_memory:.2f} KB")
        print(f"Number of Concepts: {num_concepts}")
        print(f"Memory per Concept: {memory_per_concept:.2f} B")
    
    def get_summary_stats(self):
        """Generate performance summary statistics."""
        if not self.batch_stats['total_vectors']:
            return {
                'batch_processing': {
                    'total_vectors': 0,
                    'total_time': 0,
                    'avg_processing_time': 0,
                    'avg_score': 0,
                    'avg_vectors_per_second': 0,
                    'avg_batch_size': 0
                },
                'learning': {
                    'learning_rate': self.learning_rate,
                    'weight_norm': self.weight_norm,
                    'avg_weight_change': self.avg_weight_change,
                    'total_updates': self.total_updates,
                    'avg_learning_rate': 0,
                    'avg_weight_norm': 0,
                    'avg_score': 0
                },
                'memory': {
                    'total_memory': self.memory_stats['total_memory'][-1] if self.memory_stats['total_memory'] else 0,
                    'num_concepts': self.memory_stats['num_concepts'][-1] if self.memory_stats['num_concepts'] else 0,
                    'memory_per_concept': self.memory_stats['memory_per_concept'][-1] if self.memory_stats['memory_per_concept'] else 0,
                    'avg_concepts': 0,
                    'avg_memory_per_concept': 0,
                    'peak_memory': 0
                }
            }
            
        avg_time = self.batch_stats['total_time'] / self.batch_stats['total_vectors']
        avg_score = np.mean(self.batch_stats.get('avg_scores', [0]))
        avg_batch_size = np.mean(self.batch_stats['batch_sizes'])
        avg_vectors_per_second = self.batch_stats['total_vectors'] / max(self.batch_stats['total_time'], 0.001)
        
        # Calculate learning averages
        avg_learning_rate = np.mean(self.learning_stats['learning_rates']) if self.learning_stats['learning_rates'] else 0
        avg_weight_norm = np.mean(self.learning_stats['weight_norms']) if self.learning_stats['weight_norms'] else 0
        avg_learning_score = np.mean(self.learning_stats['avg_scores']) if self.learning_stats['avg_scores'] else 0
        
        # Calculate memory averages
        avg_concepts = np.mean(self.memory_stats['num_concepts']) if self.memory_stats['num_concepts'] else 0
        avg_memory_per_concept = np.mean(self.memory_stats['memory_per_concept']) if self.memory_stats['memory_per_concept'] else 0
        peak_memory = max(self.memory_stats['total_memory']) if self.memory_stats['total_memory'] else 0
        
        return {
            'batch_processing': {
                'total_vectors': self.batch_stats['total_vectors'],
                'total_time': self.batch_stats['total_time'],
                'avg_processing_time': avg_time,
                'avg_score': avg_score,
                'avg_vectors_per_second': avg_vectors_per_second,
                'avg_batch_size': avg_batch_size
            },
            'learning': {
                'learning_rate': self.learning_rate,
                'weight_norm': self.weight_norm,
                'avg_weight_change': self.avg_weight_change,
                'total_updates': self.total_updates,
                'avg_learning_rate': avg_learning_rate,
                'avg_weight_norm': avg_weight_norm,
                'avg_score': avg_learning_score
            },
            'memory': {
                'total_memory': self.memory_stats['total_memory'][-1] if self.memory_stats['total_memory'] else 0,
                'num_concepts': self.memory_stats['num_concepts'][-1] if self.memory_stats['num_concepts'] else 0,
                'memory_per_concept': self.memory_stats['memory_per_concept'][-1] if self.memory_stats['memory_per_concept'] else 0,
                'avg_concepts': avg_concepts,
                'avg_memory_per_concept': avg_memory_per_concept,
                'peak_memory': peak_memory
            }
        }
    
    def plot_performance(self, save_dir: Optional[str] = None):
        """Generate performance plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create save directory if needed
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            # Batch processing performance
            plt.figure(figsize=(10, 6))
            plt.plot(self.batch_stats['batch_sizes'], label='Batch Size')
            plt.plot([v/t for v, t in zip(
                self.batch_stats['batch_sizes'], 
                self.batch_stats['processing_times'])
            ], label='Vectors/Second')
            plt.title('Batch Processing Performance')
            plt.xlabel('Batch Number')
            plt.ylabel('Count')
            plt.legend()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'batch_performance.png'))
            plt.close()
            
            # Learning progress
            plt.figure(figsize=(10, 6))
            plt.plot(self.learning_stats['avg_scores'], label='Average Score')
            plt.plot(self.learning_stats['weight_norms'], label='Weight Norm')
            plt.title('Learning Progress')
            plt.xlabel('Update Number')
            plt.ylabel('Value')
            plt.legend()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'learning_progress.png'))
            plt.close()
            
            # Memory usage
            plt.figure(figsize=(10, 6))
            plt.plot(self.memory_stats['total_memory'], label='Total Memory (KB)')
            plt.plot(self.memory_stats['num_concepts'], label='Active Concepts')
            plt.title('Memory Usage')
            plt.xlabel('Update Number')
            plt.ylabel('Count')
            plt.legend()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'memory_usage.png'))
            plt.close()
            
        except ImportError:
            print("matplotlib not available for plotting")

    def save_logs(self):
        """Save performance logs to file."""
        import json
        
        logs = {
            'batch_stats': self.batch_stats,
            'learning_stats': self.learning_stats,
            'memory_stats': self.memory_stats
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"\nPerformance logs saved to: {self.log_file}")
