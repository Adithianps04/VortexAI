from typing import List, Optional, Tuple, Dict
import numpy as np
import os
import pickle
from collections import defaultdict
import time
from .brain_memory import BrainMemory
from .custom_tokenizer import CustomTokenizer
from .vortex_ai import VortexAI
from .ahamov_net import AhamovNet
from .performance_monitor import PerformanceMonitor

class VortexLLM:
    def __init__(self,
                 vector_dim: int = 512,
                 learning_rate: float = 0.01,
                 nearby_size: int = 3,
                 checkpoint_interval: int = 1000,
                 enable_monitoring: bool = True):
        """Initialize VortexLLM.
        
        Args:
            vector_dim: Dimension of concept vectors
            learning_rate: Base learning rate for network
            nearby_size: Number of nearby tokens to consider
            checkpoint_interval: Steps between checkpoints
            enable_monitoring: Whether to enable performance monitoring
        """
        # Initialize components
        self.tokenizer = CustomTokenizer(vector_dim=vector_dim, nearby_size=nearby_size)
        self.brain_memory = BrainMemory()  # Using default parameters
        self.network = AhamovNet(input_dim=vector_dim, learning_rate=learning_rate)
        self.vortex = VortexAI(input_dim=vector_dim)
        
        # Checkpoint settings
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_counter = 0
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Performance monitoring
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = PerformanceMonitor()
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save current state to checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{int(time.time())}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save component states
        checkpoint_data = {
            'brain': {
                'concepts': self.brain_memory.concepts,
                'frequencies': self.brain_memory.concept_frequencies,
                'energies': self.brain_memory.concept_energies,
                'last_access': self.brain_memory.last_access,
                'creation_time': self.brain_memory.creation_time,
                'compressed': list(self.brain_memory.compressed_concepts)
            },
            'network': {
                'weights': self.network.connection_weights,
                'strengths': self.network.path_strengths,
                'learning_rates': self.network.learning_rates,
                'usage': dict(self.network.path_usage)
            },
            'tokenizer': {
                'token_vectors': self.tokenizer.token_vectors,
                'nearby_size': self.tokenizer.nearby_size,
                'vector_dim': self.tokenizer.vector_dim
            },
            'vortex': {
                'time': self.vortex.current_time,
                'state': self.vortex.get_state()
            },
            'metadata': {
                'vector_dim': self.tokenizer.vector_dim,
                'learning_rate': self.learning_rate,
                'timestamp': time.time()
            }
        }
        
        # Save checkpoint
        with open(checkpoint_path + '.pkl', 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        if self.enable_monitoring:
            self.monitor.save_logs()
    
    def load_checkpoint(self, checkpoint_name):
        """Load state from checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name + '.pkl')
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint {checkpoint_name} not found")
        
        # Load checkpoint data
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Verify compatibility
        if checkpoint_data['metadata']['vector_dim'] != self.tokenizer.vector_dim:
            raise ValueError("Vector dimension mismatch")
        
        # Restore brain state
        brain_data = checkpoint_data['brain']
        self.brain_memory.concepts = brain_data['concepts']
        self.brain_memory.concept_frequencies = brain_data['frequencies']
        self.brain_memory.concept_energies = brain_data['energies']
        self.brain_memory.last_access = brain_data['last_access']
        self.brain_memory.creation_time = brain_data['creation_time']
        self.brain_memory.compressed_concepts = set(brain_data['compressed'])
        
        # Restore network state
        network_data = checkpoint_data['network']
        self.network.connection_weights = network_data['weights']
        self.network.path_strengths = network_data['strengths']
        self.network.learning_rates = network_data['learning_rates']
        self.network.path_usage = defaultdict(int, network_data['usage'])
        
        # Restore tokenizer state
        tokenizer_data = checkpoint_data['tokenizer']
        self.tokenizer.token_vectors = tokenizer_data['token_vectors']
        self.tokenizer.nearby_size = tokenizer_data['nearby_size']
        self.tokenizer.vector_dim = tokenizer_data['vector_dim']
        
        # Restore vortex state
        vortex_data = checkpoint_data['vortex']
        self.vortex.current_time = vortex_data['time']
        self.vortex.set_state(vortex_data['state'])
    
    def _maybe_checkpoint(self):
        """Create checkpoint if interval is reached."""
        self.checkpoint_counter += 1
        if self.checkpoint_counter >= self.checkpoint_interval:
            self.save_checkpoint()
            self.checkpoint_counter = 0
    
    def train(self, text: str) -> None:
        """Train the system on input text.
        
        Args:
            text (str): Input text for training
        """
        # Train tokenizer
        self.tokenizer.train_tokens(text)
        
        # Add concepts to brain memory
        tokens = text.lower().split()
        for token in tokens:
            self.brain_memory.add_concept(token)
        
        # Auto-checkpoint
        self._maybe_checkpoint()
    
    def process(self, text: str) -> Tuple[List[int], List[float]]:
        """Process input text through the VortexLLM system.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple[List[int], List[float]]: Token IDs and their scores
        """
        # Tokenize input
        tokens = self.tokenizer.tokenize(text)
        token_vectors = self.tokenizer.get_token_vectors(tokens)
        
        # Process through network
        network_scores = self.network.process_batch(token_vectors)
        
        # Process through vortex
        vortex_scores = self.vortex.process_batch_parallel(token_vectors)
        
        # Reshape network scores to match vortex scores
        if network_scores.shape != vortex_scores.shape:
            # Take mean across hidden dimension for network scores
            network_scores = np.mean(network_scores, axis=1)
            # Ensure both have same shape
            network_scores = network_scores.reshape(-1)
            vortex_scores = vortex_scores.reshape(-1)
        
        # Combine scores with proper shapes
        combined_scores = network_scores * 0.7 + vortex_scores * 0.3
        
        # Update brain memory
        self.brain_memory.add_concept(
            key=text,
            vector=np.mean(token_vectors, axis=0),
            energy=np.mean(combined_scores)
        )
        
        # Return token IDs and scores
        return tokens, combined_scores.tolist()
    
    def feedback(self, scores: np.ndarray) -> None:
        """Provide feedback for adaptation.
        
        Args:
            scores (np.ndarray): Feedback scores for previous activation
        """
        self.network.adapt(scores)
        
        if self.enable_monitoring:
            stats = self.network.get_stats()
            self.monitor.log_learning_stats(
                learning_rate=stats['avg_learning_rate'],
                weight_norm=stats['weight_norm'],
                avg_score=float(np.mean(scores))
            )
        
        # Auto-checkpoint
        self._maybe_checkpoint()
    
    def get_memory_stats(self):
        """Get memory usage statistics for all components."""
        brain_stats = self.brain_memory.get_memory_stats()
        tokenizer_stats = self.tokenizer.get_stats()
        network_stats = self.network.get_stats()
        
        stats = {
            'brain_memory': {
                'concepts': brain_stats[0],
                'memory_mb': brain_stats[1]
            },
            'tokenizer': {
                'tokens': tokenizer_stats[0],
                'memory_mb': tokenizer_stats[1]
            },
            'network': network_stats,
            'total_memory_mb': (brain_stats[1] + tokenizer_stats[1] + 
                              network_stats['memory_mb'])
        }
        
        if self.enable_monitoring:
            self.monitor.log_memory_stats(
                total_memory=stats['total_memory_mb'],
                num_concepts=int(stats['brain_memory']['concepts']),
                memory_per_concept=stats['total_memory_mb'] / max(1, stats['brain_memory']['concepts'])
            )
        
        return stats
    
    def get_performance_summary(self) -> Optional[Dict]:
        """Get performance monitoring summary.
        
        Returns:
            Optional[Dict]: Performance summary if monitoring enabled
        """
        if self.enable_monitoring:
            return self.monitor.get_summary_stats()
        return None
    
    def plot_performance(self, save_dir: Optional[str] = None):
        """Generate performance plots.
        
        Args:
            save_dir: Optional directory to save plots
        """
        if self.enable_monitoring:
            self.monitor.plot_performance(save_dir)
