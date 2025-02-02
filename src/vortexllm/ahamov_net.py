import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import concurrent.futures
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time

class AhamovNet:
    """Neural network with adaptive learning and parallel processing."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, learning_rate: float = 0.01):
        """Initialize AhamovNet."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights with Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.weights = np.random.randn(input_dim, hidden_dim) * scale
        
        # Learning parameters
        self.base_lr = learning_rate
        self.min_lr = 1e-5
        self.max_lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # Adaptive learning parameters
        self.lr = self.base_lr
        self.beta1 = 0.9  # First moment decay rate
        self.beta2 = 0.999  # Second moment decay rate
        self.epsilon = 1e-8  # Small constant for numerical stability
        
        # Initialize momentum and adaptive learning buffers
        self.velocity = np.zeros_like(self.weights)
        self.m = np.zeros_like(self.weights)  # First moment
        self.v = np.zeros_like(self.weights)  # Second moment
        self.t = 0  # Time step
        
        # Loss history for learning rate adaptation
        self.loss_history = []
        self.patience = 5
        self.improvement_threshold = 0.001  # More sensitive to improvements
        
        # Initialize thread pool
        self.max_workers = min(32, (os.cpu_count() or 1) * 2)
        self.worker_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance tracking
        self.processing_times = []
        self.batch_sizes = []
        self.current_batch_size = 32
        self.min_batch_size = 8
        self.max_batch_size = 128
        self.optimal_chunk_size = None
        self.min_chunk_size = 16
        self.max_chunk_size = 256
        
        # Initialize other components
        self._init_network_components()
    
    def _init_network_components(self):
        """Initialize network components with memory optimization."""
        # Memory management
        self.memory_per_weight = 4  # bytes
        self.total_weights = self.input_dim * self.hidden_dim
        self.memory_usage = self.total_weights * self.memory_per_weight
        
        # Track updates
        self.path_strengths = {}
    
    def _process_chunk(self, vectors: np.ndarray) -> np.ndarray:
        """Process a chunk of vectors through the network."""
        # Input validation
        if vectors.shape[1] != self.input_dim:
            raise ValueError(f"Input vectors must have dimension {self.input_dim}")
        
        # Track processing
        print(f"\nProcessing Chunk:")
        print(f"Input Shape: {vectors.shape}")
        print(f"Input Stats:")
        print(f"  Mean: {np.mean(vectors):.6f}")
        print(f"  Std: {np.std(vectors):.6f}")
        print(f"  Norm: {np.linalg.norm(vectors):.6f}")
        
        # Apply weight transformation
        transformed = np.dot(vectors, self.weights)
        
        # Apply activation (tanh for stable gradients)
        activated = np.tanh(transformed)
        
        print(f"Output Stats:")
        print(f"  Mean: {np.mean(activated):.6f}")
        print(f"  Std: {np.std(activated):.6f}")
        print(f"  Norm: {np.linalg.norm(activated):.6f}\n")
        
        return activated
    
    def process_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Process a batch of vectors with adaptive chunking."""
        if len(vectors) == 0:
            return np.array([])
        
        # For single vectors, process directly
        if len(vectors) == 1:
            return self._process_chunk(vectors)
        
        # Adjust batch size based on processing times
        self._adjust_batch_size()
        
        # Split into optimal chunks
        chunk_size = self._calculate_optimal_chunk_size(len(vectors))
        chunks = [vectors[i:i + chunk_size] for i in range(0, len(vectors), chunk_size)]
        
        # Process chunks in parallel
        start_time = time.time()
        futures = []
        results = []
        
        try:
            # Submit chunks to worker pool
            for chunk in chunks:
                future = self.worker_pool.submit(self._process_chunk, chunk)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                chunk_result = future.result()
                results.extend(chunk_result)
                
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            # Fallback to sequential processing if parallel fails
            results = self._process_chunk(vectors)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append((len(vectors), processing_time))
        
        return np.array(results)

    def _calculate_optimal_chunk_size(self, total_size: int) -> int:
        """Calculate optimal chunk size based on vector dimension and available workers."""
        if self.optimal_chunk_size is not None:
            return min(self.optimal_chunk_size, total_size)
        
        # Base chunk size on vector dimension and available memory
        memory_per_vector = self.input_dim * 4  # 4 bytes per float32
        target_memory_per_chunk = 256 * 1024  # Target 256KB per chunk
        
        base_chunk_size = max(32, target_memory_per_chunk // memory_per_vector)
        num_chunks = max(1, self.max_workers * 2)  # Ensure enough chunks for all workers
        
        self.optimal_chunk_size = max(self.min_chunk_size, min(base_chunk_size, total_size // num_chunks))
        return self.optimal_chunk_size

    def _adjust_batch_size(self):
        """Adjust batch size based on processing performance."""
        if len(self.processing_times) < 5:
            return
        
        # Calculate recent processing efficiency
        recent_times = self.processing_times[-5:]
        vectors_per_second = [size/time for size, time in recent_times]
        avg_throughput = np.mean(vectors_per_second)
        
        # Adjust batch size based on throughput trend
        if avg_throughput > 1000:  # Good throughput
            self.current_batch_size = min(self.current_batch_size * 1.2, 512)
        else:  # Poor throughput
            self.current_batch_size = max(self.current_batch_size * 0.8, 32)

    def train_sequence(self, sequence: np.ndarray) -> float:
        """Train network on a sequence of vectors."""
        if len(sequence) < 2:
            return 0.0  # Cannot train on single vector
            
        # Convert to real domain if complex
        if np.iscomplexobj(sequence):
            sequence = np.real(sequence)
            
        total_loss = 0.0
        sequence_length = len(sequence)
        
        # Normalize sequence
        norms = np.linalg.norm(sequence, axis=1, keepdims=True)
        sequence = sequence / (norms + 1e-8)
        
        # Initial loss calculation
        initial_losses = []
        for i in range(sequence_length - 1):
            input_vec = sequence[i:i+1]
            target_vec = sequence[i+1:i+2]
            output = self.forward(input_vec)
            error = target_vec - output
            loss = np.mean(error ** 2)
            if not np.isnan(loss):
                initial_losses.append(loss)
                
        initial_loss = np.mean(initial_losses) if initial_losses else 1.0
        prev_loss = initial_loss
        
        # Training iterations
        for epoch in range(5):
            epoch_loss = 0.0
            valid_updates = 0
            
            for i in range(sequence_length - 1):
                input_vec = sequence[i:i+1]
                target_vec = sequence[i+1:i+2]
                
                # Forward pass
                output = self.forward(input_vec)
                
                # Calculate loss
                error = target_vec - output
                loss = np.mean(error ** 2)
                
                if np.isnan(loss):
                    continue
                    
                epoch_loss += loss
                valid_updates += 1
                
                # Backward pass with real-valued gradients
                gradients = self.backward(error)
                gradients = np.real(gradients)
                
                # Update first and second moments
                self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
                self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
                
                # Bias correction
                t = max(1, self.t)
                m_hat = self.m / (1 - self.beta1 ** t + 1e-8)
                v_hat = self.v / (1 - self.beta2 ** t + 1e-8)
                
                # Adaptive learning rate based on loss trend
                if valid_updates > 0:
                    loss_ratio = loss / prev_loss
                    if loss_ratio > 1.1:  # Loss increasing
                        self.lr *= 0.95  # Decrease learning rate
                    elif loss_ratio < 0.9:  # Loss decreasing
                        self.lr *= 1.05  # Increase learning rate
                    self.lr = np.clip(self.lr, 0.001, 0.1)  # Keep within bounds
                    prev_loss = loss
                
                # Update weights using Adam with gradient clipping
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                update_norm = np.linalg.norm(update)
                if update_norm > 1.0:
                    update *= 1.0 / update_norm
                    
                self.weights = np.real(self.weights - update)
                self.t += 1
                
            # Calculate average loss for this epoch
            if valid_updates > 0:
                epoch_loss /= valid_updates
                total_loss = epoch_loss
                
        return total_loss
        
    def predict_next(self, sequence: np.ndarray) -> np.ndarray:
        """Predict the next vector in a sequence."""
        if len(sequence) < 1:
            return np.zeros((1, self.input_dim))
            
        # Use last vector for prediction
        last_vec = sequence[-1:]  # Keep as 2D array
        
        # Normalize input
        last_vec = last_vec / (np.linalg.norm(last_vec, axis=1, keepdims=True) + 1e-8)
        
        # Forward pass with multiple steps
        predictions = []
        current_input = last_vec
        
        for _ in range(3):  # Generate multiple candidates
            pred = self.forward(current_input)
            predictions.append(pred)
            current_input = pred
            
        # Average predictions for stability
        prediction = np.mean(predictions, axis=0)
        
        # Normalize final prediction
        prediction = prediction / (np.linalg.norm(prediction) + 1e-8)
        
        return prediction
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        # Ensure inputs are 2D
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            
        # Project to hidden dimension with activation
        hidden = np.tanh(np.dot(inputs, self.weights))  # Shape: (batch_size, hidden_dim)
        
        # Project back to input dimension
        output = np.dot(hidden, self.weights.T)  # Shape: (batch_size, input_dim)
        
        # Normalize output
        output = output / (np.linalg.norm(output, axis=1, keepdims=True) + 1e-8)
        
        return output
        
    def backward(self, error: np.ndarray) -> np.ndarray:
        """Backward pass through the network."""
        # Ensure error is 2D
        if error.ndim == 1:
            error = error.reshape(1, -1)
            
        # Compute gradients with numerical stability
        # First, project error to hidden space
        d_hidden = np.dot(error, self.weights)  # Shape: (batch_size, hidden_dim)
        d_hidden = d_hidden * (1 - np.tanh(d_hidden) ** 2)  # tanh derivative
        
        # Then, compute weight gradients
        gradients = np.dot(error.T, d_hidden)  # Shape: (input_dim, hidden_dim)
        
        # Add weight decay
        gradients += self.weight_decay * self.weights
        
        # Clip gradients for stability
        max_grad_norm = 1.0
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > max_grad_norm:
            gradients *= max_grad_norm / grad_norm
            
        return gradients
        
    def update_weights(self, inputs: np.ndarray, targets: np.ndarray):
        """Update network weights using input-target pairs."""
        self.t += 1
        
        # Calculate gradients
        output = self.forward(inputs)
        error = targets - output
        gradients = self.backward(error)
        
        # Add weight decay
        gradients += self.weight_decay * self.weights
        
        # Update first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update weights using Adam
        self.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Calculate current loss for learning rate adjustment
        current_loss = np.mean(error ** 2)
        self.adjust_learning_rate(current_loss)
        
        return current_loss
        
    def adjust_learning_rate(self, current_loss: float):
        """Adjust learning rate based on loss."""
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
            
        if not np.isnan(current_loss):
            self.loss_history.append(current_loss)
            
        # Keep only recent history
        window_size = 10
        if len(self.loss_history) > window_size:
            self.loss_history = self.loss_history[-window_size:]
            
        # Need enough history to make adjustment
        if len(self.loss_history) < 3:
            return
            
        # Calculate loss trend
        recent_avg = np.mean(self.loss_history[-3:])
        older_avg = np.mean(self.loss_history[:-3])
        
        if older_avg == 0:
            return
            
        improvement = (older_avg - recent_avg) / older_avg
        
        # Adjust learning rate based on improvement
        if improvement > 0.1:  # Good improvement
            self.lr = min(self.lr * 1.1, 0.1)  # Increase but cap at 0.1
        elif improvement < -0.1:  # Getting worse
            self.lr = max(self.lr * 0.9, 1e-6)  # Decrease but keep above 1e-6
            
    def get_stats(self) -> Dict[str, float]:
        """Get network statistics."""
        if not self.learning_history:
            return {
                'total_updates': self.total_updates,
                'learning_rate': float(self.lr),
                'memory_usage': self.memory_usage,
                'weight_norm': float(np.linalg.norm(self.weights))
            }
            
        recent_history = self.learning_history[-100:]
        return {
            'total_updates': self.total_updates,
            'learning_rate': float(self.lr),
            'avg_batch_score': float(np.mean([h['batch_score'] for h in recent_history])),
            'memory_usage': self.memory_usage,
            'weight_norm': float(np.linalg.norm(self.weights)),
            'batch_size': self.current_batch_size
        }
    
    def get_learning_stats(self) -> dict:
        """Get current learning statistics."""
        return {
            'learning_rate': self.lr,
            'loss_history': self.loss_history[-10:] if self.loss_history else [],
            'weight_norm': np.linalg.norm(self.weights),
            'gradient_norm': np.linalg.norm(self.m),
            'time_step': self.t
        }
    
    def __del__(self):
        self.worker_pool.shutdown(wait=True)

# Initialize AhamovNet
net = AhamovNet(input_dim=64, hidden_dim=32, learning_rate=0.01)

# Print initialization message
print(f"\nAhamovNet Initialization:")
print(f"Input Dimension: {net.input_dim}")
print(f"Hidden Dimension: {net.hidden_dim}")
print(f"Total Weights: {net.total_weights:,}")
print(f"Memory Usage: {net.memory_usage / 1024:.2f} KB")
print(f"Initial Weight Stats:")
print(f"  Mean: {np.mean(net.weights):.6f}")
print(f"  Std: {np.std(net.weights):.6f}")
print(f"  Norm: {np.linalg.norm(net.weights):.6f}\n")
