import numpy as np
import time as time_module
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from .resonance_optimizer import ResonanceOptimizer
import sys
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Lock

class VortexAI:
    """Core VortexAI class for quantum-inspired wave processing."""
    
    def __init__(self, input_dim: int = 64, n_fields: int = 4):
        """Initialize VortexAI with V9 enhanced parameters."""
        self.input_dim = input_dim
        self.n_fields = n_fields
        
        # V9 Enhanced Parameters
        self.phase_coupling = 0.7
        self.field_decay = 0.1
        self.interference_strength = 0.3
        self.beta = 0.2  # Memory influence factor
        self.lambda_stability = 1.0  # Stability scaling factor
        self.ema_decay = 0.9  # Exponential moving average decay rate
        
        self._lock = Lock()
        
        # Initialize resonance fields as numpy arrays
        self.resonance_fields = np.zeros((n_fields, input_dim))
        for i in range(n_fields):
            self.resonance_fields[i] = np.random.randn(input_dim)
            # Normalize initial fields
            norm = np.sqrt(np.sum(self.resonance_fields[i] ** 2))
            if norm > 0:
                self.resonance_fields[i] /= norm
                
        # Initialize phase memory with complex values
        self.phase_memory = np.zeros(input_dim, dtype=np.complex128)
        
        # System parameters
        self.avg_energy_conservation = 1.0
        self.stability_window = []
        self.max_window_size = 10
        self.phase_history = []
        self.max_phase_history = 50
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.max_chunk_size = 256
        
    def calculate_learning_rate(self, v: float, t: float) -> float:
        """ Learning Rate Optimization using the finalized equation.
        
        Args:
            v: Concept propagation speed
            t: Reinforcement time
            
        Returns:
            float: Optimized learning rate
        """
        n = self.n_fields  # Active processing layers
        sigma_max = (np.pi ** 2 * (v * t)) / n
        return sigma_max / 2  # As per V9 equation: σ_max = (1/2)σ

    def update_phase_state(self, current_phase: float, neighbor_phases: np.ndarray, 
                          weights: np.ndarray) -> float:
        """ Phase Synchronization Update.
        
        Args:
            current_phase: Current phase state φ_t
            neighbor_phases: Array of neighboring phases φ_i
            weights: Weight distribution W_i
            
        Returns:
            float: Updated phase state
        """
        delta_phi = np.mean(np.sin(neighbor_phases - current_phase))
        alpha = self.phase_coupling
        
        # V9 Phase Update Equation
        new_phase = current_phase + alpha * delta_phi - self.beta * np.sum(
            weights * np.sin(current_phase - neighbor_phases)
        )
        
        # Phase wrapping in [-π, π]
        return np.arctan2(np.sin(new_phase), np.cos(new_phase))

    def compute_wave_attention(self, weights: np.ndarray) -> np.ndarray:
        """ Enhanced Wave Attention Mechanism.
        
        Args:
            weights: Initial attention weights
            
        Returns:
            np.ndarray: Normalized attention weights
        """
        # Global normalization with stability scaling
        exp_weights = np.exp(self.lambda_stability * weights)
        return exp_weights / np.sum(exp_weights)

    def process_batch(self, vectors: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Process batch with  optimizations for stability.
        
        Args:
            vectors: Input vectors
            time: Current time step
            
        Returns:
            np.ndarray: Processed vectors
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        # Calculate batch energy for stability
        batch_energy = np.sum(vectors ** 2) / len(vectors)
        
        # Apply wave attention
        attention_weights = self.compute_wave_attention(
            np.sum(vectors ** 2, axis=1)
        )
        
        # Apply interference with energy conservation
        processed = self.apply_wave_interference(vectors, time)
        
        # Update stability metrics
        self.update_stability_metrics(
            batch_energy / (np.sum(processed ** 2) / len(processed)),
            np.mean(attention_weights)
        )
        
        # Track phase history with EMA
        if len(self.phase_history) > 0:
            prev_phase = self.phase_history[-1]
            curr_phase = np.angle(np.sum(processed))
            ema_phase = self.ema_decay * prev_phase + (1 - self.ema_decay) * curr_phase
            self.phase_history.append(ema_phase)
        else:
            self.phase_history.append(np.angle(np.sum(processed)))
            
        # Maintain history size
        if len(self.phase_history) > self.max_phase_history:
            self.phase_history.pop(0)
        
        return processed

    def apply_wave_interference(self, vectors: np.ndarray, time: float) -> np.ndarray:
        """Apply wave interference patterns to input vectors."""
        # Ensure input is 2D
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        # Calculate phase factors
        phases = np.exp(2j * np.pi * time * np.arange(vectors.shape[1]) / vectors.shape[1])
        
        # Apply interference pattern
        interference = np.outer(
            np.sin(2 * np.pi * time * np.arange(len(vectors)) / len(vectors)),
            phases
        )
        
        # Combine with input vectors
        result = vectors * (1 + self.interference_strength * interference)
        
        # Normalize to preserve energy
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / (norms + 1e-8)

    def update_stability_metrics(self, energy_ratio: float, field_strength: float, 
                              coherence: float = None):
        """Update system stability metrics with enhanced tracking."""
        # Update stability window with combined metrics
        if coherence is None:
            coherence = field_strength
        
        self.stability_window.append((energy_ratio, coherence))
        if len(self.stability_window) > self.max_window_size:
            self.stability_window.pop(0)
            
        if self.stability_window:
            energy_ratios, coherences = zip(*self.stability_window)
            energy_stability = np.std(energy_ratios) / (np.mean(energy_ratios) + 1e-8)
            avg_coherence = np.mean(coherences)
            
            # Combine metrics with adaptive weighting
            stability_weight = np.exp(-0.1 * abs(1.0 - avg_coherence))
            self.avg_resonance_stability = (
                0.6 * avg_coherence + 
                0.4 / (1.0 + energy_stability)
            ) * stability_weight
            
        # Update energy conservation metric
        self.avg_energy_conservation = 1.0 / (1.0 + energy_stability)

    def process_batch_parallel(self, vectors: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Process a batch of vectors in parallel using wave equations."""
        if len(vectors) == 0:
            return np.array([])
            
        # Ensure input is 2D
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        # Calculate optimal chunk size
        chunk_size = min(len(vectors), self.max_chunk_size)
        chunks = np.array_split(vectors, max(1, len(vectors) // chunk_size))
        
        # Process chunks in parallel
        results = []
        for chunk in chunks:
            result = self.process_batch(chunk, time)
            # Take mean across wave dimensions to get scores
            scores = np.mean(np.abs(result), axis=1)
            results.append(scores)
            
        # Combine results
        combined = np.concatenate(results)
        
        # Normalize scores
        if len(combined) > 0:
            combined = (combined - np.min(combined)) / (np.max(combined) - np.min(combined) + 1e-8)
            
        return combined
        
    def _process_chunk_optimized(self, vectors: np.ndarray, time: float, field_influence: np.ndarray) -> np.ndarray:
        """Optimized chunk processing with energy conservation."""
        # Convert to complex and normalize while preserving energy
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        transformed = vectors.astype(np.complex128) / (norms + 1e-8)
        
        # Apply wave-based attention efficiently
        attention = self.compute_wave_attention(transformed)
        transformed = transformed * attention.reshape(-1, 1)
        
        # Enhanced phase synchronization
        ref_phase = np.angle(np.sum(transformed, axis=0))
        phase_factors = np.exp(-1j * ref_phase)
        transformed = transformed * phase_factors
        
        # Apply temporal coherence vectorized
        if len(transformed) > 1:
            phases = np.angle(transformed)
            phase_diffs = phases[1:] - phases[:-1]
            coherence = np.exp(-1j * phase_diffs * self.phase_coupling)
            transformed[1:] = transformed[1:] * coherence
            
        # Apply wave transformation with resonance
        k = 2 * np.pi
        wave_phase = k * time
        resonance_factor = 1 + self.interference_strength * np.abs(field_influence)
        
        # Combine transformations efficiently
        transformed = transformed * np.exp(1j * wave_phase) * resonance_factor
        transformed = transformed + 0.1 * np.roll(transformed, 1, axis=1)
        
        # Convert back to real while preserving energy
        result = np.real(transformed) * norms
        result_norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result * (norms / (result_norms + 1e-8))
        
        return result.astype(np.float64)
        
    def track_stability_metrics(self, initial_energy: float, current_energy: float, 
                              resonance_factor: np.ndarray):
        """Track stability metrics for wave processing."""
        energy_conservation = abs(current_energy - initial_energy) / (initial_energy + 1e-10)
        resonance_stability = np.mean(resonance_factor)
        
        # Update running averages
        alpha = 0.1  # Exponential moving average factor
        self.avg_energy_conservation = (1 - alpha) * self.avg_energy_conservation + alpha * energy_conservation
        self.avg_resonance_stability = (1 - alpha) * self.avg_resonance_stability + alpha * resonance_stability
        
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        stats = {
            'total_processed': 0,
            'avg_energy': 0.0,
            'resonance_avg_stability': float(np.mean([history[-1][1] 
                                                    for history in self.resonance_optimizer.history.values() 
                                                    if history]))
                if hasattr(self, 'resonance_optimizer') and self.resonance_optimizer.history else 0.0,
            'memory_usage': 0,
            'avg_energy_conservation': self.avg_energy_conservation,
            'avg_resonance_stability': self.avg_resonance_stability
        }
        return stats
        
    def _calculate_interference(self, vectors: np.ndarray, time: float) -> np.ndarray:
        """Compute interference patterns between input vectors.
        
        Args:
            vectors: Input vectors of shape (batch_size, vector_dim)
            time: Time value
            
        Returns:
            np.ndarray: Interference pattern of shape (batch_size,)
        """
        batch_size = len(vectors)
        if batch_size <= 1:
            return np.zeros(batch_size)
            
        # Normalize vectors for stable similarity computation
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            
        # Compute pairwise vector similarities with stable normalization
        similarities = np.dot(vectors_norm, vectors_norm.T)
        similarities = np.clip(similarities, -1, 1)  # Ensure valid range
        np.fill_diagonal(similarities, 0)  # Remove self-interference
        
        # Calculate phase differences with stability
        phase_diff = np.zeros((batch_size, batch_size))
        phase_coupling = np.cos(phase_diff) * self.phase_coupling
        
        # Combine similarity and phase effects with stability
        interference = np.sum(similarities * phase_coupling, axis=1)
        interference = interference / max(batch_size - 1, 1)  # Safe normalization
        
        # Ensure no NaN values
        interference = np.nan_to_num(interference, nan=0.0)
        return interference

    def __del__(self):
        """Clean up thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def synchronize_phases(self, vectors: np.ndarray) -> np.ndarray:
        """Synchronize phases of input vectors using wave-based attention."""
        if len(vectors) < 2:
            return vectors

        # Convert to complex representation
        complex_vectors = vectors.astype(np.complex128)
        norms = np.sqrt(np.sum(np.abs(complex_vectors)**2, axis=1))
        normalized = complex_vectors / (norms.reshape(-1, 1) + 1e-8)
        
        # Calculate reference phase using weighted average
        weights = np.ones(len(vectors)) / len(vectors)
        ref_vector = weights @ normalized
        ref_phase = np.angle(ref_vector)
        
        # Calculate phase differences for each vector component
        phases = np.angle(normalized)
        phase_diffs = phases - ref_phase.reshape(1, -1)  # Ensure correct broadcasting
        
        # Wrap phase differences to [-pi, pi]
        phase_diffs = np.mod(phase_diffs + np.pi, 2 * np.pi) - np.pi
        
        # Apply adaptive coupling strength based on phase difference
        coupling_strength = self.phase_coupling * (1.0 - np.abs(phase_diffs) / np.pi)
        
        # Apply smooth phase adjustment with enhanced stability
        target_phases = ref_phase.reshape(1, -1) + coupling_strength * phase_diffs
        
        # Apply phase transformation with energy conservation
        phase_factors = np.exp(1j * target_phases)
        result = normalized * phase_factors
        
        # Update phase memory for enhanced stability
        self.phase_memory = 0.9 * self.phase_memory + 0.1 * np.mean(result, axis=0)
        
        # Apply phase memory influence
        memory_phase = np.angle(self.phase_memory)
        memory_coupling = 0.2 * np.exp(-0.5 * np.abs(phase_diffs))
        final_phases = target_phases + memory_coupling * (memory_phase - target_phases)
        
        # Final transformation
        final_factors = np.exp(1j * final_phases)
        result = normalized * final_factors
        
        # Restore original magnitudes
        return result * norms.reshape(-1, 1)

class ResonanceOptimizer:
    """Optimize energy resonance for better stability."""
    
    def __init__(self):
        self.history = {}
        self.stability_window = 5
        self.min_stability = 0.1
        self.max_stability = 0.9
        self.resonance_factor = 1.2
        self.damping_factor = 0.8
        
    def update_stability(self, energy_before: float, energy_after: float, time: float):
        """Update resonance stability based on energy changes."""
        # Calculate stability based on energy consistency
        stability = 0.5
        
        # Apply resonance based on stability
        if stability > 0.5:
            # Strong resonance for stable patterns
            optimized = energy_before * (1.0 + (stability - 0.5) * self.resonance_factor)
        else:
            # Damping for unstable patterns
            optimized = energy_before * (1.0 - (0.5 - stability) * self.damping_factor)
            
        # Update history
        if time not in self.history:
            self.history[time] = []
        self.history[time].append((optimized, stability))
        
        # Maintain a window of recent stability values
        if len(self.history[time]) > self.stability_window:
            self.history[time].pop(0)
            
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        stats = {
            'total_processed': 0,
            'avg_energy': 0.0,
            'resonance_avg_stability': float(np.mean([history[-1][1] for history in self.history.values() if history]))
        }
        return stats
