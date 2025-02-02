import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ResonanceState:
    """Tracks the resonance state of a concept."""
    energy: float
    frequency: float
    phase: float
    stability: float
    last_update: float

class ResonanceOptimizer:
    """Optimizes neural activations using resonance patterns."""
    
    def __init__(self, 
                 min_stability: float = 0.3,
                 max_frequency: float = 10.0,
                 learning_rate: float = 0.01):
        self.min_stability = min_stability
        self.max_frequency = max_frequency
        self.learning_rate = learning_rate
        
        # Resonance tracking
        self.resonance_states: Dict[int, ResonanceState] = {}
        self.resonance_history: List[Tuple[float, float]] = []
        
        # Performance metrics
        self.total_optimizations = 0
        self.avg_stability = 0.0
    
    def optimize(self, 
                vector_id: int,
                energy: float,
                time: float) -> Tuple[float, float]:
        """Optimize energy using resonance patterns.
        
        Args:
            vector_id: Unique identifier for the vector
            energy: Current energy value
            time: Current time
            
        Returns:
            Tuple[float, float]: (optimized_energy, stability)
        """
        # Get or create resonance state
        if vector_id not in self.resonance_states:
            self.resonance_states[vector_id] = ResonanceState(
                energy=energy,
                frequency=1.0,
                phase=0.0,
                stability=0.0,
                last_update=time
            )
        
        state = self.resonance_states[vector_id]
        dt = time - state.last_update
        
        if dt <= 0:
            return energy, state.stability
            
        # Update resonance parameters
        phase_shift = state.frequency * dt
        predicted_energy = state.energy * np.cos(state.phase + phase_shift)
        
        # Calculate prediction error and stability
        error = abs(energy - predicted_energy)
        stability_update = np.exp(-error) - 0.5
        state.stability = max(0.0, min(1.0,
            state.stability * (1 - self.learning_rate) + 
            stability_update * self.learning_rate
        ))
        
        # Optimize frequency based on stability
        if state.stability > self.min_stability:
            frequency_update = np.sign(error) * self.learning_rate
            state.frequency = max(0.1, min(self.max_frequency,
                state.frequency * (1 + frequency_update)
            ))
        
        # Update state
        state.energy = energy
        state.phase = (state.phase + phase_shift) % (2 * np.pi)
        state.last_update = time
        
        # Apply resonance optimization
        if state.stability > self.min_stability:
            optimized_energy = energy * (1.0 + state.stability * 0.5)
        else:
            optimized_energy = energy
            
        # Update metrics
        self.total_optimizations += 1
        self.resonance_history.append((time, state.stability))
        if len(self.resonance_history) > 1000:
            self.resonance_history.pop(0)
        self.avg_stability = np.mean([s for _, s in self.resonance_history[-100:]])
        
        return optimized_energy, state.stability
    
    def get_stats(self) -> Dict[str, float]:
        """Get optimizer statistics."""
        return {
            'total_optimizations': self.total_optimizations,
            'avg_stability': float(self.avg_stability),
            'active_resonances': len([s for s in self.resonance_states.values() 
                                    if s.stability > self.min_stability])
        }
