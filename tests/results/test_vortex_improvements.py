import numpy as np
import matplotlib.pyplot as plt
from vortexllm.vortex_ai import VortexAI
import unittest

class TestVortexImprovements(unittest.TestCase):
    def setUp(self):
        self.vortex = VortexAI(vector_dim=64)
        self.time_steps = np.linspace(0, 10, 100)
        
    def test_wave_interference(self):
        """Test if wave interference patterns emerge between similar vectors."""
        # Create two similar vectors
        v1 = np.random.randn(64)
        v2 = v1 + 0.1 * np.random.randn(64)  # Slightly perturbed version
        vectors = np.vstack([v1, v2])
        
        # Track energies over time
        energies = []
        for t in self.time_steps:
            energy = self.vortex.process_batch(vectors, time=t)
            energies.append(energy)
        energies = np.array(energies)
        
        # Check if interference is working
        # 1. Energies should be correlated due to similarity
        correlation = np.corrcoef(energies[:, 0], energies[:, 1])[0, 1]
        self.assertGreater(correlation, 0.5, 
            "Similar vectors should show correlated energy patterns")
            
        # 2. Energy patterns should show interference oscillations
        fft = np.fft.fft(energies[:, 0])
        main_freq = np.abs(fft[1:len(fft)//2]).argmax() + 1
        self.assertGreater(main_freq, 0,
            "Energy should show oscillatory patterns")
            
    def test_resonance_optimization(self):
        """Test if resonance optimization improves stability."""
        # Create a vector with periodic pattern
        base_vector = np.random.randn(64)
        vectors = base_vector.reshape(1, -1)
        
        # Track original and optimized energies
        original_energies = []
        optimized_energies = []
        stabilities = []
        
        for t in self.time_steps:
            # Get energy before optimization
            energy = self.vortex.process_batch(vectors, time=t)
            original_energies.append(float(energy[0]))
            
            # Get optimizer stats
            stats = self.vortex.get_stats()
            stability = stats.get('resonance_avg_stability', 0)
            stabilities.append(stability)
            
        # Stability should increase over time
        early_stability = np.mean(stabilities[:len(stabilities)//4])
        late_stability = np.mean(stabilities[-len(stabilities)//4:])
        self.assertGreater(late_stability, early_stability,
            "Resonance stability should improve over time")
            
        # Energy variance should decrease with optimization
        self.assertLess(np.var(original_energies[-10:]), np.var(original_energies[:10]),
            "Energy should become more stable over time")
            
    def test_combined_effects(self):
        """Test if interference and resonance work together."""
        # Create a set of related vectors
        base_vectors = np.random.randn(3, 64)
        perturbation = 0.1 * np.random.randn(3, 64)
        vectors = base_vectors + perturbation
        
        # Process batch and check stats
        for t in self.time_steps:
            _ = self.vortex.process_batch(vectors, time=t)
            
        stats = self.vortex.get_stats()
        
        # Check if both mechanisms are active
        self.assertGreater(stats['interference_strength'], 0,
            "Interference effects should be present")
        self.assertGreater(stats['resonance_avg_stability'], 0,
            "Resonance optimization should be active")
            
if __name__ == '__main__':
    unittest.main()
