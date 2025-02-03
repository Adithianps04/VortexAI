import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vortexllm.vortex_ai import VortexAI

def test_vortex():
    print("Starting VortexAI tests...")
    
    # Initialize with default parameters
    vortex = VortexAI()
    
    # Test 1: Basic weight processing
    print("\nTest 1: Basic weight processing")
    x = 0.5
    result = vortex.process_weight(x)
    assert -1 <= result <= 1, f"Result {result} outside [-1, 1] range"
    print(f"[PASS] Successfully processed weight: input={x:.2f}, output={result:.2f}")
    
    # Test 2: Batch processing
    print("\nTest 2: Batch processing")
    weights = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    results = vortex.process_batch(weights)
    assert np.all(results >= -1) and np.all(results <= 1), "Batch results outside [-1, 1] range"
    print("[PASS] Successfully processed batch")
    print("Input weights:", weights)
    print("Output weights:", results)
    
    # Test 3: Time evolution
    print("\nTest 3: Time evolution")
    x = 0.5
    results = []
    times = [0, 1, 2, 3, 4]
    
    for t in times:
        vortex.increment_time()
        result = vortex.process_weight(x)
        results.append(result)
    
    print("Time evolution of weight 0.5:")
    for t, r in zip(times, results):
        print(f"Time {t}: {r:.4f}")
    
    # Test 4: Parameter updates
    print("\nTest 4: Parameter updates")
    original_params = vortex.get_parameters()
    print("Original parameters:", original_params)
    
    vortex.set_wave_parameters(amplitude=2.0, decay_rate=0.2)
    new_params = vortex.get_parameters()
    print("Updated parameters:", new_params)
    assert new_params[1] == 2.0 and new_params[2] == 0.2, "Parameter update failed"
    print("[PASS] Successfully updated parameters")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_vortex()
