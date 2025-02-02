import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time
import sys

class BrainMemory:
    """Memory management for concept storage and retrieval."""
    
    def __init__(self, max_concepts: int = 1000, compression_threshold: float = 0.85):
        """Initialize brain memory system."""
        self.concepts = {}
        self.max_concepts = max_concepts
        self.compression_threshold = compression_threshold
        self.access_counts = defaultdict(int)
        self.last_access = defaultdict(float)
        self.total_compressions = 0
        self.memory_usage = 0
        
    def add_concept(self, key: str, vector: np.ndarray, energy: float = 1.0):
        """Add a concept to memory with improved compression."""
        # Update memory usage
        self.memory_usage += vector.nbytes
        
        # Check if compression needed
        if len(self.concepts) >= self.max_concepts:
            self._compress_memory()
            
        # Update concept with energy weighting
        if key in self.concepts:
            old_vector, old_energy = self.concepts[key]
            total_energy = old_energy + energy
            weighted_vector = (old_vector * old_energy + vector * energy) / total_energy
            self.concepts[key] = (weighted_vector, total_energy)
        else:
            self.concepts[key] = (vector, energy)
            
        # Update access metrics
        self.access_counts[key] += 1
        self.last_access[key] = time.time()
        
    def _compress_memory(self):
        """Compress memory using improved similarity-based merging."""
        if len(self.concepts) < 2:
            return
            
        # Calculate concept similarities
        keys = list(self.concepts.keys())
        best_pair = None
        best_similarity = -1
        
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                vec1, _ = self.concepts[keys[i]]
                vec2, _ = self.concepts[keys[j]]
                
                # Calculate cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (keys[i], keys[j])
                    
        # Merge if similarity exceeds threshold
        if best_similarity > self.compression_threshold and best_pair:
            key1, key2 = best_pair
            vec1, energy1 = self.concepts[key1]
            vec2, energy2 = self.concepts[key2]
            
            # Weight by energy and access frequency
            weight1 = energy1 * self.access_counts[key1]
            weight2 = energy2 * self.access_counts[key2]
            total_weight = weight1 + weight2
            
            # Merge vectors
            merged_vector = (vec1 * weight1 + vec2 * weight2) / total_weight
            merged_energy = (energy1 + energy2) / 2
            
            # Create new key and update metrics
            new_key = f"{key1}_{key2}"
            self.concepts[new_key] = (merged_vector, merged_energy)
            self.access_counts[new_key] = (self.access_counts[key1] + self.access_counts[key2]) // 2
            self.last_access[new_key] = max(self.last_access[key1], self.last_access[key2])
            
            # Remove old concepts
            del self.concepts[key1]
            del self.concepts[key2]
            del self.access_counts[key1]
            del self.access_counts[key2]
            del self.last_access[key1]
            del self.last_access[key2]
            
            self.total_compressions += 1
            self.memory_usage = sum(v[0].nbytes for v in self.concepts.values())
            
    def store_concept(self, key: str, vector: np.ndarray) -> None:
        """Store a concept vector in memory (alias for add_concept)."""
        self.add_concept(key, vector)
        
    def get_concept(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a concept with access tracking."""
        if key in self.concepts:
            self.access_counts[key] += 1
            self.last_access[key] = time.time()
            return self.concepts[key][0]
        return None
        
    def get_nearest_concepts(self, query_vector: np.ndarray, k: int = 1) -> List[Tuple[str, np.ndarray]]:
        """Find k nearest concepts to query vector."""
        if not self.concepts:
            return []
            
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
            
        # Calculate similarities
        similarities = []
        for key, (vector, _) in self.concepts.items():
            # Normalize concept vector
            vec_norm = np.linalg.norm(vector)
            if vec_norm > 0:
                vector = vector / vec_norm
                
            # Calculate cosine similarity
            similarity = np.dot(query_vector, vector)
            similarities.append((similarity, key, vector))
            
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        return [(key, vector) for _, key, vector in similarities[:k]]

    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            'total_concepts': len(self.concepts),
            'total_compressions': self.total_compressions,
            'memory_usage': self.memory_usage,
            'compression_ratio': 1 - (len(self.concepts) / max(1, self.total_compressions + len(self.concepts))),
            'avg_access_count': float(np.mean(list(self.access_counts.values()))) if self.access_counts else 0.0,
            'concepts': list(self.concepts.keys())
        }
