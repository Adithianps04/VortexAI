import numpy as np
from typing import List, Dict, Optional, Tuple

class CustomTokenizer:
    """Custom tokenizer for VortexLLM."""
    
    def __init__(self, vector_dim: int = 512, nearby_size: int = 3):
        """Initialize tokenizer."""
        self.vector_dim = vector_dim
        self.nearby_size = nearby_size
        self.vocab: Dict[str, int] = {}
        self.vectors: Dict[int, np.ndarray] = {}
        self.next_id = 0
    
    def _create_token_vector(self) -> np.ndarray:
        """Create a new token vector."""
        vector = np.random.uniform(-1, 1, self.vector_dim)
        return vector / np.linalg.norm(vector)
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        # Simple word-based tokenization for now
        words = text.split()
        tokens = []
        
        for word in words:
            if word not in self.vocab:
                # Create new token
                self.vocab[word] = self.next_id
                # Generate random vector for new token
                self.vectors[self.next_id] = np.random.randn(self.vector_dim)
                self.next_id += 1
            
            tokens.append(self.vocab[word])
            
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        tokens = text.strip().split()
        token_ids = []
        
        for token in tokens:
            # For test tokens (token_X), use X as the ID
            if token.startswith('token_') and token[6:].isdigit():
                token_id = int(token[6:])
            else:
                # For regular tokens, use hash
                token_id = hash(token) % (2**32)
            
            # Create vector if needed
            if token_id not in self.vectors:
                self.vectors[token_id] = self._create_token_vector()
                self.next_id += 1
            
            token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            if tid in self.vectors:
                tokens.append(f"token_{tid}")
        return " ".join(tokens)
    
    def get_token_vector(self, token_id: int) -> Optional[np.ndarray]:
        """Get vector for a token ID."""
        return self.vectors.get(token_id)
    
    def get_token_vectors(self, tokens: List[int]) -> np.ndarray:
        """Get vectors for tokens."""
        vectors = []
        for token in tokens:
            if token in self.vectors:
                vectors.append(self.vectors[token])
            else:
                # Generate random vector for unknown token
                vectors.append(np.random.randn(self.vector_dim))
                
        return np.array(vectors)
    
    def get_nearby_tokens(self, token_id: int, k: int = None) -> List[int]:
        """Get k nearest tokens to the given token."""
        if k is None:
            k = self.nearby_size
            
        if token_id not in self.vectors:
            return []
            
        target_vector = self.vectors[token_id]
        distances = []
        
        for tid, vector in self.vectors.items():
            if tid != token_id:
                dist = np.linalg.norm(vector - target_vector)
                distances.append((tid, dist))
                
        distances.sort(key=lambda x: x[1])
        return [tid for tid, _ in distances[:k]]
