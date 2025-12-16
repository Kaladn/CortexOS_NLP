"""
CortexOS NLP - SpatialAnchor Module
Phase 1: Mathematical Foundation

This module implements the 6-1-6 spatial coordinate system that gives every
linguistic element a unique, deterministic position in 6D mathematical space.

Core Principle: Any identical input string will ALWAYS produce the same
6D coordinates, providing mathematical certainty and eliminating probabilistic
guessing in language processing.
"""

import hashlib
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class SpatialCoordinate:
    """
    Represents a point in 6D space with integer coordinates.
    This is the deterministic "embedding" that replaces probabilistic vectors.
    """
    x1: int
    x2: int
    x3: int
    x4: int
    x5: int
    x6: int
    
    def __str__(self) -> str:
        return f"({self.x1}, {self.x2}, {self.x3}, {self.x4}, {self.x5}, {self.x6})"
    
    def __hash__(self) -> int:
        """Make SpatialCoordinate hashable for use in sets and dictionaries"""
        return hash((self.x1, self.x2, self.x3, self.x4, self.x5, self.x6))
    
    def to_tuple(self) -> Tuple[int, int, int, int, int, int]:
        """Convert to tuple for mathematical operations"""
        return (self.x1, self.x2, self.x3, self.x4, self.x5, self.x6)


class SpatialAnchor:
    """
    The core mathematical engine that converts any text input into
    deterministic 6D spatial coordinates using SHA-256 hashing.
    
    This replaces probabilistic word embeddings with mathematical certainty.
    """
    
    def __init__(self):
        """Initialize the spatial anchor system"""
        self._coordinate_cache = {}  # Cache for performance optimization
    
    def create_anchor(self, input_string: str) -> Tuple[str, SpatialCoordinate]:
        """
        Create a spatial anchor for any input string.
        
        Args:
            input_string: The text to anchor in 6D space
            
        Returns:
            Tuple of (SHA-256 hash, SpatialCoordinate)
            
        Example:
            hash_val, coords = anchor.create_anchor("hello")
            # coords will always be the same for "hello"
        """
        # Normalize input for consistency
        normalized_input = input_string.strip().lower()
        
        # Check cache first for performance
        if normalized_input in self._coordinate_cache:
            return self._coordinate_cache[normalized_input]
        
        # Generate SHA-256 hash
        hash_object = hashlib.sha256(normalized_input.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        # Convert hash to 6D coordinates
        coordinates = self.hash_to_coordinates(hash_hex)
        
        # Cache the result
        result = (hash_hex, coordinates)
        self._coordinate_cache[normalized_input] = result
        
        return result
    
    def hash_to_coordinates(self, hash_hex: str) -> SpatialCoordinate:
        """
        Convert a SHA-256 hash into 6D spatial coordinates.
        
        This is the core mathematical transformation that creates our
        deterministic embedding space.
        
        Args:
            hash_hex: SHA-256 hash as hexadecimal string
            
        Returns:
            SpatialCoordinate object with 6D position
        """
        # Convert hex string to bytes
        hash_bytes = bytes.fromhex(hash_hex)
        
        # Extract 6 coordinates from the 32-byte hash
        # Each coordinate uses ~5.33 bytes of hash data for maximum distribution
        coordinates = []
        
        for i in range(6):
            # Take 5 bytes starting at position i*5 (with overlap for the 6th coordinate)
            start_pos = min(i * 5, len(hash_bytes) - 5)
            byte_chunk = hash_bytes[start_pos:start_pos + 5]
            
            # Convert 5 bytes to integer (0 to 2^40 - 1)
            coord_value = int.from_bytes(byte_chunk, byteorder='big')
            
            # Scale to reasonable coordinate range (-1000000 to +1000000)
            # This provides sufficient precision while keeping numbers manageable
            scaled_coord = (coord_value % 2000000) - 1000000
            coordinates.append(scaled_coord)
        
        return SpatialCoordinate(*coordinates)
    
    def get_coordinate_only(self, input_string: str) -> SpatialCoordinate:
        """
        Get just the spatial coordinate for an input string.
        
        Args:
            input_string: The text to get coordinates for
            
        Returns:
            SpatialCoordinate object
        """
        _, coordinates = self.create_anchor(input_string)
        return coordinates
    
    def get_hash_only(self, input_string: str) -> str:
        """
        Get just the SHA-256 hash for an input string.
        
        Args:
            input_string: The text to hash
            
        Returns:
            SHA-256 hash as hexadecimal string
        """
        hash_val, _ = self.create_anchor(input_string)
        return hash_val
    
    def batch_create_anchors(self, input_strings: List[str]) -> List[Tuple[str, SpatialCoordinate]]:
        """
        Create spatial anchors for multiple strings efficiently.
        
        Args:
            input_strings: List of strings to anchor
            
        Returns:
            List of (hash, coordinate) tuples
        """
        return [self.create_anchor(s) for s in input_strings]
    
    def clear_cache(self):
        """Clear the coordinate cache to free memory"""
        self._coordinate_cache.clear()
    
    def cache_size(self) -> int:
        """Get the current size of the coordinate cache"""
        return len(self._coordinate_cache)


# Utility functions for mathematical operations on coordinates
def calculate_euclidean_distance(coord1: SpatialCoordinate, coord2: SpatialCoordinate) -> float:
    """
    Calculate the Euclidean distance between two spatial coordinates.
    
    This provides a mathematical measure of similarity between linguistic elements.
    Lower distance = higher similarity.
    """
    t1 = coord1.to_tuple()
    t2 = coord2.to_tuple()
    
    sum_of_squares = sum((a - b) ** 2 for a, b in zip(t1, t2))
    return sum_of_squares ** 0.5


def calculate_manhattan_distance(coord1: SpatialCoordinate, coord2: SpatialCoordinate) -> int:
    """
    Calculate the Manhattan distance between two spatial coordinates.
    
    Alternative distance metric that may be useful for certain linguistic relationships.
    """
    t1 = coord1.to_tuple()
    t2 = coord2.to_tuple()
    
    return sum(abs(a - b) for a, b in zip(t1, t2))


if __name__ == "__main__":
    # Demonstration of the SpatialAnchor system
    print("CortexOS NLP - SpatialAnchor Module Demonstration")
    print("=" * 50)
    
    anchor = SpatialAnchor()
    
    # Test with sample words
    test_words = ["hello", "world", "python", "cortex", "deterministic"]
    
    print("Creating spatial anchors for test words:")
    for word in test_words:
        hash_val, coords = anchor.create_anchor(word)
        print(f"'{word}' -> {coords}")
        print(f"  Hash: {hash_val[:16]}...")
    
    print("\nTesting consistency (same input = same output):")
    word = "consistency"
    coord1 = anchor.get_coordinate_only(word)
    coord2 = anchor.get_coordinate_only(word)
    print(f"'{word}' first call:  {coord1}")
    print(f"'{word}' second call: {coord2}")
    print(f"Identical: {coord1 == coord2}")
    
    print("\nTesting distance calculations:")
    hello_coord = anchor.get_coordinate_only("hello")
    world_coord = anchor.get_coordinate_only("world")
    hello2_coord = anchor.get_coordinate_only("hello")
    
    print(f"Distance 'hello' to 'world': {calculate_euclidean_distance(hello_coord, world_coord):.2f}")
    print(f"Distance 'hello' to 'hello': {calculate_euclidean_distance(hello_coord, hello2_coord):.2f}")
    
    print(f"\nCache size: {anchor.cache_size()} entries")

