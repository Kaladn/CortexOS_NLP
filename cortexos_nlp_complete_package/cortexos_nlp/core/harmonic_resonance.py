"""
CortexOS NLP - HarmonicResonance Module
Phase 1: Mathematical Foundation

This module implements the harmonic resonance system that measures semantic
similarity and contextual relevance between spatial anchors. This replaces
probabilistic vector similarity with mathematical certainty.

Core Principle: Semantic similarity is calculated using geometric distance
in 6D space combined with relationship strength analysis, providing
mathematically provable similarity scores.
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from .spatial_anchor import SpatialCoordinate, calculate_euclidean_distance, calculate_manhattan_distance
    from .binary_cell_memory import BinaryCellMemory, RelationshipType
except ImportError:
    # For standalone testing
    from spatial_anchor import SpatialCoordinate, calculate_euclidean_distance, calculate_manhattan_distance
    from binary_cell_memory import BinaryCellMemory, RelationshipType


@dataclass
class ResonanceResult:
    """
    Represents the result of a harmonic resonance calculation.
    This provides detailed information about the similarity between two anchors.
    """
    anchor1: SpatialCoordinate
    anchor2: SpatialCoordinate
    geometric_distance: float
    relationship_strength: float
    resonance_frequency: float
    similarity_score: float  # 0.0 to 1.0
    confidence: float       # Mathematical certainty of the result
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HarmonicResonance:
    """
    The mathematical engine for calculating semantic similarity and contextual
    relevance between spatial anchors. This replaces probabilistic similarity
    measures with deterministic mathematical calculations.
    
    The system uses:
    1. Geometric distance in 6D space
    2. Relationship strength from BinaryCellMemory
    3. Harmonic frequency analysis
    4. Mathematical certainty calculations
    """
    
    def __init__(self, memory: Optional[BinaryCellMemory] = None):
        """
        Initialize the harmonic resonance system.
        
        Args:
            memory: Optional BinaryCellMemory for relationship analysis
        """
        self.memory = memory
        self._frequency_cache = {}  # Cache for performance optimization
    
    def calculate_distance(self, 
                         anchor1: SpatialCoordinate, 
                         anchor2: SpatialCoordinate,
                         distance_type: str = "euclidean") -> float:
        """
        Calculate the geometric distance between two spatial anchors.
        
        Args:
            anchor1: First anchor
            anchor2: Second anchor
            distance_type: Type of distance calculation ("euclidean" or "manhattan")
            
        Returns:
            Geometric distance between the anchors
        """
        if distance_type == "euclidean":
            return calculate_euclidean_distance(anchor1, anchor2)
        elif distance_type == "manhattan":
            return float(calculate_manhattan_distance(anchor1, anchor2))
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")
    
    def calculate_resonance_frequency(self, 
                                    anchor1: SpatialCoordinate,
                                    anchor2: SpatialCoordinate) -> float:
        """
        Calculate the harmonic resonance frequency between two anchors.
        
        This uses the mathematical properties of the 6D coordinates to
        determine a frequency that represents their harmonic relationship.
        
        Args:
            anchor1: First anchor
            anchor2: Second anchor
            
        Returns:
            Resonance frequency (0.0 to 1.0)
        """
        # Create cache key
        key = (hash(anchor1), hash(anchor2))
        if key in self._frequency_cache:
            return self._frequency_cache[key]
        
        # Get coordinate tuples
        coords1 = anchor1.to_tuple()
        coords2 = anchor2.to_tuple()
        
        # Calculate harmonic components
        harmonic_sum = 0.0
        
        for i in range(6):
            # Calculate the harmonic relationship between corresponding dimensions
            c1, c2 = coords1[i], coords2[i]
            
            # Avoid division by zero
            if c1 == 0 and c2 == 0:
                harmonic_component = 1.0
            elif c1 == 0 or c2 == 0:
                harmonic_component = 0.0
            else:
                # Calculate harmonic ratio
                ratio = abs(c1 / c2) if abs(c2) > abs(c1) else abs(c2 / c1)
                harmonic_component = ratio
            
            # Weight by dimension (later dimensions have less influence)
            weight = 1.0 / (i + 1)
            harmonic_sum += harmonic_component * weight
        
        # Normalize to 0.0-1.0 range
        frequency = harmonic_sum / sum(1.0 / (i + 1) for i in range(6))
        frequency = max(0.0, min(1.0, frequency))
        
        # Cache the result
        self._frequency_cache[key] = frequency
        
        return frequency
    
    def calculate_relationship_strength(self, 
                                      anchor1: SpatialCoordinate,
                                      anchor2: SpatialCoordinate) -> float:
        """
        Calculate the relationship strength between two anchors using BinaryCellMemory.
        
        Args:
            anchor1: First anchor
            anchor2: Second anchor
            
        Returns:
            Relationship strength (0.0 to 1.0)
        """
        if self.memory is None:
            return 0.0
        
        return self.memory.calculate_relationship_strength(anchor1, anchor2)
    
    def calculate_resonance(self, 
                          anchor1: SpatialCoordinate,
                          anchor2: SpatialCoordinate,
                          include_relationships: bool = True,
                          distance_weight: float = 0.4,
                          relationship_weight: float = 0.3,
                          frequency_weight: float = 0.3) -> ResonanceResult:
        """
        Calculate the complete harmonic resonance between two anchors.
        
        This is the main function that combines geometric distance,
        relationship strength, and harmonic frequency to produce a
        mathematically certain similarity score.
        
        Args:
            anchor1: First anchor
            anchor2: Second anchor
            include_relationships: Whether to include relationship analysis
            distance_weight: Weight for geometric distance component
            relationship_weight: Weight for relationship strength component
            frequency_weight: Weight for harmonic frequency component
            
        Returns:
            ResonanceResult with detailed similarity analysis
        """
        # Normalize weights
        total_weight = distance_weight + relationship_weight + frequency_weight
        if total_weight == 0:
            raise ValueError("All weights cannot be zero")
        
        distance_weight /= total_weight
        relationship_weight /= total_weight
        frequency_weight /= total_weight
        
        # Calculate geometric distance
        geometric_distance = self.calculate_distance(anchor1, anchor2)
        
        # Calculate relationship strength
        if include_relationships and self.memory is not None:
            relationship_strength = self.calculate_relationship_strength(anchor1, anchor2)
        else:
            relationship_strength = 0.0
        
        # Calculate resonance frequency
        resonance_frequency = self.calculate_resonance_frequency(anchor1, anchor2)
        
        # Convert distance to similarity (inverse relationship)
        # Use a scaling factor to normalize typical distances
        max_possible_distance = math.sqrt(6 * (2000000 ** 2))  # Maximum distance in 6D space
        distance_similarity = 1.0 - (geometric_distance / max_possible_distance)
        distance_similarity = max(0.0, min(1.0, distance_similarity))
        
        # Calculate weighted similarity score
        similarity_score = (
            distance_similarity * distance_weight +
            relationship_strength * relationship_weight +
            resonance_frequency * frequency_weight
        )
        
        # Calculate confidence based on the consistency of components
        components = [distance_similarity, relationship_strength, resonance_frequency]
        active_components = [c for c in components if c > 0.0]
        
        if len(active_components) <= 1:
            confidence = 0.5  # Low confidence with only one component
        else:
            # Confidence is higher when components agree
            mean_component = sum(active_components) / len(active_components)
            variance = sum((c - mean_component) ** 2 for c in active_components) / len(active_components)
            confidence = 1.0 - min(variance, 1.0)  # Lower variance = higher confidence
        
        return ResonanceResult(
            anchor1=anchor1,
            anchor2=anchor2,
            geometric_distance=geometric_distance,
            relationship_strength=relationship_strength,
            resonance_frequency=resonance_frequency,
            similarity_score=similarity_score,
            confidence=confidence,
            metadata={
                "distance_weight": distance_weight,
                "relationship_weight": relationship_weight,
                "frequency_weight": frequency_weight,
                "distance_similarity": distance_similarity,
                "active_components": len(active_components)
            }
        )
    
    def find_most_similar(self, 
                         target_anchor: SpatialCoordinate,
                         candidate_anchors: List[SpatialCoordinate],
                         top_k: int = 5) -> List[ResonanceResult]:
        """
        Find the most similar anchors to a target anchor.
        
        Args:
            target_anchor: The anchor to find similarities for
            candidate_anchors: List of candidate anchors to compare
            top_k: Number of top results to return
            
        Returns:
            List of ResonanceResult objects, sorted by similarity score
        """
        results = []
        
        for candidate in candidate_anchors:
            if candidate == target_anchor:
                continue  # Skip self-comparison
            
            result = self.calculate_resonance(target_anchor, candidate)
            results.append(result)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return results[:top_k]
    
    def calculate_cluster_resonance(self, 
                                  anchors: List[SpatialCoordinate]) -> Dict:
        """
        Calculate the overall resonance within a cluster of anchors.
        
        This can be used to determine how cohesive a group of linguistic
        elements are in semantic space.
        
        Args:
            anchors: List of anchors to analyze as a cluster
            
        Returns:
            Dictionary with cluster analysis results
        """
        if len(anchors) < 2:
            return {
                "cluster_size": len(anchors),
                "average_similarity": 0.0,
                "cohesion_score": 0.0,
                "total_comparisons": 0
            }
        
        similarities = []
        total_comparisons = 0
        
        # Calculate pairwise similarities
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                result = self.calculate_resonance(anchors[i], anchors[j])
                similarities.append(result.similarity_score)
                total_comparisons += 1
        
        # Calculate cluster statistics
        average_similarity = sum(similarities) / len(similarities)
        
        # Cohesion score: how tightly clustered the elements are
        # Higher cohesion = more similar elements
        variance = sum((s - average_similarity) ** 2 for s in similarities) / len(similarities)
        cohesion_score = average_similarity * (1.0 - min(variance, 1.0))
        
        return {
            "cluster_size": len(anchors),
            "average_similarity": average_similarity,
            "cohesion_score": cohesion_score,
            "similarity_variance": variance,
            "total_comparisons": total_comparisons,
            "min_similarity": min(similarities),
            "max_similarity": max(similarities)
        }
    
    def clear_cache(self):
        """Clear the frequency calculation cache"""
        self._frequency_cache.clear()
    
    def cache_size(self) -> int:
        """Get the current size of the frequency cache"""
        return len(self._frequency_cache)


if __name__ == "__main__":
    # Demonstration of the HarmonicResonance system
    print("CortexOS NLP - HarmonicResonance Module Demonstration")
    print("=" * 57)
    
    try:
        from .spatial_anchor import SpatialAnchor
        from .binary_cell_memory import BinaryCellMemory, RelationshipType
    except ImportError:
        from spatial_anchor import SpatialAnchor
        from binary_cell_memory import BinaryCellMemory, RelationshipType
    
    # Create test system
    anchor_system = SpatialAnchor()
    memory = BinaryCellMemory()
    resonance = HarmonicResonance(memory)
    
    # Create anchors for test words
    test_words = ["dog", "cat", "animal", "car", "vehicle"]
    anchors = {}
    
    print("Creating spatial anchors:")
    for word in test_words:
        _, coord = anchor_system.create_anchor(word)
        anchors[word] = coord
        print(f"'{word}' -> {coord}")
    
    print("\nStoring semantic relationships:")
    # Add some relationships to test relationship-based similarity
    memory.store_relationship(
        anchors["dog"], anchors["animal"], 
        RelationshipType.HYPONYM, strength=0.9
    )
    memory.store_relationship(
        anchors["cat"], anchors["animal"],
        RelationshipType.HYPONYM, strength=0.9
    )
    memory.store_relationship(
        anchors["car"], anchors["vehicle"],
        RelationshipType.HYPONYM, strength=0.8
    )
    print("Added hyponym relationships")
    
    print("\nCalculating harmonic resonance:")
    
    # Test basic resonance calculation
    result = resonance.calculate_resonance(anchors["dog"], anchors["cat"])
    print(f"Dog <-> Cat:")
    print(f"  Similarity Score: {result.similarity_score:.4f}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Geometric Distance: {result.geometric_distance:.2f}")
    print(f"  Relationship Strength: {result.relationship_strength:.4f}")
    print(f"  Resonance Frequency: {result.resonance_frequency:.4f}")
    
    # Test with related words
    result = resonance.calculate_resonance(anchors["dog"], anchors["animal"])
    print(f"\nDog <-> Animal:")
    print(f"  Similarity Score: {result.similarity_score:.4f}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Relationship Strength: {result.relationship_strength:.4f}")
    
    # Test with unrelated words
    result = resonance.calculate_resonance(anchors["dog"], anchors["car"])
    print(f"\nDog <-> Car:")
    print(f"  Similarity Score: {result.similarity_score:.4f}")
    print(f"  Confidence: {result.confidence:.4f}")
    
    print("\nFinding most similar to 'dog':")
    candidates = [anchors[word] for word in ["cat", "animal", "car", "vehicle"]]
    similar_results = resonance.find_most_similar(anchors["dog"], candidates, top_k=3)
    
    for i, result in enumerate(similar_results, 1):
        # Find word name for display
        word_name = "unknown"
        for word, coord in anchors.items():
            if coord == result.anchor2:
                word_name = word
                break
        print(f"  {i}. {word_name}: {result.similarity_score:.4f} (confidence: {result.confidence:.4f})")
    
    print("\nCluster analysis for animal words:")
    animal_anchors = [anchors["dog"], anchors["cat"], anchors["animal"]]
    cluster_stats = resonance.calculate_cluster_resonance(animal_anchors)
    print(f"  Cluster size: {cluster_stats['cluster_size']}")
    print(f"  Average similarity: {cluster_stats['average_similarity']:.4f}")
    print(f"  Cohesion score: {cluster_stats['cohesion_score']:.4f}")
    
    print(f"\nCache size: {resonance.cache_size()} entries")

