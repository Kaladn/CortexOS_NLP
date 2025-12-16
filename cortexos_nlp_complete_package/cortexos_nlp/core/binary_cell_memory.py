"""
CortexOS NLP - BinaryCellMemory Module
Phase 1: Mathematical Foundation

This module implements the binary cell memory system that stores relationships
between spatial anchors. This replaces the "attention mechanism" in transformers
with a deterministic, graph-like memory structure.

Core Principle: Every relationship between linguistic elements is stored as
a mathematical connection with defined types and strengths, enabling instant
retrieval and guaranteed consistency.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
try:
    from .spatial_anchor import SpatialCoordinate
except ImportError:
    # For standalone testing
    from spatial_anchor import SpatialCoordinate


class RelationshipType(Enum):
    """
    Defines the types of relationships that can exist between linguistic elements.
    This replaces probabilistic attention with explicit, deterministic connections.
    """
    # Semantic relationships
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    HYPERNYM = "hypernym"  # "animal" is hypernym of "dog"
    HYPONYM = "hyponym"    # "dog" is hyponym of "animal"
    MERONYM = "meronym"    # "wheel" is meronym of "car"
    HOLONYM = "holonym"    # "car" is holonym of "wheel"
    
    # Grammatical relationships
    SUBJECT = "subject"
    OBJECT = "object"
    MODIFIER = "modifier"
    COMPLEMENT = "complement"
    DEPENDENCY = "dependency"
    
    # Contextual relationships
    CO_OCCURRENCE = "co_occurrence"  # Words that appear together frequently
    SEQUENCE = "sequence"            # Word order relationships
    SIMILARITY = "similarity"        # Mathematical similarity
    
    # Custom relationships
    CUSTOM = "custom"


@dataclass
class Relationship:
    """
    Represents a connection between two spatial anchors.
    This is the fundamental unit of linguistic knowledge in our system.
    """
    anchor1: SpatialCoordinate
    anchor2: SpatialCoordinate
    relationship_type: RelationshipType
    strength: float  # 0.0 to 1.0, representing the certainty/importance of this relationship
    metadata: Dict = None  # Additional information about the relationship
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __str__(self) -> str:
        return f"{self.anchor1} --[{self.relationship_type.value}:{self.strength:.2f}]--> {self.anchor2}"


class BinaryCellMemory:
    """
    The deterministic memory system that stores and retrieves relationships
    between spatial anchors. This replaces probabilistic attention mechanisms
    with mathematical certainty.
    
    Architecture:
    - Forward index: anchor -> list of outgoing relationships
    - Reverse index: anchor -> list of incoming relationships
    - Type index: relationship_type -> list of relationships of that type
    """
    
    def __init__(self):
        """Initialize the binary cell memory system"""
        # Forward relationships: anchor -> outgoing relationships
        self._forward_index: Dict[str, List[Relationship]] = {}
        
        # Reverse relationships: anchor -> incoming relationships
        self._reverse_index: Dict[str, List[Relationship]] = {}
        
        # Type index: relationship_type -> relationships of that type
        self._type_index: Dict[RelationshipType, List[Relationship]] = {}
        
        # Initialize type index
        for rel_type in RelationshipType:
            self._type_index[rel_type] = []
    
    def _coordinate_to_key(self, coord: SpatialCoordinate) -> str:
        """Convert spatial coordinate to string key for indexing"""
        return f"{coord.x1},{coord.x2},{coord.x3},{coord.x4},{coord.x5},{coord.x6}"
    
    def store_relationship(self, 
                         anchor1: SpatialCoordinate, 
                         anchor2: SpatialCoordinate,
                         relationship_type: RelationshipType,
                         strength: float = 1.0,
                         metadata: Dict = None) -> Relationship:
        """
        Store a relationship between two spatial anchors.
        
        Args:
            anchor1: Source anchor
            anchor2: Target anchor
            relationship_type: Type of relationship
            strength: Strength/certainty of relationship (0.0 to 1.0)
            metadata: Additional information about the relationship
            
        Returns:
            The created Relationship object
        """
        # Validate strength
        strength = max(0.0, min(1.0, strength))
        
        # Create relationship
        relationship = Relationship(
            anchor1=anchor1,
            anchor2=anchor2,
            relationship_type=relationship_type,
            strength=strength,
            metadata=metadata or {}
        )
        
        # Update forward index
        key1 = self._coordinate_to_key(anchor1)
        if key1 not in self._forward_index:
            self._forward_index[key1] = []
        self._forward_index[key1].append(relationship)
        
        # Update reverse index
        key2 = self._coordinate_to_key(anchor2)
        if key2 not in self._reverse_index:
            self._reverse_index[key2] = []
        self._reverse_index[key2].append(relationship)
        
        # Update type index
        self._type_index[relationship_type].append(relationship)
        
        return relationship
    
    def get_outgoing_relationships(self, anchor: SpatialCoordinate) -> List[Relationship]:
        """
        Get all relationships where the anchor is the source.
        
        Args:
            anchor: The source anchor
            
        Returns:
            List of outgoing relationships
        """
        key = self._coordinate_to_key(anchor)
        return self._forward_index.get(key, [])
    
    def get_incoming_relationships(self, anchor: SpatialCoordinate) -> List[Relationship]:
        """
        Get all relationships where the anchor is the target.
        
        Args:
            anchor: The target anchor
            
        Returns:
            List of incoming relationships
        """
        key = self._coordinate_to_key(anchor)
        return self._reverse_index.get(key, [])
    
    def get_all_relationships(self, anchor: SpatialCoordinate) -> List[Relationship]:
        """
        Get all relationships involving the anchor (both incoming and outgoing).
        
        Args:
            anchor: The anchor to search for
            
        Returns:
            List of all relationships involving this anchor
        """
        outgoing = self.get_outgoing_relationships(anchor)
        incoming = self.get_incoming_relationships(anchor)
        return outgoing + incoming
    
    def get_relationships_by_type(self, 
                                relationship_type: RelationshipType,
                                anchor: Optional[SpatialCoordinate] = None) -> List[Relationship]:
        """
        Get relationships of a specific type, optionally filtered by anchor.
        
        Args:
            relationship_type: Type of relationships to retrieve
            anchor: Optional anchor to filter by
            
        Returns:
            List of relationships of the specified type
        """
        relationships = self._type_index[relationship_type]
        
        if anchor is None:
            return relationships
        
        # Filter by anchor
        key = self._coordinate_to_key(anchor)
        filtered = []
        for rel in relationships:
            key1 = self._coordinate_to_key(rel.anchor1)
            key2 = self._coordinate_to_key(rel.anchor2)
            if key1 == key or key2 == key:
                filtered.append(rel)
        
        return filtered
    
    def find_path(self, 
                  start_anchor: SpatialCoordinate,
                  end_anchor: SpatialCoordinate,
                  max_depth: int = 3) -> Optional[List[Relationship]]:
        """
        Find a path of relationships between two anchors.
        
        This enables discovering indirect connections between linguistic elements.
        
        Args:
            start_anchor: Starting anchor
            end_anchor: Target anchor
            max_depth: Maximum path length to search
            
        Returns:
            List of relationships forming a path, or None if no path found
        """
        if max_depth <= 0:
            return None
        
        start_key = self._coordinate_to_key(start_anchor)
        end_key = self._coordinate_to_key(end_anchor)
        
        # Breadth-first search
        queue = [(start_anchor, [])]
        visited = {start_key}
        
        while queue:
            current_anchor, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            # Get outgoing relationships from current anchor
            outgoing = self.get_outgoing_relationships(current_anchor)
            
            for relationship in outgoing:
                next_key = self._coordinate_to_key(relationship.anchor2)
                
                # Found target
                if next_key == end_key:
                    return path + [relationship]
                
                # Continue search if not visited and within depth limit
                if next_key not in visited and len(path) < max_depth - 1:
                    visited.add(next_key)
                    queue.append((relationship.anchor2, path + [relationship]))
        
        return None
    
    def get_connected_anchors(self, 
                            anchor: SpatialCoordinate,
                            relationship_types: Optional[List[RelationshipType]] = None) -> Set[SpatialCoordinate]:
        """
        Get all anchors connected to the given anchor.
        
        Args:
            anchor: The anchor to find connections for
            relationship_types: Optional filter for relationship types
            
        Returns:
            Set of connected anchors
        """
        relationships = self.get_all_relationships(anchor)
        
        if relationship_types:
            relationships = [r for r in relationships if r.relationship_type in relationship_types]
        
        connected = set()
        anchor_key = self._coordinate_to_key(anchor)
        
        for rel in relationships:
            key1 = self._coordinate_to_key(rel.anchor1)
            key2 = self._coordinate_to_key(rel.anchor2)
            
            if key1 == anchor_key:
                connected.add(rel.anchor2)
            else:
                connected.add(rel.anchor1)
        
        return connected
    
    def calculate_relationship_strength(self, 
                                      anchor1: SpatialCoordinate,
                                      anchor2: SpatialCoordinate) -> float:
        """
        Calculate the total relationship strength between two anchors.
        
        This considers all relationships between the anchors and their strengths.
        
        Args:
            anchor1: First anchor
            anchor2: Second anchor
            
        Returns:
            Total relationship strength (0.0 to 1.0+)
        """
        relationships = self.get_outgoing_relationships(anchor1)
        key2 = self._coordinate_to_key(anchor2)
        
        total_strength = 0.0
        for rel in relationships:
            if self._coordinate_to_key(rel.anchor2) == key2:
                total_strength += rel.strength
        
        # Also check reverse direction
        relationships = self.get_outgoing_relationships(anchor2)
        key1 = self._coordinate_to_key(anchor1)
        
        for rel in relationships:
            if self._coordinate_to_key(rel.anchor2) == key1:
                total_strength += rel.strength
        
        return min(total_strength, 1.0)  # Cap at 1.0 for consistency
    
    def memory_stats(self) -> Dict:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        total_relationships = sum(len(rels) for rels in self._forward_index.values())
        
        type_counts = {}
        for rel_type, rels in self._type_index.items():
            type_counts[rel_type.value] = len(rels)
        
        return {
            "total_anchors": len(self._forward_index),
            "total_relationships": total_relationships,
            "relationship_types": type_counts,
            "average_relationships_per_anchor": total_relationships / max(len(self._forward_index), 1)
        }
    
    def clear_memory(self):
        """Clear all stored relationships"""
        self._forward_index.clear()
        self._reverse_index.clear()
        for rel_type in RelationshipType:
            self._type_index[rel_type].clear()


if __name__ == "__main__":
    # Demonstration of the BinaryCellMemory system
    print("CortexOS NLP - BinaryCellMemory Module Demonstration")
    print("=" * 55)
    
    try:
        from .spatial_anchor import SpatialAnchor
    except ImportError:
        from spatial_anchor import SpatialAnchor
    
    # Create spatial anchors for test
    anchor_system = SpatialAnchor()
    memory = BinaryCellMemory()
    
    # Create anchors for test words
    test_words = ["dog", "animal", "pet", "cat", "mammal"]
    anchors = {}
    
    print("Creating spatial anchors:")
    for word in test_words:
        _, coord = anchor_system.create_anchor(word)
        anchors[word] = coord
        print(f"'{word}' -> {coord}")
    
    print("\nStoring relationships:")
    
    # Store semantic relationships
    memory.store_relationship(
        anchors["dog"], anchors["animal"], 
        RelationshipType.HYPONYM, strength=0.9,
        metadata={"source": "wordnet"}
    )
    print("dog --[hyponym:0.90]--> animal")
    
    memory.store_relationship(
        anchors["cat"], anchors["animal"],
        RelationshipType.HYPONYM, strength=0.9
    )
    print("cat --[hyponym:0.90]--> animal")
    
    memory.store_relationship(
        anchors["dog"], anchors["pet"],
        RelationshipType.SYNONYM, strength=0.7
    )
    print("dog --[synonym:0.70]--> pet")
    
    memory.store_relationship(
        anchors["animal"], anchors["mammal"],
        RelationshipType.HYPONYM, strength=0.8
    )
    print("animal --[hyponym:0.80]--> mammal")
    
    print("\nQuerying relationships:")
    
    # Test relationship queries
    dog_relationships = memory.get_all_relationships(anchors["dog"])
    print(f"All relationships for 'dog': {len(dog_relationships)}")
    for rel in dog_relationships:
        print(f"  {rel}")
    
    # Test path finding
    print(f"\nFinding path from 'dog' to 'mammal':")
    path = memory.find_path(anchors["dog"], anchors["mammal"])
    if path:
        print("Path found:")
        for rel in path:
            print(f"  {rel}")
    else:
        print("No path found")
    
    # Test connected anchors
    connected = memory.get_connected_anchors(anchors["animal"])
    print(f"\nAnchors connected to 'animal': {len(connected)}")
    
    # Memory statistics
    stats = memory.memory_stats()
    print(f"\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

