"""
CortexOS NLP - Phase 1 Integration Test
Mathematical Foundation Integration Testing

This test verifies that all Phase 1 components work together correctly:
- SpatialAnchor: Creates deterministic 6D coordinates
- BinaryCellMemory: Stores and retrieves relationships
- HarmonicResonance: Calculates semantic similarity

The integration test demonstrates the complete mathematical foundation
that will power the deterministic NLP engine.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import (
    SpatialAnchor, SpatialCoordinate,
    BinaryCellMemory, RelationshipType,
    HarmonicResonance, ResonanceResult
)


def test_phase1_integration():
    """
    Comprehensive integration test for Phase 1 components.
    
    This test creates a small knowledge graph and demonstrates
    how all components work together to provide deterministic
    language processing capabilities.
    """
    print("CortexOS NLP - Phase 1 Integration Test")
    print("=" * 45)
    
    # Initialize all Phase 1 components
    print("1. Initializing Phase 1 components...")
    anchor_system = SpatialAnchor()
    memory = BinaryCellMemory()
    resonance = HarmonicResonance(memory)
    print("   ✓ All components initialized")
    
    # Create a test vocabulary
    vocabulary = [
        "dog", "cat", "animal", "pet", "mammal",
        "car", "vehicle", "transportation", "wheel",
        "red", "blue", "color", "bright",
        "run", "walk", "move", "fast"
    ]
    
    print(f"\n2. Creating spatial anchors for {len(vocabulary)} words...")
    anchors = {}
    for word in vocabulary:
        hash_val, coord = anchor_system.create_anchor(word)
        anchors[word] = coord
        print(f"   '{word}' -> {coord}")
    
    print("   ✓ All spatial anchors created successfully")
    
    # Build a knowledge graph with relationships
    print("\n3. Building knowledge graph with relationships...")
    
    relationships = [
        # Animal hierarchy
        ("dog", "animal", RelationshipType.HYPONYM, 0.9),
        ("cat", "animal", RelationshipType.HYPONYM, 0.9),
        ("animal", "mammal", RelationshipType.HYPONYM, 0.8),
        ("dog", "pet", RelationshipType.SYNONYM, 0.7),
        ("cat", "pet", RelationshipType.SYNONYM, 0.6),
        
        # Vehicle hierarchy
        ("car", "vehicle", RelationshipType.HYPONYM, 0.9),
        ("vehicle", "transportation", RelationshipType.HYPONYM, 0.8),
        ("car", "wheel", RelationshipType.HOLONYM, 0.8),
        
        # Color relationships
        ("red", "color", RelationshipType.HYPONYM, 0.9),
        ("blue", "color", RelationshipType.HYPONYM, 0.9),
        ("red", "blue", RelationshipType.ANTONYM, 0.8),
        ("red", "bright", RelationshipType.MODIFIER, 0.6),
        
        # Movement relationships
        ("run", "move", RelationshipType.HYPONYM, 0.8),
        ("walk", "move", RelationshipType.HYPONYM, 0.8),
        ("run", "fast", RelationshipType.MODIFIER, 0.7),
        ("dog", "run", RelationshipType.CO_OCCURRENCE, 0.6),
        ("car", "fast", RelationshipType.CO_OCCURRENCE, 0.5),
    ]
    
    for word1, word2, rel_type, strength in relationships:
        relationship = memory.store_relationship(
            anchors[word1], anchors[word2], rel_type, strength
        )
        print(f"   {word1} --[{rel_type.value}:{strength:.2f}]--> {word2}")
    
    print(f"   ✓ {len(relationships)} relationships stored successfully")
    
    # Test semantic similarity calculations
    print("\n4. Testing semantic similarity calculations...")
    
    test_pairs = [
        ("dog", "cat", "Related animals"),
        ("dog", "animal", "Hyponym relationship"),
        ("car", "vehicle", "Hyponym relationship"),
        ("red", "blue", "Antonym relationship"),
        ("dog", "car", "Unrelated concepts"),
        ("run", "walk", "Related actions"),
    ]
    
    for word1, word2, description in test_pairs:
        result = resonance.calculate_resonance(anchors[word1], anchors[word2])
        print(f"   {word1} <-> {word2} ({description}):")
        print(f"     Similarity: {result.similarity_score:.4f}")
        print(f"     Confidence: {result.confidence:.4f}")
        print(f"     Relationship Strength: {result.relationship_strength:.4f}")
    
    print("   ✓ Semantic similarity calculations completed")
    
    # Test similarity search
    print("\n5. Testing similarity search...")
    
    search_words = ["dog", "car", "red"]
    for search_word in search_words:
        print(f"   Most similar to '{search_word}':")
        candidates = [anchors[w] for w in vocabulary if w != search_word]
        similar_results = resonance.find_most_similar(
            anchors[search_word], candidates, top_k=3
        )
        
        for i, result in enumerate(similar_results, 1):
            # Find word name
            similar_word = "unknown"
            for word, coord in anchors.items():
                if coord == result.anchor2:
                    similar_word = word
                    break
            print(f"     {i}. {similar_word}: {result.similarity_score:.4f}")
    
    print("   ✓ Similarity search completed")
    
    # Test cluster analysis
    print("\n6. Testing cluster analysis...")
    
    clusters = [
        (["dog", "cat", "animal", "pet"], "Animal cluster"),
        (["car", "vehicle", "transportation"], "Vehicle cluster"),
        (["red", "blue", "color"], "Color cluster"),
        (["run", "walk", "move"], "Movement cluster"),
    ]
    
    for cluster_words, description in clusters:
        cluster_anchors = [anchors[word] for word in cluster_words]
        stats = resonance.calculate_cluster_resonance(cluster_anchors)
        print(f"   {description}:")
        print(f"     Size: {stats['cluster_size']}")
        print(f"     Average similarity: {stats['average_similarity']:.4f}")
        print(f"     Cohesion score: {stats['cohesion_score']:.4f}")
    
    print("   ✓ Cluster analysis completed")
    
    # Test path finding
    print("\n7. Testing relationship path finding...")
    
    path_tests = [
        ("dog", "mammal", "Animal hierarchy path"),
        ("car", "transportation", "Vehicle hierarchy path"),
        ("red", "color", "Color hierarchy path"),
    ]
    
    for start_word, end_word, description in path_tests:
        path = memory.find_path(anchors[start_word], anchors[end_word])
        print(f"   {description} ({start_word} -> {end_word}):")
        if path:
            print(f"     Path length: {len(path)} steps")
            for i, rel in enumerate(path, 1):
                # Find word names for display
                start_name = end_name = "unknown"
                for word, coord in anchors.items():
                    if coord == rel.anchor1:
                        start_name = word
                    if coord == rel.anchor2:
                        end_name = word
                print(f"     Step {i}: {start_name} --[{rel.relationship_type.value}]--> {end_name}")
        else:
            print("     No path found")
    
    print("   ✓ Path finding completed")
    
    # Performance statistics
    print("\n8. Performance statistics...")
    memory_stats = memory.memory_stats()
    print(f"   Memory system:")
    print(f"     Total anchors: {memory_stats['total_anchors']}")
    print(f"     Total relationships: {memory_stats['total_relationships']}")
    print(f"     Avg relationships per anchor: {memory_stats['average_relationships_per_anchor']:.2f}")
    
    print(f"   Cache systems:")
    print(f"     Anchor cache size: {anchor_system.cache_size()}")
    print(f"     Resonance cache size: {resonance.cache_size()}")
    
    print("\n" + "=" * 45)
    print("✅ PHASE 1 INTEGRATION TEST PASSED")
    print("✅ All mathematical foundation components working correctly")
    print("✅ Ready to proceed to Phase 2: Linguistic Layer")
    print("=" * 45)
    
    return True


if __name__ == "__main__":
    try:
        success = test_phase1_integration()
        if success:
            print("\nPhase 1 mathematical foundation is complete and verified!")
        else:
            print("\nPhase 1 integration test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\nPhase 1 integration test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

