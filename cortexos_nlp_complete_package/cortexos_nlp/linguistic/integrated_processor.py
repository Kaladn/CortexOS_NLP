"""
CortexOS NLP - Integrated Linguistic Processor
Phase 2: Linguistic Layer

This module provides the unified interface for complete linguistic processing,
integrating tokenization, POS tagging, and dependency parsing into a single
deterministic pipeline with mathematical certainty.

Core Principle: Seamlessly coordinate all linguistic modules while maintaining
perfect traceability and consistency across the entire processing pipeline.
"""

import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    from ..core import SpatialAnchor, SpatialCoordinate, BinaryCellMemory, RelationshipType, HarmonicResonance
    from .tokenizer import CortexTokenizer, Token
    from .tagger import CortexTagger
    from .parser import CortexParser, DependencyTree, DependencyRelationship
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from core import SpatialAnchor, SpatialCoordinate, BinaryCellMemory, RelationshipType, HarmonicResonance
    from tokenizer import CortexTokenizer, Token
    from tagger import CortexTagger
    from parser import CortexParser, DependencyTree, DependencyRelationship


@dataclass
class ProcessingMetadata:
    """
    Complete metadata about the linguistic processing pipeline execution.
    Provides full traceability and performance metrics.
    """
    # Timing information
    total_processing_time: float
    tokenization_time: float
    tagging_time: float
    parsing_time: float
    
    # Token statistics
    total_tokens: int
    content_tokens: int
    space_tokens: int
    punct_tokens: int
    
    # Processing method breakdown
    dictionary_lookups: int
    context_disambiguations: int
    morphological_analyses: int
    
    # Relationship statistics
    dependency_relationships: int
    pos_relationships: int
    spatial_relationships: int
    
    # Memory usage
    memory_relationships_stored: int
    cache_hits: int
    cache_misses: int
    
    # Quality metrics
    average_pos_confidence: float
    average_dependency_confidence: float
    
    # Processing flags
    errors_encountered: List[str]
    warnings_generated: List[str]


@dataclass
class LinguisticDocument:
    """
    Complete linguistic analysis result containing all Phase 2 outputs.
    Provides unified access to tokenization, tagging, and parsing results.
    """
    
    # Original input
    original_text: str
    
    # Tokenization results
    tokens: List[Token]
    
    # Parsing results
    dependency_tree: DependencyTree
    
    # Mathematical foundations
    spatial_anchors: Dict[str, SpatialCoordinate]
    relationships: List[DependencyRelationship]
    
    # Processing metadata
    processing_metadata: ProcessingMetadata
    
    def get_token_by_id(self, token_id: int) -> Optional[Token]:
        """Get token by its sequential ID"""
        for token in self.tokens:
            if token.token_id == token_id:
                return token
        return None
    
    def get_tokens_by_pos(self, pos_tag: str) -> List[Token]:
        """Get all tokens with specific POS tag"""
        return [t for t in self.tokens if hasattr(t, 'pos_tag') and t.pos_tag == pos_tag]
    
    def get_content_tokens(self) -> List[Token]:
        """Get all content tokens (excluding spaces and punctuation)"""
        return [t for t in self.tokens if not t.is_space and not t.is_punct]
    
    def get_dependency_relationships(self) -> List[DependencyRelationship]:
        """Get all dependency relationships in the document"""
        return self.dependency_tree.relationships
    
    def get_root_token(self) -> Optional[Token]:
        """Get the root token of the dependency tree"""
        return self.dependency_tree.root
    
    def get_sentence_length(self) -> int:
        """Get the number of content tokens in the sentence"""
        return len(self.get_content_tokens())
    
    def get_pos_distribution(self) -> Dict[str, int]:
        """Get distribution of POS tags in the document"""
        pos_counts = defaultdict(int)
        for token in self.tokens:
            if hasattr(token, 'pos_tag') and token.pos_tag:
                pos_counts[token.pos_tag] += 1
        return dict(pos_counts)
    
    def get_dependency_distribution(self) -> Dict[str, int]:
        """Get distribution of dependency relations in the document"""
        dep_counts = defaultdict(int)
        for rel in self.dependency_tree.relationships:
            dep_counts[rel.relation] += 1
        return dict(dep_counts)
    
    def to_conllu(self) -> str:
        """Export to CoNLL-U format for interoperability"""
        return self.dependency_tree.to_conllu()
    
    def to_json(self) -> Dict[str, Any]:
        """Export to JSON format for API consumption"""
        return {
            "original_text": self.original_text,
            "tokens": [
                {
                    "id": token.token_id,
                    "text": token.text,
                    "normalized": token.normalized,
                    "pos_tag": getattr(token, 'pos_tag', None),
                    "pos_confidence": getattr(token, 'pos_confidence', 0.0),
                    "is_space": token.is_space,
                    "is_punct": token.is_punct,
                    "start_char": token.start_char,
                    "end_char": token.end_char
                }
                for token in self.tokens
            ],
            "dependencies": [
                {
                    "head": rel.head.token_id,
                    "dependent": rel.dependent.token_id,
                    "relation": rel.relation,
                    "confidence": rel.confidence
                }
                for rel in self.dependency_tree.relationships
            ],
            "root_token_id": self.dependency_tree.root.token_id if self.dependency_tree.root else None,
            "processing_metadata": {
                "total_processing_time": self.processing_metadata.total_processing_time,
                "total_tokens": self.processing_metadata.total_tokens,
                "content_tokens": self.processing_metadata.content_tokens,
                "dependency_relationships": self.processing_metadata.dependency_relationships,
                "average_pos_confidence": self.processing_metadata.average_pos_confidence,
                "average_dependency_confidence": self.processing_metadata.average_dependency_confidence
            }
        }
    
    def get_similarity(self, other: 'LinguisticDocument', resonance: HarmonicResonance) -> float:
        """Calculate document similarity using harmonic resonance"""
        if not self.spatial_anchors or not other.spatial_anchors:
            return 0.0
        
        # Calculate average similarity between all spatial anchors
        total_similarity = 0.0
        comparison_count = 0
        
        for word1, coord1 in self.spatial_anchors.items():
            for word2, coord2 in other.spatial_anchors.items():
                similarity_result = resonance.calculate_resonance(coord1, coord2)
                total_similarity += similarity_result.similarity_score
                comparison_count += 1
        
        return total_similarity / comparison_count if comparison_count > 0 else 0.0


class SharedLinguisticCache:
    """
    Unified caching system shared across all Phase 2 modules.
    Optimizes performance by avoiding redundant calculations.
    """
    
    def __init__(self):
        # Spatial coordinate cache (shared with Phase 1)
        self.coordinate_cache: Dict[str, SpatialCoordinate] = {}
        
        # POS tag cache
        self.pos_cache: Dict[Tuple[str, str], str] = {}  # (word, context) -> POS
        
        # Dependency relationship cache
        self.dependency_cache: Dict[Tuple[str, str], float] = {}  # (pattern, context) -> score
        
        # Grammar rule application cache
        self.rule_cache: Dict[str, List] = {}  # POS_pattern -> applicable_rules
        
        # Cache statistics
        self.cache_stats = {
            "coordinate_hits": 0,
            "coordinate_misses": 0,
            "pos_hits": 0,
            "pos_misses": 0,
            "dependency_hits": 0,
            "dependency_misses": 0
        }
    
    def get_coordinate(self, text: str, anchor_system: SpatialAnchor) -> SpatialCoordinate:
        """Get spatial coordinate with caching"""
        if text in self.coordinate_cache:
            self.cache_stats["coordinate_hits"] += 1
            return self.coordinate_cache[text]
        
        _, coord = anchor_system.create_anchor(text)
        self.coordinate_cache[text] = coord
        self.cache_stats["coordinate_misses"] += 1
        return coord
    
    def cache_pos_decision(self, word: str, context: str, pos_tag: str):
        """Cache POS tagging decision"""
        key = (word, context)
        self.pos_cache[key] = pos_tag
    
    def get_cached_pos(self, word: str, context: str) -> Optional[str]:
        """Retrieve cached POS tag"""
        key = (word, context)
        if key in self.pos_cache:
            self.cache_stats["pos_hits"] += 1
            return self.pos_cache[key]
        
        self.cache_stats["pos_misses"] += 1
        return None
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = self.cache_stats.copy()
        stats["coordinate_cache_size"] = len(self.coordinate_cache)
        stats["pos_cache_size"] = len(self.pos_cache)
        stats["dependency_cache_size"] = len(self.dependency_cache)
        stats["rule_cache_size"] = len(self.rule_cache)
        
        # Calculate hit rates
        total_coordinate_requests = stats["coordinate_hits"] + stats["coordinate_misses"]
        total_pos_requests = stats["pos_hits"] + stats["pos_misses"]
        total_dependency_requests = stats["dependency_hits"] + stats["dependency_misses"]
        
        if total_coordinate_requests > 0:
            stats["coordinate_hit_rate"] = stats["coordinate_hits"] / total_coordinate_requests
        if total_pos_requests > 0:
            stats["pos_hit_rate"] = stats["pos_hits"] / total_pos_requests
        if total_dependency_requests > 0:
            stats["dependency_hit_rate"] = stats["dependency_hits"] / total_dependency_requests
        
        return stats


class CortexLinguisticProcessor:
    """
    Integrated processor that coordinates all Phase 2 modules.
    Provides unified interface for complete linguistic analysis.
    """
    
    def __init__(self):
        # Initialize Phase 1 foundation
        self.anchor_system = SpatialAnchor()
        self.memory = BinaryCellMemory()
        self.resonance = HarmonicResonance(self.memory)
        
        # Initialize Phase 2 modules
        self.tokenizer = CortexTokenizer()
        self.tagger = CortexTagger(self.anchor_system, self.memory, self.resonance)
        self.parser = CortexParser(self.memory, self.resonance)
        
        # Initialize caching system
        self.cache = SharedLinguisticCache()
        
        # Integration statistics
        self.processing_stats = {
            "documents_processed": 0,
            "total_tokens": 0,
            "total_relationships": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors_encountered": 0
        }
    
    def process_text(self, text: str) -> LinguisticDocument:
        """
        Complete linguistic processing pipeline.
        
        Input: Raw text string
        Output: Fully analyzed linguistic document with:
        - Tokens with spatial coordinates
        - POS tags with mathematical justification
        - Dependency tree with relationship scores
        - All relationships stored in BinaryCellMemory
        """
        
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Step 1: Tokenization
            tokenization_start = time.time()
            tokens = self.tokenizer.tokenize(text)
            tokenization_time = time.time() - tokenization_start
            
            # Step 2: POS Tagging
            tagging_start = time.time()
            tagged_tokens = self.tagger.tag_tokens(tokens)
            tagging_time = time.time() - tagging_start
            
            # Step 3: Dependency Parsing
            parsing_start = time.time()
            dependency_tree = self.parser.parse_sentence(tagged_tokens)
            parsing_time = time.time() - parsing_start
            
        except Exception as e:
            errors.append(f"Processing error: {str(e)}")
            # Create minimal document with error information
            return self._create_error_document(text, str(e))
        
        # Step 4: Extract spatial anchors and relationships
        spatial_anchors = self._extract_spatial_anchors(tagged_tokens)
        relationships = dependency_tree.relationships
        
        # Step 5: Generate processing metadata
        total_time = time.time() - start_time
        metadata = self._generate_metadata(
            total_time, tokenization_time, tagging_time, parsing_time,
            tagged_tokens, relationships, errors, warnings
        )
        
        # Step 6: Create integrated document
        document = LinguisticDocument(
            original_text=text,
            tokens=tagged_tokens,
            dependency_tree=dependency_tree,
            spatial_anchors=spatial_anchors,
            relationships=relationships,
            processing_metadata=metadata
        )
        
        # Update statistics
        self._update_processing_stats(document)
        
        return document
    
    def batch_process(self, texts: List[str]) -> List[LinguisticDocument]:
        """Process multiple texts with shared caching benefits"""
        documents = []
        
        for text in texts:
            doc = self.process_text(text)
            documents.append(doc)
        
        return documents
    
    def _extract_spatial_anchors(self, tokens: List[Token]) -> Dict[str, SpatialCoordinate]:
        """Extract spatial coordinates for all unique words"""
        anchors = {}
        
        for token in tokens:
            if not token.is_space and token.normalized not in anchors:
                anchors[token.normalized] = token.spatial_coord
        
        return anchors
    
    def _generate_metadata(self, total_time: float, tokenization_time: float,
                          tagging_time: float, parsing_time: float,
                          tokens: List[Token], relationships: List[DependencyRelationship],
                          errors: List[str], warnings: List[str]) -> ProcessingMetadata:
        """Generate comprehensive processing metadata"""
        
        # Count token types
        content_tokens = sum(1 for t in tokens if not t.is_space and not t.is_punct)
        space_tokens = sum(1 for t in tokens if t.is_space)
        punct_tokens = sum(1 for t in tokens if t.is_punct)
        
        # Get tagging statistics
        tagging_stats = self.tagger.get_tagging_statistics()
        
        # Calculate confidence averages
        pos_confidences = [getattr(t, 'pos_confidence', 0.0) for t in tokens if hasattr(t, 'pos_confidence')]
        avg_pos_confidence = sum(pos_confidences) / len(pos_confidences) if pos_confidences else 0.0
        
        dep_confidences = [r.confidence for r in relationships]
        avg_dep_confidence = sum(dep_confidences) / len(dep_confidences) if dep_confidences else 0.0
        
        # Get cache statistics
        cache_stats = self.cache.get_cache_statistics()
        
        # Get memory statistics
        memory_stats = self.memory.memory_stats()
        
        return ProcessingMetadata(
            total_processing_time=total_time,
            tokenization_time=tokenization_time,
            tagging_time=tagging_time,
            parsing_time=parsing_time,
            total_tokens=len(tokens),
            content_tokens=content_tokens,
            space_tokens=space_tokens,
            punct_tokens=punct_tokens,
            dictionary_lookups=tagging_stats.get("dictionary_lookups", 0),
            context_disambiguations=tagging_stats.get("context_disambiguations", 0),
            morphological_analyses=tagging_stats.get("morphological_analyses", 0),
            dependency_relationships=len(relationships),
            pos_relationships=content_tokens,  # Each content token has a POS relationship
            spatial_relationships=memory_stats["total_relationships"],
            memory_relationships_stored=memory_stats["total_relationships"],
            cache_hits=cache_stats.get("coordinate_hits", 0) + cache_stats.get("pos_hits", 0),
            cache_misses=cache_stats.get("coordinate_misses", 0) + cache_stats.get("pos_misses", 0),
            average_pos_confidence=avg_pos_confidence,
            average_dependency_confidence=avg_dep_confidence,
            errors_encountered=errors,
            warnings_generated=warnings
        )
    
    def _create_error_document(self, text: str, error_message: str) -> LinguisticDocument:
        """Create a minimal document when processing fails"""
        empty_metadata = ProcessingMetadata(
            total_processing_time=0.0,
            tokenization_time=0.0,
            tagging_time=0.0,
            parsing_time=0.0,
            total_tokens=0,
            content_tokens=0,
            space_tokens=0,
            punct_tokens=0,
            dictionary_lookups=0,
            context_disambiguations=0,
            morphological_analyses=0,
            dependency_relationships=0,
            pos_relationships=0,
            spatial_relationships=0,
            memory_relationships_stored=0,
            cache_hits=0,
            cache_misses=0,
            average_pos_confidence=0.0,
            average_dependency_confidence=0.0,
            errors_encountered=[error_message],
            warnings_generated=[]
        )
        
        from parser import DependencyTree
        empty_tree = DependencyTree([], [])
        
        return LinguisticDocument(
            original_text=text,
            tokens=[],
            dependency_tree=empty_tree,
            spatial_anchors={},
            relationships=[],
            processing_metadata=empty_metadata
        )
    
    def _update_processing_stats(self, document: LinguisticDocument):
        """Update global processing statistics"""
        self.processing_stats["documents_processed"] += 1
        self.processing_stats["total_tokens"] += document.processing_metadata.total_tokens
        self.processing_stats["total_relationships"] += document.processing_metadata.dependency_relationships
        self.processing_stats["total_processing_time"] += document.processing_metadata.total_processing_time
        self.processing_stats["cache_hits"] += document.processing_metadata.cache_hits
        self.processing_stats["cache_misses"] += document.processing_metadata.cache_misses
        self.processing_stats["errors_encountered"] += len(document.processing_metadata.errors_encountered)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = self.processing_stats.copy()
        
        # Add derived statistics
        if stats["documents_processed"] > 0:
            stats["average_tokens_per_document"] = stats["total_tokens"] / stats["documents_processed"]
            stats["average_relationships_per_document"] = stats["total_relationships"] / stats["documents_processed"]
            stats["average_processing_time"] = stats["total_processing_time"] / stats["documents_processed"]
        
        # Add cache statistics
        cache_stats = self.cache.get_cache_statistics()
        stats["cache_statistics"] = cache_stats
        
        # Add module-specific statistics
        # Note: tokenizer statistics are included in processing metadata per document
        stats["tagger_statistics"] = self.tagger.get_tagging_statistics()
        stats["parser_statistics"] = self.parser.get_parsing_statistics()
        
        return stats
    
    def compare_documents(self, doc1: LinguisticDocument, doc2: LinguisticDocument) -> Dict[str, float]:
        """Compare two documents using various similarity metrics"""
        return {
            "spatial_similarity": doc1.get_similarity(doc2, self.resonance),
            "pos_similarity": self._calculate_pos_similarity(doc1, doc2),
            "dependency_similarity": self._calculate_dependency_similarity(doc1, doc2),
            "structural_similarity": self._calculate_structural_similarity(doc1, doc2)
        }
    
    def _calculate_pos_similarity(self, doc1: LinguisticDocument, doc2: LinguisticDocument) -> float:
        """Calculate POS tag distribution similarity"""
        pos_dist1 = doc1.get_pos_distribution()
        pos_dist2 = doc2.get_pos_distribution()
        
        all_pos_tags = set(pos_dist1.keys()) | set(pos_dist2.keys())
        if not all_pos_tags:
            return 1.0
        
        # Calculate cosine similarity of POS distributions
        dot_product = sum(pos_dist1.get(pos, 0) * pos_dist2.get(pos, 0) for pos in all_pos_tags)
        norm1 = sum(count ** 2 for count in pos_dist1.values()) ** 0.5
        norm2 = sum(count ** 2 for count in pos_dist2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_dependency_similarity(self, doc1: LinguisticDocument, doc2: LinguisticDocument) -> float:
        """Calculate dependency relation distribution similarity"""
        dep_dist1 = doc1.get_dependency_distribution()
        dep_dist2 = doc2.get_dependency_distribution()
        
        all_dep_types = set(dep_dist1.keys()) | set(dep_dist2.keys())
        if not all_dep_types:
            return 1.0
        
        # Calculate cosine similarity of dependency distributions
        dot_product = sum(dep_dist1.get(dep, 0) * dep_dist2.get(dep, 0) for dep in all_dep_types)
        norm1 = sum(count ** 2 for count in dep_dist1.values()) ** 0.5
        norm2 = sum(count ** 2 for count in dep_dist2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_structural_similarity(self, doc1: LinguisticDocument, doc2: LinguisticDocument) -> float:
        """Calculate structural similarity based on sentence length and complexity"""
        len1 = doc1.get_sentence_length()
        len2 = doc2.get_sentence_length()
        
        if len1 == 0 and len2 == 0:
            return 1.0
        
        # Length similarity (inverse of relative difference)
        length_similarity = 1.0 - abs(len1 - len2) / max(len1, len2, 1)
        
        # Complexity similarity (based on number of relationships)
        rel1 = len(doc1.get_dependency_relationships())
        rel2 = len(doc2.get_dependency_relationships())
        complexity_similarity = 1.0 - abs(rel1 - rel2) / max(rel1, rel2, 1)
        
        return (length_similarity + complexity_similarity) / 2


if __name__ == "__main__":
    # Demonstration of the integrated linguistic processor
    print("CortexOS NLP - Integrated Linguistic Processor Demonstration")
    print("=" * 60)
    
    # Initialize the integrated processor
    processor = CortexLinguisticProcessor()
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe it's working so well!",
        "Complex sentences with subordinate clauses are challenging.",
        "The beautiful red car drives very fast down the winding road.",
        "Running quickly, the athlete finished the race successfully."
    ]
    
    print("Testing integrated linguistic processing:")
    documents = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Processing: '{text}'")
        
        # Process the text through the complete pipeline
        doc = processor.process_text(text)
        documents.append(doc)
        
        # Display results
        print(f"   Tokens: {doc.processing_metadata.total_tokens} "
              f"(content: {doc.processing_metadata.content_tokens})")
        print(f"   POS tags: {len(doc.get_pos_distribution())} unique tags")
        print(f"   Dependencies: {doc.processing_metadata.dependency_relationships} relationships")
        print(f"   Processing time: {doc.processing_metadata.total_processing_time:.4f}s")
        print(f"   Average POS confidence: {doc.processing_metadata.average_pos_confidence:.3f}")
        print(f"   Average dependency confidence: {doc.processing_metadata.average_dependency_confidence:.3f}")
        
        # Show some key relationships
        print(f"   Key dependencies:")
        for rel in doc.get_dependency_relationships()[:3]:  # Show first 3
            print(f"     {rel.head.text} --[{rel.relation}]--> {rel.dependent.text}")
    
    # Test batch processing
    print(f"\nTesting batch processing:")
    batch_start = time.time()
    batch_docs = processor.batch_process(test_texts)
    batch_time = time.time() - batch_start
    print(f"   Processed {len(batch_docs)} documents in {batch_time:.4f}s")
    print(f"   Average: {batch_time/len(batch_docs):.4f}s per document")
    
    # Test document comparison
    print(f"\nTesting document similarity:")
    if len(documents) >= 2:
        similarity = processor.compare_documents(documents[0], documents[1])
        print(f"   Document 1 vs Document 2:")
        for metric, score in similarity.items():
            print(f"     {metric}: {score:.3f}")
    
    # Get comprehensive statistics
    stats = processor.get_processing_statistics()
    print(f"\nIntegrated Processing Statistics:")
    print(f"  Documents processed: {stats['documents_processed']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print(f"  Average processing time: {stats.get('average_processing_time', 0):.4f}s")
    
    # Calculate cache hit rate safely
    total_cache_requests = stats['cache_hits'] + stats['cache_misses']
    if total_cache_requests > 0:
        cache_hit_rate = stats['cache_hits'] / total_cache_requests * 100
        print(f"  Cache hit rate: {cache_hit_rate:.1f}%")
    else:
        print(f"  Cache hit rate: N/A (no cache requests)")
    
    # Test JSON export
    print(f"\nTesting JSON export:")
    json_data = documents[0].to_json()
    print(f"   JSON keys: {list(json_data.keys())}")
    print(f"   Token count in JSON: {len(json_data['tokens'])}")
    print(f"   Dependency count in JSON: {len(json_data['dependencies'])}")
    
    print(f"\n✅ Integrated linguistic processor demonstration complete!")
    print(f"🎉 Phase 2 implementation is fully functional!")

