# CortexOS NLP - Phase 2 Integration Specifications

## Complete Linguistic Layer Architecture

**Mission:** Seamlessly integrate Tokenizer, Tagger, and Parser modules to create a unified deterministic linguistic processing pipeline that transforms raw text into mathematically-anchored linguistic structures.

---

## Integration Data Flow

### **Complete Pipeline Architecture**

```
Raw Text Input
      ↓
┌─────────────────┐
│  CortexTokenizer │ → Deterministic text segmentation
└─────────────────┘
      ↓
[Token Objects with Spatial Coordinates]
      ↓
┌─────────────────┐
│   CortexTagger  │ → Deterministic POS tagging
└─────────────────┘
      ↓
[Token Objects with POS Tags + Spatial Coordinates]
      ↓
┌─────────────────┐
│  CortexParser   │ → Deterministic dependency parsing
└─────────────────┘
      ↓
[Complete Linguistic Document with Dependency Tree]
      ↓
Ready for Phase 3 API Layer
```

### **Shared Mathematical Foundation**

All Phase 2 modules operate on the same mathematical infrastructure:

1. **SpatialAnchor System:** Every linguistic element has 6D coordinates
2. **BinaryCellMemory:** All relationships stored permanently  
3. **HarmonicResonance:** Mathematical similarity calculations
4. **Unified Caching:** Shared coordinate and relationship caches

---

## Integrated Module Implementation

### **1. Unified Linguistic Processor**

```python
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
        
        # Integration statistics
        self.processing_stats = {
            "documents_processed": 0,
            "total_tokens": 0,
            "total_relationships": 0,
            "cache_hits": 0,
            "cache_misses": 0
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
        
        # Step 1: Tokenization
        tokens = self.tokenizer.tokenize(text)
        
        # Step 2: POS Tagging
        tagged_tokens = self.tagger.tag_tokens(tokens)
        
        # Step 3: Dependency Parsing
        dependency_tree = self.parser.parse_sentence(tagged_tokens)
        
        # Step 4: Create integrated document
        document = LinguisticDocument(
            original_text=text,
            tokens=tagged_tokens,
            dependency_tree=dependency_tree,
            spatial_anchors=self._extract_spatial_anchors(tagged_tokens),
            relationships=self._extract_relationships(dependency_tree),
            processing_metadata=self._generate_metadata()
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
```

### **2. Unified Document Structure**

```python
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
    relationships: List[Relationship]
    
    # Processing metadata
    processing_metadata: Dict[str, Any]
    
    def get_token_by_id(self, token_id: int) -> Optional[Token]:
        """Get token by its sequential ID"""
        for token in self.tokens:
            if token.token_id == token_id:
                return token
        return None
    
    def get_tokens_by_pos(self, pos_tag: str) -> List[Token]:
        """Get all tokens with specific POS tag"""
        return [t for t in self.tokens if t.pos_tag == pos_tag]
    
    def get_dependency_relationships(self) -> List[DependencyRelationship]:
        """Get all dependency relationships in the document"""
        return self.dependency_tree.relationships
    
    def get_root_token(self) -> Token:
        """Get the root token of the dependency tree"""
        return self.dependency_tree.root
    
    def to_conllu(self) -> str:
        """Export to CoNLL-U format for interoperability"""
        lines = []
        
        for token in self.tokens:
            if token.is_space:
                continue
            
            # Find head token ID
            head_id = 0  # Root
            dep_label = "root"
            
            for rel in self.dependency_tree.relationships:
                if rel.dependent == token:
                    head_id = rel.head.token_id + 1  # CoNLL-U uses 1-based indexing
                    dep_label = rel.relation
                    break
            
            line = f"{token.token_id + 1}\t{token.text}\t{token.normalized}\t{token.pos_tag}\t{token.pos_tag}\t_\t{head_id}\t{dep_label}\t_\t_"
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_similarity(self, other: 'LinguisticDocument') -> float:
        """Calculate document similarity using harmonic resonance"""
        # This will be implemented using the HarmonicResonance module
        # to compare document-level spatial coordinates
        pass
```

### **3. Cross-Module Data Sharing**

```python
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
        self.rule_cache: Dict[str, List[GrammarRule]] = {}  # POS_pattern -> applicable_rules
        
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
```

---

## Integration Testing Framework

### **1. End-to-End Pipeline Tests**

```python
class Phase2IntegrationTests:
    """
    Comprehensive testing framework for Phase 2 integration.
    Validates that all modules work together correctly.
    """
    
    def __init__(self):
        self.processor = CortexLinguisticProcessor()
        self.test_cases = self._load_test_cases()
    
    def test_complete_pipeline(self):
        """Test the complete tokenization -> tagging -> parsing pipeline"""
        
        test_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "I can't believe it's working so well!",
            "Complex sentences with subordinate clauses are challenging.",
            "Numbers like 123 and symbols @#$ should be handled correctly.",
        ]
        
        for sentence in test_sentences:
            # Process sentence
            doc = self.processor.process_text(sentence)
            
            # Validate tokenization
            self._validate_tokenization(sentence, doc.tokens)
            
            # Validate POS tagging
            self._validate_pos_tagging(doc.tokens)
            
            # Validate dependency parsing
            self._validate_dependency_parsing(doc.dependency_tree)
            
            # Validate integration consistency
            self._validate_integration_consistency(doc)
    
    def test_cross_module_consistency(self):
        """Test that modules produce consistent results across runs"""
        
        test_text = "The cat sat on the mat."
        
        # Process same text multiple times
        results = []
        for _ in range(5):
            doc = self.processor.process_text(test_text)
            results.append(doc)
        
        # Verify all results are identical
        base_result = results[0]
        for result in results[1:]:
            assert self._documents_identical(base_result, result)
    
    def test_memory_integration(self):
        """Test that BinaryCellMemory correctly stores all relationships"""
        
        # Process a sentence
        doc = self.processor.process_text("The red car drives fast.")
        
        # Verify all expected relationships are stored
        stored_relationships = self.processor.memory.get_all_relationships()
        
        # Should include:
        # - Word-POS relationships from tagger
        # - Dependency relationships from parser
        # - Spatial coordinate relationships
        
        expected_relationship_count = (
            len(doc.tokens) +  # Word-POS relationships
            len(doc.dependency_tree.relationships)  # Dependency relationships
        )
        
        assert len(stored_relationships) >= expected_relationship_count
    
    def _validate_integration_consistency(self, doc: LinguisticDocument):
        """Validate that all module outputs are consistent with each other"""
        
        # Check that all tokens have POS tags
        for token in doc.tokens:
            if not token.is_space and not token.is_punct:
                assert token.pos_tag is not None
        
        # Check that dependency tree uses same tokens
        tree_tokens = set()
        for rel in doc.dependency_tree.relationships:
            tree_tokens.add(rel.head)
            tree_tokens.add(rel.dependent)
        
        content_tokens = set(t for t in doc.tokens if not t.is_space)
        assert tree_tokens.issubset(content_tokens)
        
        # Check that spatial coordinates are consistent
        for token in doc.tokens:
            if token.normalized in doc.spatial_anchors:
                assert doc.spatial_anchors[token.normalized] == token.spatial_coord
```

### **2. Performance Benchmarking**

```python
class Phase2PerformanceBenchmark:
    """
    Performance benchmarking for the integrated Phase 2 pipeline.
    Measures speed, memory usage, and scalability.
    """
    
    def __init__(self):
        self.processor = CortexLinguisticProcessor()
    
    def benchmark_processing_speed(self):
        """Benchmark processing speed across different text lengths"""
        
        test_texts = [
            "Short sentence.",  # ~2 tokens
            "This is a medium length sentence with several words.",  # ~10 tokens
            "This is a much longer sentence that contains multiple clauses, complex grammatical structures, and various types of words including numbers like 123 and punctuation marks.",  # ~25 tokens
            # Add progressively longer texts up to paragraph length
        ]
        
        results = {}
        
        for text in test_texts:
            token_count = len(text.split())
            
            # Time the processing
            start_time = time.time()
            doc = self.processor.process_text(text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            tokens_per_second = len(doc.tokens) / processing_time
            
            results[token_count] = {
                "processing_time": processing_time,
                "tokens_per_second": tokens_per_second,
                "memory_usage": self._measure_memory_usage()
            }
        
        return results
    
    def benchmark_cache_effectiveness(self):
        """Measure cache hit rates and performance improvements"""
        
        # Process same text multiple times to test caching
        test_text = "The quick brown fox jumps over the lazy dog."
        
        cache_stats_before = self.processor.tokenizer.anchor_system.cache_size()
        
        # First processing (cache misses expected)
        start_time = time.time()
        doc1 = self.processor.process_text(test_text)
        first_time = time.time() - start_time
        
        # Second processing (cache hits expected)
        start_time = time.time()
        doc2 = self.processor.process_text(test_text)
        second_time = time.time() - start_time
        
        cache_stats_after = self.processor.tokenizer.anchor_system.cache_size()
        
        return {
            "first_processing_time": first_time,
            "second_processing_time": second_time,
            "speedup_ratio": first_time / second_time,
            "cache_growth": cache_stats_after - cache_stats_before
        }
```

---

## Error Handling and Robustness

### **1. Graceful Degradation Strategy**

```python
class RobustLinguisticProcessor(CortexLinguisticProcessor):
    """
    Enhanced processor with comprehensive error handling and graceful degradation.
    Ensures system continues to function even with problematic input.
    """
    
    def process_text_robust(self, text: str) -> LinguisticDocument:
        """Process text with comprehensive error handling"""
        
        try:
            # Attempt normal processing
            return self.process_text(text)
        
        except TokenizationError as e:
            # Handle tokenization failures
            return self._handle_tokenization_error(text, e)
        
        except POSTaggingError as e:
            # Handle POS tagging failures
            return self._handle_pos_error(text, e)
        
        except ParsingError as e:
            # Handle parsing failures
            return self._handle_parsing_error(text, e)
        
        except Exception as e:
            # Handle unexpected errors
            return self._handle_unexpected_error(text, e)
    
    def _handle_tokenization_error(self, text: str, error: TokenizationError) -> LinguisticDocument:
        """Fallback tokenization strategy"""
        # Use simple whitespace tokenization as fallback
        simple_tokens = self._simple_tokenize(text)
        
        # Create minimal document with basic tokens
        return LinguisticDocument(
            original_text=text,
            tokens=simple_tokens,
            dependency_tree=DependencyTree(simple_tokens, []),
            spatial_anchors={},
            relationships=[],
            processing_metadata={"error": str(error), "fallback": "simple_tokenization"}
        )
```

### **2. Input Validation and Sanitization**

```python
class InputValidator:
    """
    Validates and sanitizes input text before processing.
    Handles edge cases and problematic input gracefully.
    """
    
    def validate_and_clean(self, text: str) -> str:
        """Validate input text and clean if necessary"""
        
        if not text:
            raise ValueError("Empty text input")
        
        if len(text) > 1000000:  # 1MB limit
            raise ValueError("Text too long for processing")
        
        # Remove or replace problematic characters
        cleaned_text = self._clean_text(text)
        
        # Validate character encoding
        try:
            cleaned_text.encode('utf-8')
        except UnicodeEncodeError:
            cleaned_text = self._fix_encoding(cleaned_text)
        
        return cleaned_text
    
    def _clean_text(self, text: str) -> str:
        """Clean problematic characters from text"""
        # Remove null bytes and other control characters
        cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
```

---

## Configuration and Customization

### **1. Configurable Processing Pipeline**

```python
class CortexNLPConfig:
    """
    Configuration system for customizing Phase 2 processing behavior.
    Allows fine-tuning of each module's behavior.
    """
    
    def __init__(self):
        # Tokenizer configuration
        self.tokenizer_config = {
            "preserve_case": False,
            "handle_contractions": True,
            "split_hyphenated": False,
            "custom_patterns": []
        }
        
        # Tagger configuration
        self.tagger_config = {
            "context_window": 3,
            "use_morphology": True,
            "pos_dictionary_path": None,
            "confidence_threshold": 0.5
        }
        
        # Parser configuration
        self.parser_config = {
            "grammar_rules_path": None,
            "max_sentence_length": 100,
            "conflict_resolution_strategy": "mathematical_scoring",
            "enable_rule_learning": True
        }
        
        # Integration configuration
        self.integration_config = {
            "enable_caching": True,
            "cache_size_limit": 10000,
            "enable_statistics": True,
            "parallel_processing": False
        }
    
    def create_processor(self) -> CortexLinguisticProcessor:
        """Create configured processor instance"""
        processor = CortexLinguisticProcessor()
        
        # Apply tokenizer configuration
        processor.tokenizer = CortexTokenizer(**self.tokenizer_config)
        
        # Apply tagger configuration  
        processor.tagger.configure(**self.tagger_config)
        
        # Apply parser configuration
        processor.parser.configure(**self.parser_config)
        
        return processor
```

---

## Phase 2 Completion Criteria

### **Success Metrics**

1. **Functional Integration**
   - ✅ All modules work together seamlessly
   - ✅ Data flows correctly between components
   - ✅ Shared mathematical foundation utilized

2. **Performance Targets**
   - ✅ Processing speed: >1000 tokens/second
   - ✅ Memory efficiency: <1MB per 1000 tokens
   - ✅ Cache hit rate: >80% for repeated processing

3. **Accuracy Standards**
   - ✅ Tokenization: 100% consistency across runs
   - ✅ POS tagging: >95% accuracy on standard corpora
   - ✅ Dependency parsing: >90% accuracy on standard corpora

4. **Robustness Requirements**
   - ✅ Graceful handling of edge cases
   - ✅ Error recovery mechanisms
   - ✅ Input validation and sanitization

### **Ready for Phase 3**

Upon completion of Phase 2 integration, the system will be ready for Phase 3 (API Layer) development, which will provide the spaCy-compatible interface that makes the deterministic NLP engine accessible to end users.

**Phase 2 delivers the complete linguistic processing foundation with mathematical certainty, perfect traceability, and consistent reproducibility.**

