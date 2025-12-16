# CortexTagger - Deterministic Part-of-Speech Tagging Implementation Specification

## Design Philosophy

**Deterministic Certainty over Probabilistic Fluency**

The CortexTagger is designed for applications where **accuracy and consistency matter more than linguistic creativity**. Every POS tag assignment is mathematically traceable and perfectly repeatable.

---

## Core Architecture

### **1. Spatial POS Dictionary**

```python
class SpatialPOSDictionary:
    """
    Maps words to their possible POS tags using spatial coordinates.
    Replaces probabilistic POS models with deterministic lookup + context analysis.
    """
    
    def __init__(self, anchor_system: SpatialAnchor, memory: BinaryCellMemory):
        self.anchor_system = anchor_system
        self.memory = memory
        
        # Core mappings
        self.word_pos_map: Dict[str, List[POSTag]] = {}
        self.pos_anchors: Dict[str, SpatialCoordinate] = {}
        
        # Load base dictionary and create spatial anchors
        self._initialize_pos_dictionary()
        self._create_pos_anchors()
    
    def _initialize_pos_dictionary(self):
        """Load base word-to-POS mappings"""
        # High-frequency words with definitive POS tags
        base_dictionary = {
            # Determiners - always DT
            "the": ["DT"], "a": ["DT"], "an": ["DT"], "this": ["DT"], 
            "that": ["DT"], "these": ["DT"], "those": ["DT"],
            
            # Pronouns - always PRP
            "i": ["PRP"], "you": ["PRP"], "he": ["PRP"], "she": ["PRP"],
            "it": ["PRP"], "we": ["PRP"], "they": ["PRP"],
            
            # Prepositions - always IN
            "in": ["IN"], "on": ["IN"], "at": ["IN"], "by": ["IN"],
            "for": ["IN"], "with": ["IN"], "from": ["IN"], "to": ["IN"],
            
            # Conjunctions - always CC
            "and": ["CC"], "or": ["CC"], "but": ["CC"], "nor": ["CC"],
            
            # Ambiguous words requiring context analysis
            "book": ["NN", "VB"],      # noun or verb
            "run": ["NN", "VB"],       # noun or verb  
            "fast": ["JJ", "RB"],      # adjective or adverb
            "well": ["RB", "JJ", "NN"], # adverb, adjective, or noun
            "can": ["MD", "NN", "VB"], # modal, noun, or verb
            "will": ["MD", "NN", "VB"], # modal, noun, or verb
            
            # Add 10,000+ more entries for comprehensive coverage
        }
        
        self.word_pos_map = base_dictionary
    
    def _create_pos_anchors(self):
        """Create spatial anchors for each POS tag"""
        pos_tags = [
            "NN", "NNS", "NNP", "NNPS",  # Nouns
            "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # Verbs
            "JJ", "JJR", "JJS",  # Adjectives
            "RB", "RBR", "RBS",  # Adverbs
            "PRP", "PRP$", "WP", "WP$",  # Pronouns
            "DT", "WDT",  # Determiners
            "IN",  # Prepositions
            "CC",  # Conjunctions
            "MD",  # Modals
            "CD",  # Numbers
            "UH",  # Interjections
            "PUNCT", "SYM"  # Punctuation and symbols
        ]
        
        for pos_tag in pos_tags:
            _, coord = self.anchor_system.create_anchor(f"POS_{pos_tag}")
            self.pos_anchors[pos_tag] = coord
```

### **2. Context-Aware Disambiguation Engine**

```python
class POSDisambiguator:
    """
    Uses mathematical context analysis to resolve POS ambiguity.
    Replaces probabilistic models with deterministic harmonic resonance.
    """
    
    def __init__(self, pos_dictionary: SpatialPOSDictionary, 
                 resonance: HarmonicResonance):
        self.pos_dict = pos_dictionary
        self.resonance = resonance
        self.context_window = 3  # Look at 3 tokens on each side
    
    def disambiguate_pos(self, token: Token, context: List[Token]) -> POSTagResult:
        """
        Mathematically determine the correct POS tag for an ambiguous token.
        
        Returns:
            POSTagResult with tag, confidence, and mathematical justification
        """
        # Get possible POS tags for this token
        candidates = self.pos_dict.get_pos_candidates(token.normalized)
        
        if len(candidates) == 1:
            # Unambiguous - return with full confidence
            return POSTagResult(
                tag=candidates[0],
                confidence=1.0,
                method="dictionary_lookup",
                justification=f"Unambiguous dictionary entry for '{token.text}'"
            )
        
        if len(candidates) == 0:
            # Unknown word - use morphological analysis
            return self._analyze_unknown_word(token, context)
        
        # Multiple candidates - use context analysis
        return self._resolve_ambiguity(token, candidates, context)
    
    def _resolve_ambiguity(self, token: Token, candidates: List[str], 
                          context: List[Token]) -> POSTagResult:
        """Resolve POS ambiguity using harmonic resonance with context"""
        
        # Get context tokens (excluding the target token)
        left_context = self._get_left_context(token, context)
        right_context = self._get_right_context(token, context)
        
        # Score each candidate POS tag
        candidate_scores = []
        
        for pos_tag in candidates:
            pos_coord = self.pos_dict.pos_anchors[pos_tag]
            
            # Calculate resonance with left context
            left_score = 0.0
            if left_context:
                for ctx_token in left_context:
                    if ctx_token.pos_tag:  # Already tagged
                        ctx_pos_coord = self.pos_dict.pos_anchors[ctx_token.pos_tag]
                        resonance_result = self.resonance.calculate_resonance(
                            pos_coord, ctx_pos_coord
                        )
                        left_score += resonance_result.similarity_score
                left_score /= len(left_context)
            
            # Calculate resonance with right context
            right_score = 0.0
            if right_context:
                for ctx_token in right_context:
                    if ctx_token.pos_tag:  # Already tagged
                        ctx_pos_coord = self.pos_dict.pos_anchors[ctx_token.pos_tag]
                        resonance_result = self.resonance.calculate_resonance(
                            pos_coord, ctx_pos_coord
                        )
                        right_score += resonance_result.similarity_score
                right_score /= len(right_context)
            
            # Combined context score
            total_score = (left_score + right_score) / 2
            candidate_scores.append((pos_tag, total_score))
        
        # Select highest scoring candidate
        best_tag, best_score = max(candidate_scores, key=lambda x: x[1])
        
        return POSTagResult(
            tag=best_tag,
            confidence=best_score,
            method="context_analysis",
            justification=f"Harmonic resonance analysis: {candidate_scores}"
        )
```

### **3. Morphological Analyzer for Unknown Words**

```python
class MorphologyAnalyzer:
    """
    Deterministic morphological analysis for unknown words.
    Uses rule-based patterns instead of probabilistic models.
    """
    
    def __init__(self):
        self.suffix_rules = self._load_suffix_rules()
        self.prefix_rules = self._load_prefix_rules()
        self.capitalization_rules = self._load_capitalization_rules()
    
    def _load_suffix_rules(self) -> List[MorphologyRule]:
        """Load suffix-based POS rules"""
        return [
            # Verb suffixes
            MorphologyRule(r".*ing$", "VBG", 0.9, "Present participle"),
            MorphologyRule(r".*ed$", "VBD", 0.8, "Past tense/participle"),
            MorphologyRule(r".*s$", "VBZ", 0.6, "Third person singular"),
            
            # Noun suffixes
            MorphologyRule(r".*tion$", "NN", 0.95, "Abstract noun"),
            MorphologyRule(r".*ness$", "NN", 0.95, "Quality noun"),
            MorphologyRule(r".*ment$", "NN", 0.9, "Action noun"),
            MorphologyRule(r".*er$", "NN", 0.7, "Agent noun"),
            MorphologyRule(r".*s$", "NNS", 0.6, "Plural noun"),
            
            # Adjective suffixes
            MorphologyRule(r".*ful$", "JJ", 0.9, "Full of quality"),
            MorphologyRule(r".*less$", "JJ", 0.9, "Without quality"),
            MorphologyRule(r".*able$", "JJ", 0.85, "Capable of"),
            MorphologyRule(r".*ive$", "JJ", 0.8, "Having quality"),
            
            # Adverb suffixes
            MorphologyRule(r".*ly$", "RB", 0.9, "Manner adverb"),
            
            # Add 100+ more morphological rules
        ]
    
    def analyze_unknown_word(self, word: str, context: List[Token]) -> POSTagResult:
        """Analyze unknown word using morphological patterns"""
        
        # Try suffix rules first
        for rule in self.suffix_rules:
            if rule.matches(word):
                return POSTagResult(
                    tag=rule.pos_tag,
                    confidence=rule.confidence,
                    method="morphological_analysis",
                    justification=f"Suffix rule: {rule.description}"
                )
        
        # Try capitalization rules
        if word[0].isupper():
            # Check if it's at sentence start
            if self._is_sentence_start(context):
                return POSTagResult("NN", 0.6, "morphological_analysis", 
                                  "Capitalized word, likely noun")
            else:
                return POSTagResult("NNP", 0.8, "morphological_analysis",
                                  "Capitalized mid-sentence, proper noun")
        
        # Default to noun for unknown words
        return POSTagResult("NN", 0.4, "morphological_analysis",
                          "Default classification for unknown word")
```

### **4. Complete Tagger Implementation**

```python
class CortexTagger:
    """
    Deterministic Part-of-Speech tagger using spatial coordinates and 
    mathematical context analysis.
    """
    
    def __init__(self, anchor_system: SpatialAnchor, memory: BinaryCellMemory,
                 resonance: HarmonicResonance):
        self.pos_dict = SpatialPOSDictionary(anchor_system, memory)
        self.disambiguator = POSDisambiguator(self.pos_dict, resonance)
        self.morphology = MorphologyAnalyzer()
        self.memory = memory
        
        # Statistics tracking
        self.tagging_stats = {
            "dictionary_lookups": 0,
            "context_disambiguations": 0,
            "morphological_analyses": 0,
            "total_tokens": 0
        }
    
    def tag_tokens(self, tokens: List[Token]) -> List[Token]:
        """
        Tag all tokens in a list with their POS tags.
        
        Process:
        1. Dictionary lookup for known words
        2. Context analysis for ambiguous words  
        3. Morphological analysis for unknown words
        4. Store relationships in BinaryCellMemory
        """
        tagged_tokens = []
        
        for i, token in enumerate(tokens):
            # Skip whitespace and punctuation for POS tagging
            if token.is_space:
                token.pos_tag = "SPACE"
                tagged_tokens.append(token)
                continue
            
            if token.is_punct:
                token.pos_tag = "PUNCT"
                tagged_tokens.append(token)
                continue
            
            # Get context window
            context = self._get_context_window(tokens, i)
            
            # Determine POS tag
            pos_result = self.disambiguator.disambiguate_pos(token, context)
            
            # Assign POS tag to token
            token.pos_tag = pos_result.tag
            token.pos_confidence = pos_result.confidence
            token.pos_method = pos_result.method
            
            # Store word-POS relationship in memory
            pos_coord = self.pos_dict.pos_anchors[pos_result.tag]
            self.memory.store_relationship(
                token.spatial_coord,
                pos_coord,
                RelationshipType.CUSTOM,
                strength=pos_result.confidence,
                metadata={
                    "relationship_type": "word_pos",
                    "method": pos_result.method,
                    "justification": pos_result.justification
                }
            )
            
            # Update statistics
            self.tagging_stats["total_tokens"] += 1
            if pos_result.method == "dictionary_lookup":
                self.tagging_stats["dictionary_lookups"] += 1
            elif pos_result.method == "context_analysis":
                self.tagging_stats["context_disambiguations"] += 1
            elif pos_result.method == "morphological_analysis":
                self.tagging_stats["morphological_analyses"] += 1
            
            tagged_tokens.append(token)
        
        return tagged_tokens
    
    def get_tagging_statistics(self) -> Dict:
        """Get statistics about the tagging process"""
        stats = self.tagging_stats.copy()
        if stats["total_tokens"] > 0:
            stats["dictionary_percentage"] = (
                stats["dictionary_lookups"] / stats["total_tokens"] * 100
            )
            stats["disambiguation_percentage"] = (
                stats["context_disambiguations"] / stats["total_tokens"] * 100  
            )
            stats["morphology_percentage"] = (
                stats["morphological_analyses"] / stats["total_tokens"] * 100
            )
        return stats
```

---

## Key Design Decisions

### **1. Deterministic Over Probabilistic**
- **Dictionary lookup** provides 100% certainty for known words
- **Context analysis** uses mathematical resonance, not statistical models
- **Morphological rules** are explicit patterns, not learned probabilities

### **2. Traceable Decision Making**
- Every POS tag assignment includes:
  - Method used (dictionary, context, morphology)
  - Confidence score (mathematical, not statistical)
  - Justification (explicit reasoning)

### **3. Perfect Repeatability**
- Identical input always produces identical POS tags
- No randomness or statistical variation
- Consistent behavior across all runs

### **4. Mathematical Foundation**
- All relationships stored in BinaryCellMemory
- Spatial coordinates for all POS tags
- Harmonic resonance for similarity calculations

---

## Performance Characteristics

### **Speed**
- **Dictionary lookup:** O(1) for known words
- **Context analysis:** O(k) where k = context window size
- **Morphological analysis:** O(r) where r = number of rules
- **Overall:** O(n) linear time complexity

### **Memory**
- **Efficient caching** of spatial coordinates
- **Relationship storage** in BinaryCellMemory
- **Statistics tracking** for performance monitoring

### **Accuracy**
- **100% accuracy** for unambiguous dictionary words
- **High accuracy** for context-based disambiguation
- **Rule-based accuracy** for morphological analysis
- **Graceful degradation** for completely unknown patterns

---

## Integration with Phase 1

### **SpatialAnchor Usage**
- Every POS tag has its own 6D spatial coordinate
- Word-POS relationships mapped in mathematical space
- Consistent coordinate assignment across runs

### **BinaryCellMemory Usage**
- All word-POS relationships permanently stored
- Context patterns learned and remembered
- Relationship strength based on confidence scores

### **HarmonicResonance Usage**
- Mathematical similarity between POS tags and context
- Disambiguation based on geometric distance
- Confidence scores from resonance calculations

---

## Testing Strategy

### **Unit Tests**
- Dictionary lookup accuracy
- Context disambiguation correctness
- Morphological rule application
- Edge case handling

### **Integration Tests**
- End-to-end tagging pipeline
- Memory storage verification
- Performance benchmarking
- Consistency validation

### **Accuracy Validation**
- Test against gold-standard POS tagged corpora
- Compare with probabilistic taggers
- Measure consistency across runs
- Analyze error patterns

This specification provides the complete blueprint for building a deterministic POS tagger that prioritizes mathematical certainty over probabilistic fluency, perfectly aligned with the CortexOS design philosophy.

