# CortexOS NLP - Phase 2: Linguistic Layer - Detailed Architecture

**Mission:** Connect the Phase 1 mathematical foundation to actual language processing through deterministic linguistic components that replace probabilistic NLP with mathematical certainty.

**Core Architecture:** Each Phase 2 module leverages the Phase 1 foundation (SpatialAnchor, BinaryCellMemory, HarmonicResonance) to provide deterministic language understanding with mathematical proof.

---

## Module 1: CortexTokenizer - Deterministic Text Segmentation

### **Purpose**
Convert raw text into spatially-anchored tokens with mathematical certainty, replacing probabilistic tokenization with rule-based deterministic segmentation.

### **Architecture Overview**
```
Raw Text → Rule-Based Segmentation → Spatial Anchoring → Token Objects
    ↓              ↓                      ↓               ↓
"Hello world"  → ["Hello", " ", "world"] → 6D Coords → [Token, Token, Token]
```

### **Core Components**

#### **1. Token Class**
```python
@dataclass
class Token:
    text: str                    # Original text
    normalized: str             # Normalized for spatial anchoring
    spatial_coord: SpatialCoordinate  # 6D mathematical position
    start_char: int             # Character position in source
    end_char: int               # End position in source
    token_id: int               # Sequential ID in document
    is_alpha: bool              # Contains letters
    is_digit: bool              # Contains numbers
    is_punct: bool              # Is punctuation
    is_space: bool              # Is whitespace
    is_stop: bool               # Is stop word
    pos_tag: str = None         # Part-of-speech (added by Tagger)
    dep_label: str = None       # Dependency label (added by Parser)
    head_token: 'Token' = None  # Syntactic head (added by Parser)
```

#### **2. Tokenization Engine**
- **Rule-Based Patterns:** Regex patterns for consistent segmentation
- **Contraction Handling:** Deterministic expansion (can't → can not)
- **Normalization:** Case folding and whitespace handling
- **Spatial Anchoring:** Immediate 6D coordinate assignment via SpatialAnchor

#### **3. Performance Characteristics**
- **Speed:** O(n) linear time complexity
- **Memory:** Efficient caching of spatial coordinates
- **Consistency:** Identical input always produces identical tokens
- **Reconstruction:** Perfect text reconstruction from tokens

### **Integration with Phase 1**
- **SpatialAnchor:** Every token gets deterministic 6D coordinates
- **BinaryCellMemory:** Token relationships stored for context
- **HarmonicResonance:** Token similarity calculations

---

## Module 2: CortexTagger - Deterministic Part-of-Speech Tagging

### **Purpose**
Assign grammatical roles (Noun, Verb, Adjective, etc.) to tokens using mathematical certainty instead of probabilistic models.

### **Architecture Overview**
```
Tokens → Dictionary Lookup → Context Analysis → Disambiguation → POS Tags
   ↓           ↓                 ↓               ↓             ↓
[Token]  → Base POS Tags → Relationship Check → Final Tags → Tagged Tokens
```

### **Core Components**

#### **1. Deterministic POS Dictionary**
```python
class POSDictionary:
    # Core word-to-POS mappings with spatial coordinates
    word_pos_map: Dict[SpatialCoordinate, List[POSTag]]
    
    # POS tag spatial anchors for mathematical operations
    pos_anchors: Dict[str, SpatialCoordinate]  # "NOUN" → 6D coords
    
    # Morphological rules for unknown words
    morphology_rules: List[MorphologyRule]
```

**POS Tag Set (Penn Treebank Extended):**
- **Nouns:** NN, NNS, NNP, NNPS
- **Verbs:** VB, VBD, VBG, VBN, VBP, VBZ
- **Adjectives:** JJ, JJR, JJS
- **Adverbs:** RB, RBR, RBS
- **Pronouns:** PRP, PRP$, WP, WP$
- **Determiners:** DT, WDT
- **Prepositions:** IN
- **Conjunctions:** CC
- **Particles:** RP
- **Interjections:** UH
- **Punctuation:** PUNCT
- **Numbers:** CD
- **Symbols:** SYM

#### **2. Context-Aware Disambiguation**
```python
class POSDisambiguator:
    def disambiguate(self, token: Token, context: List[Token]) -> str:
        # 1. Get all possible POS tags for token
        candidates = self.get_pos_candidates(token)
        
        # 2. Calculate harmonic resonance with context
        context_scores = []
        for pos_tag in candidates:
            score = self.calculate_context_resonance(pos_tag, context)
            context_scores.append((pos_tag, score))
        
        # 3. Return highest scoring POS tag
        return max(context_scores, key=lambda x: x[1])[0]
```

#### **3. Mathematical POS Assignment Process**

**Step 1: Dictionary Lookup**
```python
# Get spatial coordinate for token
token_coord = token.spatial_coord

# Lookup possible POS tags
pos_candidates = pos_dictionary.get_pos_tags(token_coord)
```

**Step 2: Context Analysis Using HarmonicResonance**
```python
# For each POS candidate, calculate resonance with surrounding tokens
for pos_tag in pos_candidates:
    pos_coord = pos_dictionary.get_pos_anchor(pos_tag)
    
    # Calculate resonance with left and right context
    left_resonance = harmonic_resonance.calculate_resonance(
        pos_coord, context_left.spatial_coord
    )
    right_resonance = harmonic_resonance.calculate_resonance(
        pos_coord, context_right.spatial_coord
    )
    
    # Combine scores mathematically
    total_score = (left_resonance.similarity_score + 
                   right_resonance.similarity_score) / 2
```

**Step 3: Deterministic Selection**
- Choose POS tag with highest mathematical resonance score
- Store relationship in BinaryCellMemory for future reference
- Assign with mathematical confidence level

#### **4. Unknown Word Handling**
```python
class MorphologyAnalyzer:
    def analyze_unknown_word(self, token: Token) -> str:
        # Suffix-based rules
        if token.text.endswith('ing'):
            return 'VBG'  # Present participle
        elif token.text.endswith('ed'):
            return 'VBD'  # Past tense
        elif token.text.endswith('ly'):
            return 'RB'   # Adverb
        elif token.text.endswith('tion'):
            return 'NN'   # Noun
        
        # Capitalization rules
        if token.text[0].isupper() and not token.start_char == 0:
            return 'NNP'  # Proper noun
        
        # Default to noun for unknown words
        return 'NN'
```

### **Performance Characteristics**
- **Accuracy:** Mathematical certainty for known words, rule-based for unknown
- **Speed:** O(1) dictionary lookup + O(k) context analysis (k = context window)
- **Consistency:** Identical context produces identical POS tags
- **Learning:** Relationships stored in BinaryCellMemory improve over time

### **Integration with Phase 1**
- **SpatialAnchor:** POS tags have their own 6D coordinates
- **BinaryCellMemory:** Word-POS relationships stored permanently
- **HarmonicResonance:** Context-based disambiguation through similarity

---

## Module 3: CortexParser - Deterministic Dependency Parsing

### **Purpose**
Determine grammatical structure and word relationships using mathematical graph analysis instead of probabilistic parsing models.

### **Architecture Overview**
```
Tagged Tokens → Grammar Rules → Relationship Graph → Dependency Tree
      ↓              ↓              ↓                 ↓
   [Token+POS] → Apply Rules → BinaryCellMemory → Mathematical Tree
```

### **Core Components**

#### **1. Dependency Relationship Types**
```python
class DependencyType(Enum):
    # Core grammatical relationships
    ROOT = "root"           # Root of the sentence
    SUBJECT = "nsubj"       # Nominal subject
    OBJECT = "dobj"         # Direct object
    MODIFIER = "amod"       # Adjectival modifier
    DETERMINER = "det"      # Determiner
    PREPOSITION = "prep"    # Prepositional modifier
    CONJUNCTION = "conj"    # Conjunct
    AUXILIARY = "aux"       # Auxiliary verb
    COPULA = "cop"         # Copula
    COMPLEMENT = "xcomp"    # Open clausal complement
    
    # Extended relationships
    COMPOUND = "compound"   # Compound modifier
    APPOSITION = "appos"    # Appositional modifier
    RELATIVE = "relcl"      # Relative clause modifier
    ADVERBIAL = "advmod"    # Adverbial modifier
```

#### **2. Grammar Rule Engine**
```python
class GrammarRuleEngine:
    def __init__(self, memory: BinaryCellMemory):
        self.memory = memory
        self.rules = self._load_grammar_rules()
    
    def _load_grammar_rules(self) -> List[GrammarRule]:
        return [
            # Subject-Verb-Object patterns
            GrammarRule(
                pattern=["NOUN", "VERB", "NOUN"],
                relationships=[
                    (0, 1, DependencyType.SUBJECT),  # NOUN[0] is subject of VERB[1]
                    (1, 2, DependencyType.OBJECT)    # VERB[1] has object NOUN[2]
                ]
            ),
            
            # Determiner-Noun patterns
            GrammarRule(
                pattern=["DT", "NOUN"],
                relationships=[
                    (1, 0, DependencyType.DETERMINER)  # NOUN[1] has determiner DT[0]
                ]
            ),
            
            # Adjective-Noun patterns
            GrammarRule(
                pattern=["ADJ", "NOUN"],
                relationships=[
                    (1, 0, DependencyType.MODIFIER)  # NOUN[1] modified by ADJ[0]
                ]
            ),
            
            # Add 50+ more grammar rules for comprehensive coverage
        ]
```

#### **3. Mathematical Parsing Process**

**Step 1: Pattern Recognition**
```python
def find_grammar_patterns(self, tokens: List[Token]) -> List[PatternMatch]:
    matches = []
    
    for rule in self.rules:
        # Slide window across tokens looking for pattern matches
        for i in range(len(tokens) - len(rule.pattern) + 1):
            window = tokens[i:i + len(rule.pattern)]
            
            if self.matches_pattern(window, rule.pattern):
                matches.append(PatternMatch(rule, i, window))
    
    return matches
```

**Step 2: Relationship Scoring**
```python
def score_relationship(self, head: Token, dependent: Token, 
                      rel_type: DependencyType) -> float:
    # Get spatial coordinates
    head_coord = head.spatial_coord
    dep_coord = dependent.spatial_coord
    rel_coord = self.get_relationship_anchor(rel_type)
    
    # Calculate harmonic resonance for this relationship
    head_rel_resonance = self.harmonic_resonance.calculate_resonance(
        head_coord, rel_coord
    )
    dep_rel_resonance = self.harmonic_resonance.calculate_resonance(
        dep_coord, rel_coord
    )
    
    # Mathematical relationship strength
    return (head_rel_resonance.similarity_score + 
            dep_rel_resonance.similarity_score) / 2
```

**Step 3: Conflict Resolution**
```python
def resolve_conflicts(self, potential_relationships: List[Relationship]) -> List[Relationship]:
    # When multiple rules could apply, choose based on mathematical scores
    relationship_scores = []
    
    for rel in potential_relationships:
        score = self.score_relationship(rel.head, rel.dependent, rel.type)
        relationship_scores.append((rel, score))
    
    # Sort by score and resolve conflicts
    sorted_rels = sorted(relationship_scores, key=lambda x: x[1], reverse=True)
    
    # Apply non-conflicting relationships with highest scores
    final_relationships = []
    used_tokens = set()
    
    for rel, score in sorted_rels:
        if rel.dependent not in used_tokens:
            final_relationships.append(rel)
            used_tokens.add(rel.dependent)
    
    return final_relationships
```

#### **4. Dependency Tree Construction**
```python
class DependencyTree:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.root = None
        self.relationships = []
    
    def build_tree(self, relationships: List[Relationship]):
        # Store relationships in BinaryCellMemory
        for rel in relationships:
            self.memory.store_relationship(
                rel.head.spatial_coord,
                rel.dependent.spatial_coord,
                RelationshipType.DEPENDENCY,
                strength=rel.confidence,
                metadata={"dep_type": rel.type.value}
            )
        
        # Build tree structure
        self.relationships = relationships
        self.root = self.find_root()
    
    def find_root(self) -> Token:
        # Find token that is not dependent on any other token
        dependents = {rel.dependent for rel in self.relationships}
        for token in self.tokens:
            if token not in dependents and token.pos_tag.startswith('V'):
                return token
        
        # Fallback: first verb or first token
        return next((t for t in self.tokens if t.pos_tag.startswith('V')), 
                   self.tokens[0])
```

### **Performance Characteristics**
- **Accuracy:** Rule-based with mathematical scoring for disambiguation
- **Speed:** O(n²) for pattern matching, O(n log n) for conflict resolution
- **Consistency:** Deterministic tree construction from identical input
- **Extensibility:** Easy to add new grammar rules and relationship types

### **Integration with Phase 1**
- **SpatialAnchor:** Dependency relationships have 6D coordinates
- **BinaryCellMemory:** All grammatical relationships stored permanently
- **HarmonicResonance:** Mathematical scoring for relationship strength

---

## Phase 2 Integration Architecture

### **Data Flow**
```
Raw Text
    ↓
CortexTokenizer → [Token objects with spatial coordinates]
    ↓
CortexTagger → [Token objects with POS tags]
    ↓
CortexParser → [Token objects with dependency relationships]
    ↓
Structured Document (ready for Phase 3 API)
```

### **Shared Components**
1. **Spatial Coordinate System:** All linguistic elements have 6D positions
2. **Relationship Storage:** BinaryCellMemory stores all linguistic relationships
3. **Similarity Calculations:** HarmonicResonance provides mathematical similarity
4. **Caching System:** Efficient storage and retrieval of computed results

### **Performance Targets**
- **Speed:** 10-100x faster than probabilistic NLP (CPU-only processing)
- **Memory:** Efficient caching with minimal memory footprint
- **Accuracy:** Mathematical certainty for known patterns, high accuracy for unknown
- **Consistency:** Perfect reproducibility across runs

### **Error Handling**
- **Unknown Words:** Morphological analysis with confidence scores
- **Ambiguous Grammar:** Mathematical scoring chooses best interpretation
- **Malformed Input:** Graceful degradation with partial parsing

---

## Implementation Priority

### **Phase 2.1: Complete Tokenizer** ✅ (In Progress)
- Finalize contraction handling
- Add comprehensive test suite
- Optimize performance

### **Phase 2.2: Implement Tagger**
- Build POS dictionary with spatial anchors
- Implement context-aware disambiguation
- Add morphological analysis for unknown words

### **Phase 2.3: Implement Parser**
- Create grammar rule engine
- Build dependency relationship scoring
- Implement tree construction algorithm

### **Phase 2.4: Integration Testing**
- End-to-end pipeline testing
- Performance benchmarking
- Accuracy validation

This architecture provides the detailed blueprint for building the world's first mathematically certain NLP linguistic layer, replacing probabilistic guessing with deterministic mathematical processing.

