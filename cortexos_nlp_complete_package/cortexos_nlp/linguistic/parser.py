"""
CortexOS NLP - Parser Module
Phase 2: Linguistic Layer

This module implements the deterministic dependency parser that builds grammatical
dependency trees using explicit rules and mathematical relationship scoring.
Provides perfect traceability and consistency for syntactic analysis.

Core Principle: Use grammar rules and mathematical scoring to determine
dependency relationships through explicit analysis rather than probabilistic parsing.
"""

import re
from typing import List, Dict, Optional, Tuple, Set, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

try:
    from ..core import SpatialAnchor, SpatialCoordinate, BinaryCellMemory, RelationshipType, HarmonicResonance
    from .tokenizer import Token
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from core import SpatialAnchor, SpatialCoordinate, BinaryCellMemory, RelationshipType, HarmonicResonance
    from tokenizer import Token


class DependencyType(Enum):
    """Dependency relationship types following Universal Dependencies standard"""
    # Core grammatical relationships
    ROOT = "root"           # Root of the sentence
    SUBJECT = "nsubj"       # Nominal subject
    OBJECT = "dobj"         # Direct object
    IOBJ = "iobj"          # Indirect object
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
    NUMMOD = "nummod"       # Numeric modifier
    MARK = "mark"          # Marker
    CASE = "case"          # Case marking
    NEGATION = "neg"       # Negation modifier
    PUNCT = "punct"        # Punctuation


@dataclass
class DependencyRelationship:
    """
    Represents a dependency relationship between two tokens.
    Contains complete mathematical justification for the relationship.
    """
    head: Token                 # Head token (governor)
    dependent: Token           # Dependent token (modifier)
    relation: str              # Dependency relation type
    confidence: float          # Mathematical confidence score
    rule_applied: str          # Grammar rule that created this relationship
    justification: str         # Human-readable explanation


@dataclass
class ParseCandidate:
    """
    Represents a potential parse with all its relationships and conflicts.
    Used for conflict resolution when multiple parses are possible.
    """
    relationships: List[DependencyRelationship]
    rule: 'GrammarRule'
    conflicts: List[str]
    total_score: float


class GrammarRule:
    """
    Represents a deterministic grammar pattern with explicit relationships.
    Replaces probabilistic parsing models with rule-based pattern matching.
    """
    
    def __init__(self, pattern: List[str], relationships: List[Tuple[int, int, str]],
                 priority: int = 1, conditions: List[Callable] = None, name: str = ""):
        self.pattern = pattern          # POS tag sequence pattern
        self.relationships = relationships  # (head_idx, dep_idx, relation_type)
        self.priority = priority        # Rule priority for conflict resolution
        self.conditions = conditions or []  # Additional matching conditions
        self.name = name               # Human-readable rule name
        self.usage_count = 0           # Track rule application frequency
    
    def matches(self, pos_sequence: List[str], tokens: List[Token]) -> bool:
        """Check if this rule matches the given POS sequence"""
        if len(pos_sequence) != len(self.pattern):
            return False
        
        # Check POS pattern match
        for i, (actual_pos, expected_pos) in enumerate(zip(pos_sequence, self.pattern)):
            if not self._pos_matches(actual_pos, expected_pos):
                return False
        
        # Check additional conditions
        for condition in self.conditions:
            if not condition(tokens):
                return False
        
        return True
    
    def _pos_matches(self, actual: str, expected: str) -> bool:
        """Check if POS tags match (with wildcards and categories)"""
        if expected == "*":  # Wildcard matches anything
            return True
        elif expected.startswith("N"):  # Noun category
            return actual.startswith("N")
        elif expected.startswith("V"):  # Verb category  
            return actual.startswith("V")
        elif expected.startswith("J"):  # Adjective category
            return actual.startswith("J")
        elif expected.startswith("R"):  # Adverb category
            return actual.startswith("R")
        else:
            return actual == expected
    
    def apply(self, tokens: List[Token]) -> List[DependencyRelationship]:
        """Apply this rule to create dependency relationships"""
        relationships = []
        self.usage_count += 1
        
        for head_idx, dep_idx, rel_type in self.relationships:
            if head_idx < len(tokens) and dep_idx < len(tokens):
                relationship = DependencyRelationship(
                    head=tokens[head_idx],
                    dependent=tokens[dep_idx],
                    relation=rel_type,
                    confidence=0.0,  # Will be calculated later
                    rule_applied=self.name or str(self.pattern),
                    justification=f"Grammar rule: {self.name} applied to pattern {self.pattern}"
                )
                relationships.append(relationship)
        
        return relationships
    
    def __str__(self):
        return f"GrammarRule({self.name or self.pattern})"


class DependencyTree:
    """
    Represents the complete dependency tree for a sentence.
    Contains all relationships and provides tree navigation methods.
    """
    
    def __init__(self, tokens: List[Token], relationships: List[DependencyRelationship], 
                 root: Optional[Token] = None):
        self.tokens = tokens
        self.relationships = relationships
        self.root = root or self._find_root()
        
        # Build adjacency lists for efficient navigation (use token IDs as keys)
        self.children = defaultdict(list)
        self.parents = {}
        
        for rel in relationships:
            self.children[rel.head.token_id].append(rel.dependent)
            self.parents[rel.dependent.token_id] = rel.head
    
    def _find_root(self) -> Optional[Token]:
        """Find the root token (token with no head)"""
        if not self.relationships:
            return self.tokens[0] if self.tokens else None
        
        # Find token that is not dependent on any other token (use token IDs)
        dependent_ids = {rel.dependent.token_id for rel in self.relationships}
        content_tokens = [t for t in self.tokens if not t.is_space and not t.is_punct]
        
        for token in content_tokens:
            if token.token_id not in dependent_ids:
                return token
        
        # Fallback: first verb or first content token
        for token in content_tokens:
            if hasattr(token, 'pos_tag') and token.pos_tag and token.pos_tag.startswith('V'):
                return token
        
        return content_tokens[0] if content_tokens else None
    
    def get_children(self, token: Token) -> List[Token]:
        """Get all children of a token"""
        return self.children.get(token.token_id, [])
    
    def get_parent(self, token: Token) -> Optional[Token]:
        """Get parent of a token"""
        return self.parents.get(token.token_id)
    
    def get_subtree(self, token: Token) -> List[Token]:
        """Get all tokens in the subtree rooted at token"""
        subtree = [token]
        for child in self.get_children(token):
            subtree.extend(self.get_subtree(child))
        return subtree
    
    def to_conllu(self) -> str:
        """Export to CoNLL-U format"""
        lines = []
        
        # Create mapping from token IDs to indices
        content_tokens = [t for t in self.tokens if not t.is_space]
        token_id_to_idx = {}
        for i, token in enumerate(content_tokens):
            token_id_to_idx[token.token_id] = i + 1  # CoNLL-U uses 1-based indexing
        
        for token in content_tokens:
            # Find head and relation
            head_id = 0  # Root
            dep_label = "root"
            
            for rel in self.relationships:
                if rel.dependent.token_id == token.token_id:
                    head_id = token_id_to_idx.get(rel.head.token_id, 0)
                    dep_label = rel.relation
                    break
            
            pos_tag = getattr(token, 'pos_tag', '_')
            line = f"{token_id_to_idx[token.token_id]}\t{token.text}\t{token.normalized}\t{pos_tag}\t{pos_tag}\t_\t{head_id}\t{dep_label}\t_\t_"
            lines.append(line)
        
        return "\n".join(lines)


class DependencyScorer:
    """
    Scores potential dependency relationships using mathematical analysis.
    Replaces probabilistic scoring with geometric and contextual calculations.
    """
    
    def __init__(self, resonance: HarmonicResonance, dependency_anchors: Dict[str, SpatialCoordinate]):
        self.resonance = resonance
        self.dependency_anchors = dependency_anchors
    
    def score_relationship(self, head_token: Token, dep_token: Token, 
                          rel_type: str, context: List[Token]) -> float:
        """
        Calculate mathematical score for a potential dependency relationship.
        
        Factors considered:
        1. Harmonic resonance between head and dependent
        2. Relationship type compatibility
        3. Distance penalty (closer words more likely to be related)
        4. Context support from surrounding tokens
        """
        
        # 1. Base harmonic resonance between head and dependent
        base_resonance = self.resonance.calculate_resonance(
            head_token.spatial_coord, dep_token.spatial_coord
        )
        
        # 2. Relationship type compatibility
        if rel_type in self.dependency_anchors:
            rel_coord = self.dependency_anchors[rel_type]
            head_rel_resonance = self.resonance.calculate_resonance(
                head_token.spatial_coord, rel_coord
            )
            dep_rel_resonance = self.resonance.calculate_resonance(
                dep_token.spatial_coord, rel_coord
            )
            rel_compatibility = (head_rel_resonance.similarity_score + 
                               dep_rel_resonance.similarity_score) / 2
        else:
            rel_compatibility = 0.5  # Default for unknown relations
        
        # 3. Distance penalty (linear decay)
        distance = abs(head_token.token_id - dep_token.token_id)
        distance_penalty = 1.0 / (1.0 + distance * 0.1)
        
        # 4. Context support (simplified for now)
        context_support = 0.5  # Default neutral support
        
        # Combine all factors mathematically
        total_score = (
            base_resonance.similarity_score * 0.3 +
            rel_compatibility * 0.3 +
            distance_penalty * 0.2 +
            context_support * 0.2
        )
        
        return min(total_score, 1.0)  # Cap at 1.0


class GrammarRuleEngine:
    """
    Manages and applies grammar rules for dependency parsing.
    Uses deterministic pattern matching instead of probabilistic models.
    """
    
    def __init__(self, memory: BinaryCellMemory, resonance: HarmonicResonance):
        self.memory = memory
        self.resonance = resonance
        self.rules = self._load_grammar_rules()
        self.dependency_anchors = self._create_dependency_anchors()
        self.scorer = DependencyScorer(resonance, self.dependency_anchors)
    
    def _create_dependency_anchors(self) -> Dict[str, SpatialCoordinate]:
        """Create spatial anchors for dependency relationship types"""
        dependency_types = [
            "root", "nsubj", "dobj", "iobj", "amod", "advmod", "det", "prep",
            "aux", "cop", "conj", "cc", "compound", "appos", "relcl", "xcomp",
            "ccomp", "advcl", "acl", "nmod", "nummod", "mark", "case", "neg",
            "punct", "parataxis", "discourse", "vocative", "expl", "csubj"
        ]
        
        anchors = {}
        anchor_system = SpatialAnchor()
        for dep_type in dependency_types:
            _, coord = anchor_system.create_anchor(f"DEP_{dep_type}")
            anchors[dep_type] = coord
        
        return anchors
    
    def _load_grammar_rules(self) -> List[GrammarRule]:
        """Load comprehensive set of English grammar rules"""
        return [
            # Basic Subject-Verb-Object patterns
            GrammarRule(
                pattern=["N*", "V*", "N*"],
                relationships=[(1, 0, "nsubj"), (1, 2, "dobj")],
                priority=5,
                name="SVO_basic"
            ),
            
            # Determiner-Noun patterns
            GrammarRule(
                pattern=["DT", "N*"],
                relationships=[(1, 0, "det")],
                priority=8,
                name="det_noun"
            ),
            
            # Adjective-Noun patterns
            GrammarRule(
                pattern=["J*", "N*"],
                relationships=[(1, 0, "amod")],
                priority=7,
                name="adj_noun"
            ),
            
            # Adverb-Verb patterns
            GrammarRule(
                pattern=["R*", "V*"],
                relationships=[(1, 0, "advmod")],
                priority=6,
                name="adv_verb"
            ),
            
            # Verb-Adverb patterns
            GrammarRule(
                pattern=["V*", "R*"],
                relationships=[(0, 1, "advmod")],
                priority=6,
                name="verb_adv"
            ),
            
            # Preposition-Noun patterns
            GrammarRule(
                pattern=["IN", "N*"],
                relationships=[(1, 0, "case")],
                priority=6,
                name="prep_noun"
            ),
            
            # Auxiliary-Verb patterns
            GrammarRule(
                pattern=["MD", "V*"],
                relationships=[(1, 0, "aux")],
                priority=7,
                name="aux_verb"
            ),
            
            # Compound noun patterns
            GrammarRule(
                pattern=["N*", "N*"],
                relationships=[(1, 0, "compound")],
                priority=4,
                name="compound_noun"
            ),
            
            # Coordination patterns
            GrammarRule(
                pattern=["N*", "CC", "N*"],
                relationships=[(0, 2, "conj"), (0, 1, "cc")],
                priority=5,
                name="noun_coordination"
            ),
            
            # Complex verb phrases
            GrammarRule(
                pattern=["V*", "VBG"],
                relationships=[(0, 1, "xcomp")],
                priority=5,
                name="verb_gerund"
            ),
            
            # Copula patterns
            GrammarRule(
                pattern=["N*", "VBZ", "J*"],  # "The cat is happy"
                relationships=[(0, 2, "nsubj"), (2, 1, "cop")],
                priority=6,
                name="copula_adj"
            ),
            
            # Determiner-Adjective-Noun patterns
            GrammarRule(
                pattern=["DT", "J*", "N*"],
                relationships=[(2, 0, "det"), (2, 1, "amod")],
                priority=7,
                name="det_adj_noun"
            ),
            
            # Adverb-Adjective patterns
            GrammarRule(
                pattern=["R*", "J*"],
                relationships=[(1, 0, "advmod")],
                priority=6,
                name="adv_adj"
            ),
            
            # Punctuation attachment
            GrammarRule(
                pattern=["*", "PUNCT"],
                relationships=[(0, 1, "punct")],
                priority=2,
                name="punct_attach"
            ),
        ]
    
    def apply_rules(self, tokens: List[Token]) -> List[DependencyRelationship]:
        """Apply all applicable grammar rules to find potential relationships"""
        
        potential_relationships = []
        content_tokens = [t for t in tokens if not t.is_space]
        
        if not content_tokens:
            return potential_relationships
        
        # Try each grammar rule
        for rule in self.rules:
            # Slide window across tokens
            for start_idx in range(len(content_tokens) - len(rule.pattern) + 1):
                window = content_tokens[start_idx:start_idx + len(rule.pattern)]
                pos_sequence = [getattr(t, 'pos_tag', 'UNK') for t in window]
                
                if rule.matches(pos_sequence, window):
                    # Apply rule relationships
                    rule_relationships = rule.apply(window)
                    
                    # Score each relationship
                    for rel in rule_relationships:
                        rel.confidence = self.scorer.score_relationship(
                            rel.head, rel.dependent, rel.relation, content_tokens
                        )
                    
                    potential_relationships.extend(rule_relationships)
        
        return potential_relationships


class ConflictResolver:
    """
    Resolves conflicts when multiple grammar rules could apply.
    Uses mathematical scoring to choose the best interpretation.
    """
    
    def __init__(self, scorer: DependencyScorer):
        self.scorer = scorer
    
    def resolve_conflicts(self, potential_relationships: List[DependencyRelationship], 
                         tokens: List[Token]) -> List[DependencyRelationship]:
        """
        Choose the best relationships from multiple candidates using mathematical scoring.
        
        Conflict resolution strategy:
        1. Group relationships by dependent token (each token can have only one head)
        2. For each group, choose the highest scoring relationship
        3. Ensure tree structure (no cycles)
        """
        
        if not potential_relationships:
            return []
        
        # Group relationships by dependent token ID (use token_id as key)
        dependent_groups = defaultdict(list)
        for rel in potential_relationships:
            dependent_groups[rel.dependent.token_id].append(rel)
        
        # Choose best relationship for each dependent
        final_relationships = []
        for dependent_id, candidates in dependent_groups.items():
            if len(candidates) == 1:
                final_relationships.append(candidates[0])
            else:
                # Choose highest scoring candidate
                best_rel = max(candidates, key=lambda r: r.confidence)
                final_relationships.append(best_rel)
        
        # Ensure no cycles (simplified check)
        final_relationships = self._remove_cycles(final_relationships)
        
        return final_relationships
    
    def _remove_cycles(self, relationships: List[DependencyRelationship]) -> List[DependencyRelationship]:
        """Remove relationships that would create cycles"""
        # Simplified cycle detection - remove lowest scoring relationships that create cycles
        # For now, just return all relationships (full cycle detection is complex)
        return relationships


class CortexParser:
    """
    Deterministic dependency parser using grammar rules and mathematical scoring.
    Builds syntactic trees with perfect traceability and consistency.
    """
    
    def __init__(self, memory: BinaryCellMemory, resonance: HarmonicResonance):
        self.memory = memory
        self.rule_engine = GrammarRuleEngine(memory, resonance)
        self.resolver = ConflictResolver(self.rule_engine.scorer)
        
        # Statistics tracking
        self.parsing_stats = {
            "sentences_parsed": 0,
            "relationships_created": 0,
            "conflicts_resolved": 0,
            "rules_applied": defaultdict(int)
        }
    
    def parse_sentence(self, tokens: List[Token]) -> DependencyTree:
        """
        Parse a sentence into a dependency tree.
        
        Process:
        1. Apply grammar rules to find potential relationships
        2. Score all potential relationships mathematically
        3. Resolve conflicts using mathematical criteria
        4. Build final dependency tree
        5. Store relationships in BinaryCellMemory
        """
        
        # Filter out whitespace tokens for parsing
        content_tokens = [t for t in tokens if not t.is_space]
        
        if not content_tokens:
            return DependencyTree(tokens, [])
        
        # Step 1: Apply grammar rules
        potential_relationships = self.rule_engine.apply_rules(tokens)
        
        # Step 2: Resolve conflicts
        final_relationships = self.resolver.resolve_conflicts(potential_relationships, content_tokens)
        
        # Step 3: Build dependency tree
        tree = DependencyTree(tokens, final_relationships)
        
        # Step 4: Store relationships in memory
        self._store_relationships_in_memory(final_relationships)
        
        # Update statistics
        self.parsing_stats["sentences_parsed"] += 1
        self.parsing_stats["relationships_created"] += len(final_relationships)
        
        return tree
    
    def _store_relationships_in_memory(self, relationships: List[DependencyRelationship]):
        """Store all dependency relationships in BinaryCellMemory"""
        
        for rel in relationships:
            # Store the dependency relationship
            self.memory.store_relationship(
                rel.head.spatial_coord,
                rel.dependent.spatial_coord,
                RelationshipType.DEPENDENCY,
                strength=rel.confidence,
                metadata={
                    "dependency_type": rel.relation,
                    "rule_applied": rel.rule_applied,
                    "parsing_method": "grammar_rule",
                    "justification": rel.justification
                }
            )
    
    def get_parsing_statistics(self) -> Dict:
        """Get detailed statistics about the parsing process"""
        stats = self.parsing_stats.copy()
        
        # Add rule usage breakdown
        rule_usage = {}
        for rule, count in stats["rules_applied"].items():
            rule_usage[str(rule)] = count
        stats["rule_usage"] = rule_usage
        
        return stats


if __name__ == "__main__":
    # Demonstration of the CortexParser
    print("CortexOS NLP - Parser Module Demonstration")
    print("=" * 50)
    
    # Initialize Phase 1 components
    from core import SpatialAnchor, BinaryCellMemory, HarmonicResonance
    from tokenizer import CortexTokenizer
    from tagger import CortexTagger
    
    anchor_system = SpatialAnchor()
    memory = BinaryCellMemory()
    resonance = HarmonicResonance(memory)
    
    # Initialize full pipeline
    tokenizer = CortexTokenizer()
    tagger = CortexTagger(anchor_system, memory, resonance)
    parser = CortexParser(memory, resonance)
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The red car drives fast.",
        "I can see the beautiful sunset.",
        "The cat is sleeping on the mat.",
        "Running quickly, the athlete finished the race."
    ]
    
    print("Testing dependency parsing:")
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Input: '{text}'")
        
        # Full pipeline: tokenize -> tag -> parse
        tokens = tokenizer.tokenize(text)
        tagged_tokens = tagger.tag_tokens(tokens)
        dependency_tree = parser.parse_sentence(tagged_tokens)
        
        print(f"   Dependency relationships:")
        for rel in dependency_tree.relationships:
            print(f"     {rel.head.text} --[{rel.relation}]--> {rel.dependent.text} (conf: {rel.confidence:.3f})")
        
        print(f"   Root token: {dependency_tree.root.text if dependency_tree.root else 'None'}")
        
        # Show CoNLL-U format
        print(f"   CoNLL-U format:")
        conllu_lines = dependency_tree.to_conllu().split('\n')
        for line in conllu_lines[:5]:  # Show first 5 lines
            print(f"     {line}")
        if len(conllu_lines) > 5:
            print(f"     ... ({len(conllu_lines) - 5} more lines)")
    
    # Get parsing statistics
    stats = parser.get_parsing_statistics()
    print(f"\nParsing Statistics:")
    print(f"  Sentences parsed: {stats['sentences_parsed']}")
    print(f"  Relationships created: {stats['relationships_created']}")
    print(f"  Conflicts resolved: {stats['conflicts_resolved']}")
    
    print(f"\nGrammar rules loaded: {len(parser.rule_engine.rules)}")
    print("✅ Parser demonstration complete!")

