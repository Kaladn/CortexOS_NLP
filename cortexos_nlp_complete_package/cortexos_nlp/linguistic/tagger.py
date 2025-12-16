"""
CortexOS NLP - Tagger Module
Phase 2: Linguistic Layer

This module implements the deterministic part-of-speech tagger that assigns
grammatical roles to tokens using mathematical certainty instead of probabilistic
models. Every POS tag assignment is traceable and perfectly repeatable.

Core Principle: Use spatial coordinates and harmonic resonance to determine
POS tags through mathematical analysis rather than statistical guessing.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
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


@dataclass
class POSTagResult:
    """
    Result of POS tag assignment with mathematical justification.
    Provides complete traceability for every tagging decision.
    """
    tag: str                    # The assigned POS tag
    confidence: float           # Mathematical confidence score (0.0-1.0)
    method: str                 # Method used: dictionary_lookup, context_analysis, morphological_analysis
    justification: str          # Human-readable explanation of the decision
    alternatives: List[Tuple[str, float]] = None  # Alternative tags with scores


class MorphologyRule:
    """
    Represents a morphological rule for unknown word analysis.
    Uses deterministic patterns instead of probabilistic models.
    """
    
    def __init__(self, pattern: str, pos_tag: str, confidence: float, description: str):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.pos_tag = pos_tag
        self.confidence = confidence
        self.description = description
        self.usage_count = 0
    
    def matches(self, word: str) -> bool:
        """Check if this rule matches the given word"""
        return bool(self.pattern.match(word))
    
    def apply(self, word: str) -> POSTagResult:
        """Apply this rule to get POS tag result"""
        self.usage_count += 1
        return POSTagResult(
            tag=self.pos_tag,
            confidence=self.confidence,
            method="morphological_analysis",
            justification=f"Morphological rule: {self.description} (pattern: {self.pattern.pattern})"
        )


class SpatialPOSDictionary:
    """
    Maps words to their possible POS tags using spatial coordinates.
    Replaces probabilistic POS models with deterministic lookup + context analysis.
    """
    
    def __init__(self, anchor_system: SpatialAnchor, memory: BinaryCellMemory):
        self.anchor_system = anchor_system
        self.memory = memory
        
        # Core mappings
        self.word_pos_map: Dict[str, List[str]] = {}
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
            "that": ["DT"], "these": ["DT"], "those": ["DT"], "some": ["DT"],
            "any": ["DT"], "each": ["DT"], "every": ["DT"], "all": ["DT"],
            "both": ["DT"], "either": ["DT"], "neither": ["DT"],
            
            # Pronouns - always PRP
            "i": ["PRP"], "you": ["PRP"], "he": ["PRP"], "she": ["PRP"],
            "it": ["PRP"], "we": ["PRP"], "they": ["PRP"], "me": ["PRP"],
            "him": ["PRP"], "her": ["PRP"], "us": ["PRP"], "them": ["PRP"],
            
            # Possessive pronouns
            "my": ["PRP$"], "your": ["PRP$"], "his": ["PRP$"], "her": ["PRP$"],
            "its": ["PRP$"], "our": ["PRP$"], "their": ["PRP$"],
            
            # Prepositions - always IN
            "in": ["IN"], "on": ["IN"], "at": ["IN"], "by": ["IN"],
            "for": ["IN"], "with": ["IN"], "from": ["IN"], "to": ["IN"],
            "of": ["IN"], "about": ["IN"], "into": ["IN"], "through": ["IN"],
            "during": ["IN"], "before": ["IN"], "after": ["IN"], "above": ["IN"],
            "below": ["IN"], "up": ["IN"], "down": ["IN"], "out": ["IN"],
            "off": ["IN"], "over": ["IN"], "under": ["IN"], "between": ["IN"],
            
            # Conjunctions - always CC
            "and": ["CC"], "or": ["CC"], "but": ["CC"], "nor": ["CC"],
            "for": ["CC"], "yet": ["CC"], "so": ["CC"],
            
            # Common verbs
            "is": ["VBZ"], "are": ["VBP"], "was": ["VBD"], "were": ["VBD"],
            "be": ["VB"], "been": ["VBN"], "being": ["VBG"],
            "have": ["VBP"], "has": ["VBZ"], "had": ["VBD"],
            "do": ["VBP"], "does": ["VBZ"], "did": ["VBD"],
            "will": ["MD"], "would": ["MD"], "can": ["MD"], "could": ["MD"],
            "should": ["MD"], "shall": ["MD"], "may": ["MD"], "might": ["MD"],
            "must": ["MD"], "ought": ["MD"],
            
            # Common nouns
            "time": ["NN"], "person": ["NN"], "year": ["NN"], "way": ["NN"],
            "day": ["NN"], "thing": ["NN"], "man": ["NN"], "world": ["NN"],
            "life": ["NN"], "hand": ["NN"], "part": ["NN"], "child": ["NN"],
            "eye": ["NN"], "woman": ["NN"], "place": ["NN"], "work": ["NN"],
            "week": ["NN"], "case": ["NN"], "point": ["NN"], "government": ["NN"],
            "company": ["NN"], "number": ["NN"], "group": ["NN"], "problem": ["NN"],
            "fact": ["NN"], "home": ["NN"], "water": ["NN"], "room": ["NN"],
            "mother": ["NN"], "area": ["NN"], "money": ["NN"], "story": ["NN"],
            
            # Common adjectives
            "good": ["JJ"], "new": ["JJ"], "first": ["JJ"], "last": ["JJ"],
            "long": ["JJ"], "great": ["JJ"], "little": ["JJ"], "own": ["JJ"],
            "other": ["JJ"], "old": ["JJ"], "right": ["JJ"], "big": ["JJ"],
            "high": ["JJ"], "different": ["JJ"], "small": ["JJ"], "large": ["JJ"],
            "next": ["JJ"], "early": ["JJ"], "young": ["JJ"], "important": ["JJ"],
            "few": ["JJ"], "public": ["JJ"], "bad": ["JJ"], "same": ["JJ"],
            "able": ["JJ"], "red": ["JJ"], "blue": ["JJ"], "white": ["JJ"],
            "black": ["JJ"], "green": ["JJ"], "yellow": ["JJ"], "brown": ["JJ"],
            
            # Common adverbs
            "not": ["RB"], "so": ["RB"], "out": ["RB"], "just": ["RB"],
            "now": ["RB"], "how": ["RB"], "then": ["RB"], "more": ["RB"],
            "also": ["RB"], "here": ["RB"], "well": ["RB"], "only": ["RB"],
            "very": ["RB"], "even": ["RB"], "back": ["RB"], "there": ["RB"],
            "down": ["RB"], "still": ["RB"], "in": ["RB"], "as": ["RB"],
            "too": ["RB"], "when": ["RB"], "never": ["RB"], "really": ["RB"],
            "most": ["RB"], "on": ["RB"], "why": ["RB"], "what": ["RB"],
            "up": ["RB"], "off": ["RB"], "again": ["RB"], "where": ["RB"],
            
            # Ambiguous words requiring context analysis
            "book": ["NN", "VB"],      # noun or verb
            "run": ["NN", "VB", "VBP"], # noun or verb
            "fast": ["JJ", "RB"],      # adjective or adverb
            "well": ["RB", "JJ", "NN"], # adverb, adjective, or noun
            "can": ["MD", "NN", "VB"], # modal, noun, or verb
            "will": ["MD", "NN", "VB"], # modal, noun, or verb
            "work": ["NN", "VB"],      # noun or verb
            "play": ["NN", "VB"],      # noun or verb
            "love": ["NN", "VB"],      # noun or verb
            "help": ["NN", "VB"],      # noun or verb
            "call": ["NN", "VB"],      # noun or verb
            "try": ["NN", "VB"],       # noun or verb
            "turn": ["NN", "VB"],      # noun or verb
            "move": ["NN", "VB"],      # noun or verb
            "live": ["JJ", "VB"],      # adjective or verb
            "close": ["JJ", "VB", "RB"], # adjective, verb, or adverb
            "open": ["JJ", "VB"],      # adjective or verb
            "clean": ["JJ", "VB"],     # adjective or verb
            "clear": ["JJ", "VB"],     # adjective or verb
            "free": ["JJ", "VB"],      # adjective or verb
            "light": ["NN", "JJ", "VB"], # noun, adjective, or verb
            "right": ["NN", "JJ", "RB"], # noun, adjective, or adverb
            "left": ["NN", "JJ", "VBD"], # noun, adjective, or verb (past)
            "back": ["NN", "JJ", "RB", "VB"], # noun, adjective, adverb, or verb
            "round": ["NN", "JJ", "RB", "VB"], # noun, adjective, adverb, or verb
        }
        
        self.word_pos_map = base_dictionary
    
    def _create_pos_anchors(self):
        """Create spatial anchors for each POS tag"""
        pos_tags = [
            # Nouns
            "NN", "NNS", "NNP", "NNPS",
            # Verbs
            "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
            # Adjectives
            "JJ", "JJR", "JJS",
            # Adverbs
            "RB", "RBR", "RBS",
            # Pronouns
            "PRP", "PRP$", "WP", "WP$",
            # Determiners
            "DT", "WDT",
            # Prepositions
            "IN",
            # Conjunctions
            "CC",
            # Modals
            "MD",
            # Numbers
            "CD",
            # Interjections
            "UH",
            # Punctuation and symbols
            "PUNCT", "SYM",
            # Special tokens
            "SPACE"
        ]
        
        for pos_tag in pos_tags:
            _, coord = self.anchor_system.create_anchor(f"POS_{pos_tag}")
            self.pos_anchors[pos_tag] = coord
    
    def get_pos_candidates(self, word: str) -> List[str]:
        """Get possible POS tags for a word"""
        normalized_word = word.lower()
        return self.word_pos_map.get(normalized_word, [])
    
    def add_word_pos_mapping(self, word: str, pos_tags: List[str]):
        """Add new word-POS mapping to dictionary"""
        normalized_word = word.lower()
        if normalized_word in self.word_pos_map:
            # Merge with existing tags
            existing_tags = set(self.word_pos_map[normalized_word])
            new_tags = set(pos_tags)
            self.word_pos_map[normalized_word] = list(existing_tags.union(new_tags))
        else:
            self.word_pos_map[normalized_word] = pos_tags


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
            # Unknown word - will be handled by morphological analysis
            return None
        
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
                    if hasattr(ctx_token, 'pos_tag') and ctx_token.pos_tag and ctx_token.pos_tag in self.pos_dict.pos_anchors:
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
                    if hasattr(ctx_token, 'pos_tag') and ctx_token.pos_tag and ctx_token.pos_tag in self.pos_dict.pos_anchors:
                        ctx_pos_coord = self.pos_dict.pos_anchors[ctx_token.pos_tag]
                        resonance_result = self.resonance.calculate_resonance(
                            pos_coord, ctx_pos_coord
                        )
                        right_score += resonance_result.similarity_score
                right_score /= len(right_context)
            
            # Combined context score
            total_score = (left_score + right_score) / 2 if (left_context or right_context) else 0.5
            candidate_scores.append((pos_tag, total_score))
        
        # Sort by score and select best
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_tag, best_score = candidate_scores[0]
        
        return POSTagResult(
            tag=best_tag,
            confidence=best_score,
            method="context_analysis",
            justification=f"Harmonic resonance analysis selected '{best_tag}' with score {best_score:.4f}",
            alternatives=candidate_scores[1:3]  # Include top 2 alternatives
        )
    
    def _get_left_context(self, target_token: Token, context: List[Token]) -> List[Token]:
        """Get tokens to the left of target token"""
        target_idx = None
        for i, token in enumerate(context):
            if token == target_token:
                target_idx = i
                break
        
        if target_idx is None:
            return []
        
        start_idx = max(0, target_idx - self.context_window)
        return [t for t in context[start_idx:target_idx] if not t.is_space]
    
    def _get_right_context(self, target_token: Token, context: List[Token]) -> List[Token]:
        """Get tokens to the right of target token"""
        target_idx = None
        for i, token in enumerate(context):
            if token == target_token:
                target_idx = i
                break
        
        if target_idx is None:
            return []
        
        end_idx = min(len(context), target_idx + self.context_window + 1)
        return [t for t in context[target_idx + 1:end_idx] if not t.is_space]


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
            MorphologyRule(r".*ing$", "VBG", 0.9, "Present participle (-ing)"),
            MorphologyRule(r".*ed$", "VBD", 0.8, "Past tense/participle (-ed)"),
            MorphologyRule(r".*s$", "VBZ", 0.6, "Third person singular (-s)"),
            
            # Noun suffixes
            MorphologyRule(r".*tion$", "NN", 0.95, "Abstract noun (-tion)"),
            MorphologyRule(r".*sion$", "NN", 0.95, "Abstract noun (-sion)"),
            MorphologyRule(r".*ness$", "NN", 0.95, "Quality noun (-ness)"),
            MorphologyRule(r".*ment$", "NN", 0.9, "Action noun (-ment)"),
            MorphologyRule(r".*ity$", "NN", 0.9, "Quality noun (-ity)"),
            MorphologyRule(r".*er$", "NN", 0.7, "Agent noun (-er)"),
            MorphologyRule(r".*or$", "NN", 0.7, "Agent noun (-or)"),
            MorphologyRule(r".*ist$", "NN", 0.8, "Person noun (-ist)"),
            MorphologyRule(r".*ism$", "NN", 0.8, "Doctrine noun (-ism)"),
            MorphologyRule(r".*s$", "NNS", 0.6, "Plural noun (-s)"),
            
            # Adjective suffixes
            MorphologyRule(r".*ful$", "JJ", 0.9, "Full of quality (-ful)"),
            MorphologyRule(r".*less$", "JJ", 0.9, "Without quality (-less)"),
            MorphologyRule(r".*able$", "JJ", 0.85, "Capable of (-able)"),
            MorphologyRule(r".*ible$", "JJ", 0.85, "Capable of (-ible)"),
            MorphologyRule(r".*ive$", "JJ", 0.8, "Having quality (-ive)"),
            MorphologyRule(r".*ous$", "JJ", 0.8, "Having quality (-ous)"),
            MorphologyRule(r".*al$", "JJ", 0.7, "Related to (-al)"),
            MorphologyRule(r".*ic$", "JJ", 0.7, "Related to (-ic)"),
            MorphologyRule(r".*ical$", "JJ", 0.8, "Related to (-ical)"),
            
            # Adverb suffixes
            MorphologyRule(r".*ly$", "RB", 0.9, "Manner adverb (-ly)"),
            MorphologyRule(r".*ward$", "RB", 0.8, "Direction adverb (-ward)"),
            MorphologyRule(r".*wise$", "RB", 0.8, "Manner adverb (-wise)"),
        ]
    
    def _load_prefix_rules(self) -> List[MorphologyRule]:
        """Load prefix-based POS rules"""
        return [
            # Verb prefixes
            MorphologyRule(r"^re.*", "VB", 0.6, "Re- prefix (verb)"),
            MorphologyRule(r"^un.*", "VB", 0.5, "Un- prefix (verb)"),
            MorphologyRule(r"^pre.*", "VB", 0.5, "Pre- prefix (verb)"),
            
            # Adjective prefixes
            MorphologyRule(r"^un.*", "JJ", 0.6, "Un- prefix (adjective)"),
            MorphologyRule(r"^non.*", "JJ", 0.7, "Non- prefix (adjective)"),
            MorphologyRule(r"^anti.*", "JJ", 0.7, "Anti- prefix (adjective)"),
            MorphologyRule(r"^pre.*", "JJ", 0.5, "Pre- prefix (adjective)"),
            MorphologyRule(r"^post.*", "JJ", 0.5, "Post- prefix (adjective)"),
        ]
    
    def _load_capitalization_rules(self) -> List[MorphologyRule]:
        """Load capitalization-based POS rules"""
        return [
            MorphologyRule(r"^[A-Z][a-z]+$", "NNP", 0.8, "Capitalized word (proper noun)"),
            MorphologyRule(r"^[A-Z]+$", "NNP", 0.9, "All caps (proper noun/acronym)"),
        ]
    
    def analyze_unknown_word(self, word: str, context: List[Token]) -> POSTagResult:
        """Analyze unknown word using morphological patterns"""
        
        # Try suffix rules first (most reliable)
        for rule in self.suffix_rules:
            if rule.matches(word):
                return rule.apply(word)
        
        # Try prefix rules
        for rule in self.prefix_rules:
            if rule.matches(word):
                return rule.apply(word)
        
        # Try capitalization rules
        for rule in self.capitalization_rules:
            if rule.matches(word):
                # Check if it's at sentence start
                if not self._is_sentence_start(word, context):
                    return rule.apply(word)
        
        # Default to noun for unknown words
        return POSTagResult(
            tag="NN",
            confidence=0.4,
            method="morphological_analysis",
            justification="Default classification for unknown word (noun is most common)"
        )
    
    def _is_sentence_start(self, word: str, context: List[Token]) -> bool:
        """Check if word is at the start of a sentence"""
        # Simple heuristic: if previous non-space token ends with sentence-ending punctuation
        if not context:
            return True
        
        prev_content_tokens = [t for t in context if not t.is_space]
        if not prev_content_tokens:
            return True
        
        last_token = prev_content_tokens[-1]
        return last_token.text in '.!?'


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
            # Skip whitespace and handle punctuation
            if token.is_space:
                token.pos_tag = "SPACE"
                tagged_tokens.append(token)
                continue
            
            if token.is_punct:
                token.pos_tag = "PUNCT"
                tagged_tokens.append(token)
                continue
            
            # Get context window
            context = tokens  # Full context for now
            
            # Determine POS tag
            pos_result = self.disambiguator.disambiguate_pos(token, context)
            
            # If no result from disambiguator, use morphological analysis
            if pos_result is None:
                pos_result = self.morphology.analyze_unknown_word(token.text, context)
            
            # Assign POS tag to token
            token.pos_tag = pos_result.tag
            token.pos_confidence = pos_result.confidence
            token.pos_method = pos_result.method
            
            # Store word-POS relationship in memory
            if pos_result.tag in self.pos_dict.pos_anchors:
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


if __name__ == "__main__":
    # Demonstration of the CortexTagger
    print("CortexOS NLP - Tagger Module Demonstration")
    print("=" * 50)
    
    # Initialize Phase 1 components
    from core import SpatialAnchor, BinaryCellMemory, HarmonicResonance
    from tokenizer import CortexTokenizer
    
    anchor_system = SpatialAnchor()
    memory = BinaryCellMemory()
    resonance = HarmonicResonance(memory)
    
    # Initialize tokenizer and tagger
    tokenizer = CortexTokenizer()
    tagger = CortexTagger(anchor_system, memory, resonance)
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe it's working so well!",
        "Complex sentences with subordinate clauses are challenging.",
        "The beautiful red car drives very fast down the winding road.",
        "Running quickly, the athlete finished the race successfully."
    ]
    
    print("Testing POS tagging:")
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Input: '{text}'")
        
        # Tokenize first
        tokens = tokenizer.tokenize(text)
        
        # Then tag
        tagged_tokens = tagger.tag_tokens(tokens)
        
        print(f"   Tagged tokens:")
        for token in tagged_tokens:
            if not token.is_space:  # Skip whitespace for cleaner output
                confidence = getattr(token, 'pos_confidence', 0.0)
                method = getattr(token, 'pos_method', 'unknown')
                print(f"     '{token.text}' -> {token.pos_tag} (conf: {confidence:.3f}, method: {method})")
    
    # Get tagging statistics
    stats = tagger.get_tagging_statistics()
    print(f"\nTagging Statistics:")
    print(f"  Total tokens processed: {stats['total_tokens']}")
    print(f"  Dictionary lookups: {stats['dictionary_lookups']} ({stats.get('dictionary_percentage', 0):.1f}%)")
    print(f"  Context disambiguations: {stats['context_disambiguations']} ({stats.get('disambiguation_percentage', 0):.1f}%)")
    print(f"  Morphological analyses: {stats['morphological_analyses']} ({stats.get('morphology_percentage', 0):.1f}%)")
    
    print(f"\nMemory relationships stored: {len(memory.get_all_relationships())}")
    print("✅ Tagger demonstration complete!")

