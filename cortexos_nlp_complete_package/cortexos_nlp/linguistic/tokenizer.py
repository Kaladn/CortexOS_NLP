"""
CortexOS NLP - Tokenizer Module
Phase 2: Linguistic Layer

This module implements the deterministic tokenizer that breaks down raw text
into individual tokens and immediately assigns each token its spatial anchor
coordinates. This replaces probabilistic tokenization with mathematical certainty.

Core Principle: Every token gets a permanent, deterministic position in 6D space
upon creation, enabling mathematical operations on linguistic elements.
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

try:
    from ..core import SpatialAnchor, SpatialCoordinate
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from core import SpatialAnchor, SpatialCoordinate


@dataclass
class Token:
    """
    Represents a single token with its text and spatial coordinates.
    This is the fundamental unit of deterministic language processing.
    """
    text: str                    # Original text of the token
    normalized: str             # Normalized form for spatial anchoring
    spatial_coord: SpatialCoordinate  # 6D coordinates in mathematical space
    start_char: int             # Starting character position in original text
    end_char: int               # Ending character position in original text
    token_id: int               # Sequential ID within the document
    is_alpha: bool              # Whether token contains alphabetic characters
    is_digit: bool              # Whether token contains digits
    is_punct: bool              # Whether token is punctuation
    is_space: bool              # Whether token is whitespace
    is_stop: bool               # Whether token is a stop word (determined later)
    
    def __str__(self) -> str:
        return f"Token('{self.text}', {self.spatial_coord})"
    
    def __repr__(self) -> str:
        return self.__str__()


class CortexTokenizer:
    """
    The deterministic tokenizer that converts raw text into spatially-anchored tokens.
    
    Unlike probabilistic tokenizers, this system:
    1. Uses rule-based tokenization for consistency
    2. Immediately assigns spatial coordinates to each token
    3. Provides mathematical certainty in token boundaries
    4. Enables deterministic reconstruction of original text
    """
    
    def __init__(self, 
                 preserve_case: bool = False,
                 handle_contractions: bool = True,
                 split_hyphenated: bool = False):
        """
        Initialize the CortexOS tokenizer.
        
        Args:
            preserve_case: Whether to preserve original case in spatial anchoring
            handle_contractions: Whether to split contractions (don't -> do n't)
            split_hyphenated: Whether to split hyphenated words
        """
        self.preserve_case = preserve_case
        self.handle_contractions = handle_contractions
        self.split_hyphenated = split_hyphenated
        
        # Initialize spatial anchor system
        self.anchor_system = SpatialAnchor()
        
        # Tokenization patterns
        self._setup_patterns()
        
        # Common stop words for classification
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
    
    def _setup_patterns(self):
        """Setup regex patterns for tokenization"""
        
        # Contraction patterns
        self.contraction_patterns = [
            (r"won't", "will not"),
            (r"can't", "can not"),
            (r"n't", " not"),
            (r"'re", " are"),
            (r"'ve", " have"),
            (r"'ll", " will"),
            (r"'d", " would"),
            (r"'m", " am"),
        ] if self.handle_contractions else []
        
        # Main tokenization pattern
        # This pattern captures:
        # 1. Words (including apostrophes if not handling contractions)
        # 2. Numbers (including decimals)
        # 3. Punctuation
        # 4. Whitespace
        if self.handle_contractions:
            word_pattern = r"[a-zA-Z]+"
        else:
            word_pattern = r"[a-zA-Z']+(?:[a-zA-Z]+)?"
        
        if self.split_hyphenated:
            hyphen_pattern = r""
        else:
            hyphen_pattern = r"|[a-zA-Z]+-[a-zA-Z]+"
        
        self.token_pattern = re.compile(
            rf"({word_pattern}{hyphen_pattern}|"  # Words and hyphenated words
            r"\d+\.?\d*|"                         # Numbers (including decimals)
            r"[^\w\s]|"                          # Punctuation
            r"\s+)"                              # Whitespace
        )
    
    def _preprocess_contractions(self, text: str) -> str:
        """
        Preprocess contractions before tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Text with contractions expanded
        """
        if not self.handle_contractions:
            return text
        
        processed_text = text
        for pattern, replacement in self.contraction_patterns:
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def _classify_token(self, text: str) -> Tuple[bool, bool, bool, bool, bool]:
        """
        Classify token characteristics.
        
        Args:
            text: Token text
            
        Returns:
            Tuple of (is_alpha, is_digit, is_punct, is_space, is_stop)
        """
        is_alpha = bool(re.search(r'[a-zA-Z]', text))
        is_digit = bool(re.search(r'\d', text))
        is_punct = bool(re.search(r'[^\w\s]', text)) and not is_alpha and not is_digit
        is_space = text.isspace()
        is_stop = text.lower() in self.stop_words
        
        return is_alpha, is_digit, is_punct, is_space, is_stop
    
    def _normalize_for_anchoring(self, text: str) -> str:
        """
        Normalize token text for spatial anchoring.
        
        Args:
            text: Original token text
            
        Returns:
            Normalized text for consistent spatial coordinates
        """
        if self.preserve_case:
            return text.strip()
        else:
            return text.lower().strip()
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize input text into spatially-anchored tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of Token objects with spatial coordinates
        """
        if not text:
            return []
        
        # Preprocess contractions
        processed_text = self._preprocess_contractions(text)
        
        # Find all tokens using regex
        matches = list(self.token_pattern.finditer(processed_text))
        
        tokens = []
        for token_id, match in enumerate(matches):
            token_text = match.group(1)
            start_char = match.start(1)
            end_char = match.end(1)
            
            # Skip empty tokens
            if not token_text:
                continue
            
            # Normalize for spatial anchoring
            normalized = self._normalize_for_anchoring(token_text)
            
            # Get spatial coordinates
            if normalized:  # Only anchor non-empty normalized text
                _, spatial_coord = self.anchor_system.create_anchor(normalized)
            else:
                # For empty normalized text (like pure whitespace), use original
                _, spatial_coord = self.anchor_system.create_anchor(token_text)
            
            # Classify token
            is_alpha, is_digit, is_punct, is_space, is_stop = self._classify_token(token_text)
            
            # Create token
            token = Token(
                text=token_text,
                normalized=normalized,
                spatial_coord=spatial_coord,
                start_char=start_char,
                end_char=end_char,
                token_id=token_id,
                is_alpha=is_alpha,
                is_digit=is_digit,
                is_punct=is_punct,
                is_space=is_space,
                is_stop=is_stop
            )
            
            tokens.append(token)
        
        return tokens
    
    def detokenize(self, tokens: List[Token]) -> str:
        """
        Reconstruct original text from tokens.
        
        Args:
            tokens: List of tokens to reconstruct
            
        Returns:
            Reconstructed text
        """
        if not tokens:
            return ""
        
        # Sort tokens by their original position
        sorted_tokens = sorted(tokens, key=lambda t: t.start_char)
        
        # Reconstruct text
        result = ""
        for token in sorted_tokens:
            result += token.text
        
        return result
    
    def get_token_statistics(self, tokens: List[Token]) -> Dict:
        """
        Get statistics about the tokenized text.
        
        Args:
            tokens: List of tokens to analyze
            
        Returns:
            Dictionary with tokenization statistics
        """
        if not tokens:
            return {
                "total_tokens": 0,
                "alpha_tokens": 0,
                "digit_tokens": 0,
                "punct_tokens": 0,
                "space_tokens": 0,
                "stop_tokens": 0,
                "unique_tokens": 0,
                "unique_normalized": 0
            }
        
        alpha_count = sum(1 for t in tokens if t.is_alpha)
        digit_count = sum(1 for t in tokens if t.is_digit)
        punct_count = sum(1 for t in tokens if t.is_punct)
        space_count = sum(1 for t in tokens if t.is_space)
        stop_count = sum(1 for t in tokens if t.is_stop)
        
        unique_texts = set(t.text for t in tokens)
        unique_normalized = set(t.normalized for t in tokens)
        
        return {
            "total_tokens": len(tokens),
            "alpha_tokens": alpha_count,
            "digit_tokens": digit_count,
            "punct_tokens": punct_count,
            "space_tokens": space_count,
            "stop_tokens": stop_count,
            "unique_tokens": len(unique_texts),
            "unique_normalized": len(unique_normalized)
        }
    
    def filter_tokens(self, 
                     tokens: List[Token],
                     include_alpha: bool = True,
                     include_digits: bool = True,
                     include_punct: bool = False,
                     include_space: bool = False,
                     exclude_stop: bool = False) -> List[Token]:
        """
        Filter tokens based on their characteristics.
        
        Args:
            tokens: List of tokens to filter
            include_alpha: Include alphabetic tokens
            include_digits: Include digit tokens
            include_punct: Include punctuation tokens
            include_space: Include whitespace tokens
            exclude_stop: Exclude stop words
            
        Returns:
            Filtered list of tokens
        """
        filtered = []
        
        for token in tokens:
            # Check inclusion criteria
            should_include = False
            
            if include_alpha and token.is_alpha:
                should_include = True
            if include_digits and token.is_digit:
                should_include = True
            if include_punct and token.is_punct:
                should_include = True
            if include_space and token.is_space:
                should_include = True
            
            # Check exclusion criteria
            if exclude_stop and token.is_stop:
                should_include = False
            
            if should_include:
                filtered.append(token)
        
        return filtered


if __name__ == "__main__":
    # Demonstration of the CortexTokenizer
    print("CortexOS NLP - Tokenizer Module Demonstration")
    print("=" * 50)
    
    # Initialize tokenizer
    tokenizer = CortexTokenizer(
        preserve_case=False,
        handle_contractions=True,
        split_hyphenated=False
    )
    
    # Test texts
    test_texts = [
        "Hello, world! This is a test.",
        "I can't believe it's working so well.",
        "The quick brown fox jumps over the lazy dog.",
        "Numbers: 123, 45.67, and symbols: @#$%",
        "Hyphenated-words and contractions don't break it."
    ]
    
    print("Testing tokenization:")
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Input: '{text}'")
        tokens = tokenizer.tokenize(text)
        
        print(f"   Tokens ({len(tokens)}):")
        for token in tokens:
            if not token.is_space:  # Skip whitespace for cleaner output
                print(f"     '{token.text}' -> {token.spatial_coord}")
        
        # Test detokenization
        reconstructed = tokenizer.detokenize(tokens)
        print(f"   Reconstructed: '{reconstructed}'")
        print(f"   Perfect reconstruction: {text == reconstructed}")
        
        # Get statistics
        stats = tokenizer.get_token_statistics(tokens)
        print(f"   Stats: {stats['alpha_tokens']} alpha, {stats['punct_tokens']} punct, {stats['stop_tokens']} stop")
    
    print("\nTesting token filtering:")
    text = "The quick brown fox jumps over the lazy dog!"
    tokens = tokenizer.tokenize(text)
    
    # Filter out stop words and punctuation
    filtered = tokenizer.filter_tokens(
        tokens, 
        include_alpha=True, 
        include_punct=False, 
        exclude_stop=True
    )
    
    print(f"Original: {[t.text for t in tokens if not t.is_space]}")
    print(f"Filtered: {[t.text for t in filtered]}")
    
    print(f"\nTokenizer cache size: {tokenizer.anchor_system.cache_size()}")
    print("✅ Tokenizer demonstration complete!")

