"""
CortexOS NLP - Span Class

The Span class represents a slice of a document with multiple tokens,
providing spaCy-compatible interface for working with token sequences.
"""

from typing import List, Optional, Any, Iterator, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .cortex_doc import Doc
    from .cortex_token import Token


class Span:
    """
    A slice of a Doc object containing multiple tokens.
    
    The Span class provides access to sequences of tokens with
    aggregate linguistic properties and analysis capabilities.
    
    Attributes:
        doc (Doc): The document this span belongs to
        start (int): Start token index (inclusive)
        end (int): End token index (exclusive)
        label (str): Optional label for the span
        label_ (str): String label for the span
    """
    
    def __init__(self, doc: 'Doc', start: int, end: int, label: str = None):
        """
        Initialize a Span object.
        
        Args:
            doc: The Doc object this span belongs to
            start: Start token index (inclusive)
            end: End token index (exclusive)
            label: Optional label for the span
        """
        self._doc = doc
        self._start = start
        self._end = end
        self._label = label or ""
        
        # Validate indices
        if start < 0 or end > len(doc) or start >= end:
            raise IndexError(f"Invalid span indices: [{start}:{end}] for doc of length {len(doc)}")
        
        # Cache for expensive operations
        self._root = None
        self._lefts = None
        self._rights = None
        self._subtree = None
    
    @property
    def doc(self) -> 'Doc':
        """The document this span belongs to."""
        return self._doc
    
    @property
    def start(self) -> int:
        """Start token index (inclusive)."""
        return self._start
    
    @property
    def end(self) -> int:
        """End token index (exclusive)."""
        return self._end
    
    @property
    def label(self) -> int:
        """Numeric label for the span (spaCy compatibility)."""
        # For now, return hash of string label
        return hash(self._label) if self._label else 0
    
    @property
    def label_(self) -> str:
        """String label for the span."""
        return self._label
    
    @property
    def text(self) -> str:
        """The text content of the span."""
        return "".join(token.text_with_ws for token in self)
    
    @property
    def text_with_ws(self) -> str:
        """The text content with trailing whitespace."""
        return self.text
    
    @property
    def lemma_(self) -> str:
        """Lemmatized form of the span."""
        return " ".join(token.lemma_ for token in self if not token.is_space)
    
    @property
    def pos_(self) -> str:
        """Most common POS tag in the span."""
        pos_tags = [token.pos_ for token in self if not token.is_space]
        if not pos_tags:
            return ""
        
        # Return most frequent POS tag
        from collections import Counter
        return Counter(pos_tags).most_common(1)[0][0]
    
    @property
    def tag_(self) -> str:
        """Most common detailed tag in the span."""
        tags = [token.tag_ for token in self if not token.is_space]
        if not tags:
            return ""
        
        # Return most frequent tag
        from collections import Counter
        return Counter(tags).most_common(1)[0][0]
    
    @property
    def dep_(self) -> str:
        """Dependency relation of the span's root."""
        return self.root.dep_
    
    @property
    def ent_type_(self) -> str:
        """Entity type (future implementation)."""
        return ""
    
    @property
    def ent_iob_(self) -> str:
        """IOB entity tag (future implementation)."""
        return ""
    
    @property
    def root(self) -> 'Token':
        """The syntactic root of the span."""
        if self._root is None:
            self._root = self._find_root()
        return self._root
    
    def _find_root(self) -> 'Token':
        """Find the syntactic root token of this span."""
        tokens = list(self)
        if not tokens:
            raise ValueError("Empty span has no root")
        
        if len(tokens) == 1:
            return tokens[0]
        
        # Find token whose head is outside the span or is itself
        for token in tokens:
            if token.head == token or token.head not in tokens:
                return token
        
        # Fallback: return first token
        return tokens[0]
    
    @property
    def lefts(self) -> List['Token']:
        """Tokens to the left of the span that depend on the span."""
        if self._lefts is None:
            self._lefts = self._find_lefts()
        return self._lefts
    
    def _find_lefts(self) -> List['Token']:
        """Find left-dependent tokens."""
        lefts = []
        span_tokens = set(self)
        
        for token in span_tokens:
            for child in token.children:
                if child.i < self.start and child not in span_tokens:
                    lefts.append(child)
        
        return sorted(lefts, key=lambda t: t.i)
    
    @property
    def rights(self) -> List['Token']:
        """Tokens to the right of the span that depend on the span."""
        if self._rights is None:
            self._rights = self._find_rights()
        return self._rights
    
    def _find_rights(self) -> List['Token']:
        """Find right-dependent tokens."""
        rights = []
        span_tokens = set(self)
        
        for token in span_tokens:
            for child in token.children:
                if child.i >= self.end and child not in span_tokens:
                    rights.append(child)
        
        return sorted(rights, key=lambda t: t.i)
    
    @property
    def subtree(self) -> 'Span':
        """The syntactic subtree of the span."""
        if self._subtree is None:
            self._subtree = self._find_subtree()
        return self._subtree
    
    def _find_subtree(self) -> 'Span':
        """Find the complete syntactic subtree."""
        # Collect all descendants of tokens in this span
        descendants = set(self)
        
        def collect_descendants(token):
            for child in token.children:
                if child not in descendants:
                    descendants.add(child)
                    collect_descendants(child)
        
        for token in self:
            collect_descendants(token)
        
        # Find span boundaries
        indices = [token.i for token in descendants]
        if not indices:
            return self
        
        start = min(indices)
        end = max(indices) + 1
        
        return Span(self._doc, start, end, label="SUBTREE")
    
    # Sequence methods
    def __len__(self) -> int:
        """Number of tokens in the span."""
        return self._end - self._start
    
    def __getitem__(self, key: Union[int, slice]) -> Union['Token', 'Span']:
        """Get token(s) by index or slice."""
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if not 0 <= key < len(self):
                raise IndexError(f"Span index {key} out of range")
            return self._doc.tokens[self._start + key]
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                raise ValueError("Step slicing not supported")
            new_start = self._start + start
            new_end = self._start + stop
            return Span(self._doc, new_start, new_end, label=self._label)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __iter__(self) -> Iterator['Token']:
        """Iterate over tokens in the span."""
        for i in range(self._start, self._end):
            yield self._doc.tokens[i]
    
    def __contains__(self, token: 'Token') -> bool:
        """Check if a token is in this span."""
        return self._start <= token.i < self._end
    
    # String methods
    def __str__(self) -> str:
        """String representation of the span."""
        return self.text.strip()
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        text_preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"Span('{text_preview}', [{self._start}:{self._end}])"
    
    def __eq__(self, other) -> bool:
        """Check equality with another span or string."""
        if isinstance(other, Span):
            return (self._doc == other._doc and 
                   self._start == other._start and 
                   self._end == other._end)
        elif isinstance(other, str):
            return self.text.strip() == other
        return False
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash((id(self._doc), self._start, self._end))
    
    # Boolean properties
    @property
    def is_alpha(self) -> bool:
        """Whether all tokens in the span are alphabetic."""
        return all(token.is_alpha for token in self if not token.is_space)
    
    @property
    def is_digit(self) -> bool:
        """Whether all tokens in the span are digits."""
        return all(token.is_digit for token in self if not token.is_space)
    
    @property
    def is_punct(self) -> bool:
        """Whether all tokens in the span are punctuation."""
        return all(token.is_punct for token in self if not token.is_space)
    
    @property
    def is_space(self) -> bool:
        """Whether all tokens in the span are whitespace."""
        return all(token.is_space for token in self)
    
    @property
    def is_stop(self) -> bool:
        """Whether all tokens in the span are stop words."""
        return all(token.is_stop for token in self if not token.is_space)
    
    @property
    def is_sent_start(self) -> bool:
        """Whether this span starts a sentence."""
        if len(self) == 0:
            return False
        return self[0].is_sent_start
    
    @property
    def is_sent_end(self) -> bool:
        """Whether this span ends a sentence."""
        if len(self) == 0:
            return False
        return self[-1].is_sent_end
    
    @property
    def is_title(self) -> bool:
        """Whether the span text is title-cased."""
        return self.text.strip().istitle()
    
    @property
    def is_upper(self) -> bool:
        """Whether the span text is uppercase."""
        return self.text.strip().isupper()
    
    @property
    def is_lower(self) -> bool:
        """Whether the span text is lowercase."""
        return self.text.strip().islower()
    
    @property
    def is_ascii(self) -> bool:
        """Whether all characters in the span are ASCII."""
        return all(ord(char) < 128 for char in self.text)
    
    # Analysis methods
    def similarity(self, other: Union['Span', 'Token', str]) -> float:
        """
        Calculate similarity with another span, token, or string.
        
        Args:
            other: Object to compare with
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if isinstance(other, str):
            # Simple text similarity
            return 1.0 if self.text.strip().lower() == other.lower() else 0.0
        elif hasattr(other, 'text'):
            # Compare text content
            return 1.0 if self.text.strip().lower() == other.text.strip().lower() else 0.0
        else:
            return 0.0
    
    def char_span(self, start: int, end: int, label: str = None) -> Optional['Span']:
        """
        Create a sub-span based on character indices within this span.
        
        Args:
            start: Start character index within this span
            end: End character index within this span
            label: Optional label for the new span
            
        Returns:
            New Span object or None if indices are invalid
        """
        # Convert character indices to token indices
        char_pos = 0
        token_start = None
        token_end = None
        
        for i, token in enumerate(self):
            token_char_start = char_pos
            token_char_end = char_pos + len(token.text)
            
            if token_start is None and token_char_start >= start:
                token_start = i
            
            if token_char_end <= end:
                token_end = i + 1
            
            char_pos = token_char_end
        
        if token_start is not None and token_end is not None and token_start < token_end:
            abs_start = self._start + token_start
            abs_end = self._start + token_end
            return Span(self._doc, abs_start, abs_end, label=label)
        
        return None
    
    def merge(self, *args, **kwargs) -> 'Token':
        """
        Merge the span into a single token (spaCy compatibility).
        Currently not implemented - returns the root token.
        """
        return self.root
    
    def as_doc(self) -> 'Doc':
        """
        Create a new Doc object from this span (spaCy compatibility).
        Currently not implemented - returns the original doc.
        """
        return self._doc
    
    # Extension methods (spaCy compatibility)
    def get_extension(self, name: str) -> Any:
        """
        Get custom extension attribute.
        
        Args:
            name: Name of the extension attribute
            
        Returns:
            Extension attribute value
        """
        return getattr(self, f"_{name}", None)
    
    def set_extension(self, name: str, value: Any):
        """
        Set custom extension attribute.
        
        Args:
            name: Name of the extension attribute
            value: Value to set
        """
        setattr(self, f"_{name}", value)
    
    def has_extension(self, name: str) -> bool:
        """
        Check if extension attribute exists.
        
        Args:
            name: Name of the extension attribute
            
        Returns:
            True if the extension exists
        """
        return hasattr(self, f"_{name}")
    
    # CortexOS-specific methods
    @property
    def confidence_scores(self) -> dict:
        """Get confidence scores for tokens in this span."""
        scores = {
            'pos_confidence': [],
            'certainty_levels': []
        }
        
        for token in self:
            if not token.is_space:
                scores['pos_confidence'].append(token.pos_confidence)
                scores['certainty_levels'].append(token.certainty_level)
        
        if scores['pos_confidence']:
            scores['avg_pos_confidence'] = sum(scores['pos_confidence']) / len(scores['pos_confidence'])
        else:
            scores['avg_pos_confidence'] = 0.0
        
        return scores
    
    @property
    def is_deterministic(self) -> bool:
        """Whether this span was processed deterministically."""
        return True  # CortexOS is always deterministic
    
    def explain_structure(self) -> str:
        """
        Get explanation of the span's syntactic structure.
        
        Returns:
            Human-readable explanation
        """
        lines = []
        lines.append(f"Span: '{self.text.strip()}'")
        lines.append(f"Tokens: {len(self)}")
        lines.append(f"Root: '{self.root.text}' ({self.root.pos_})")
        
        if self.lefts:
            lines.append(f"Left dependencies: {[t.text for t in self.lefts]}")
        
        if self.rights:
            lines.append(f"Right dependencies: {[t.text for t in self.rights]}")
        
        return "\n".join(lines)
    
    def get_pos_distribution(self) -> dict:
        """
        Get distribution of POS tags in this span.
        
        Returns:
            Dictionary mapping POS tags to counts
        """
        from collections import Counter
        pos_tags = [token.pos_ for token in self if not token.is_space]
        return dict(Counter(pos_tags))
    
    def to_dict(self) -> dict:
        """
        Convert span to dictionary representation.
        
        Returns:
            Dictionary containing span properties
        """
        return {
            'text': self.text.strip(),
            'start': self.start,
            'end': self.end,
            'label': self.label_,
            'length': len(self),
            'root_text': self.root.text,
            'root_pos': self.root.pos_,
            'pos_distribution': self.get_pos_distribution(),
            'confidence_scores': self.confidence_scores,
            'is_deterministic': self.is_deterministic
        }
    
    def tokens_list(self) -> List['Token']:
        """
        Get list of all tokens in this span.
        
        Returns:
            List of Token objects
        """
        return list(self)
    
    def content_tokens(self) -> List['Token']:
        """
        Get list of content tokens (non-space) in this span.
        
        Returns:
            List of non-space Token objects
        """
        return [token for token in self if not token.is_space]

