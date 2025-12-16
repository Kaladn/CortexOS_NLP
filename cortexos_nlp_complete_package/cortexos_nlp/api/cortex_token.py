"""
CortexOS NLP - Token Class

The Token class represents a single token with all its linguistic properties,
providing spaCy-compatible interface while exposing deterministic analysis results.
"""

from typing import List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .cortex_doc import Doc
    from .cortex_span import Span


class Token:
    """
    A single token with linguistic annotations.
    
    The Token class provides access to all linguistic properties
    determined by the CortexOS deterministic NLP engine.
    
    Attributes:
        text (str): The token text
        pos_ (str): Part-of-speech tag
        tag_ (str): Detailed POS tag
        dep_ (str): Dependency relation
        lemma_ (str): Lemmatized form
        head (Token): Syntactic head token
        children (List[Token]): Syntactic children tokens
    """
    
    def __init__(self, doc: 'Doc', index: int, linguistic_token):
        """
        Initialize a Token object.
        
        Args:
            doc: The Doc object this token belongs to
            index: Index of this token in the document
            linguistic_token: The underlying linguistic token from Phase 2
        """
        self._doc = doc
        self._index = index
        self._linguistic_token = linguistic_token
        
        # Cache for expensive operations
        self._head = None
        self._children = None
        self._dep = "ROOT"  # Default dependency relation
        
        # Character position in document
        self._idx = self._calculate_char_position()
        
        # Linguistic properties
        self._pos = getattr(linguistic_token, 'pos_tag', 'UNKNOWN')
        self._tag = getattr(linguistic_token, 'detailed_tag', self._pos)
        self._lemma = getattr(linguistic_token, 'lemma', linguistic_token.text.lower())
        
        # Confidence scores
        self._pos_confidence = getattr(linguistic_token, 'pos_confidence', 1.0)
        self._certainty_level = getattr(linguistic_token, 'certainty_level', 'HIGH')
    
    def _calculate_char_position(self) -> int:
        """Calculate the character position of this token in the document."""
        position = 0
        for i in range(self._index):
            position += len(self._doc.tokens[i].text)
        return position
    
    @property
    def text(self) -> str:
        """The token text."""
        return self._linguistic_token.text
    
    @property
    def text_with_ws(self) -> str:
        """The token text with trailing whitespace."""
        # Check if next token is a space
        if (self._index + 1 < len(self._doc.tokens) and 
            self._doc.tokens[self._index + 1].is_space):
            return self.text + " "
        return self.text
    
    @property
    def pos_(self) -> str:
        """Part-of-speech tag."""
        return self._pos
    
    @property
    def tag_(self) -> str:
        """Detailed part-of-speech tag."""
        return self._tag
    
    @property
    def dep_(self) -> str:
        """Dependency relation to the head."""
        return self._dep
    
    @property
    def lemma_(self) -> str:
        """Lemmatized form of the token."""
        return self._lemma
    
    @property
    def head(self) -> 'Token':
        """The syntactic head of this token."""
        if self._head is None:
            return self  # Self-reference for root tokens
        return self._head
    
    @property
    def children(self) -> List['Token']:
        """List of syntactic children of this token."""
        if self._children is None:
            return []
        return self._children
    
    @property
    def idx(self) -> int:
        """Character index of the token in the document."""
        return self._idx
    
    @property
    def i(self) -> int:
        """Token index in the document."""
        return self._index
    
    @property
    def doc(self) -> 'Doc':
        """The Doc object this token belongs to."""
        return self._doc
    
    # Boolean properties
    @property
    def is_alpha(self) -> bool:
        """Whether the token consists of alphabetic characters."""
        return self._linguistic_token.is_alpha
    
    @property
    def is_digit(self) -> bool:
        """Whether the token consists of digits."""
        return self._linguistic_token.is_digit
    
    @property
    def is_punct(self) -> bool:
        """Whether the token is punctuation."""
        return self._linguistic_token.is_punct
    
    @property
    def is_space(self) -> bool:
        """Whether the token is whitespace."""
        return self._linguistic_token.is_space
    
    @property
    def is_stop(self) -> bool:
        """Whether the token is a stop word."""
        # Simple stop word detection
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'this',
            'these', 'those', 'but', 'or', 'not', 'have', 'had', 'do', 'does',
            'did', 'can', 'could', 'should', 'would', 'may', 'might', 'must'
        }
        return self.text.lower() in stop_words
    
    @property
    def is_sent_start(self) -> bool:
        """Whether this token starts a sentence."""
        # First non-space token or token after sentence-ending punctuation
        if self._index == 0:
            return True
        
        # Look for sentence-ending punctuation in previous tokens
        for i in range(self._index - 1, -1, -1):
            prev_token = self._doc.tokens[i]
            if prev_token.is_space:
                continue
            if prev_token.text in '.!?':
                return True
            break
        
        return False
    
    @property
    def is_sent_end(self) -> bool:
        """Whether this token ends a sentence."""
        return self.text in '.!?'
    
    @property
    def is_title(self) -> bool:
        """Whether the token is title-cased."""
        return self.text.istitle()
    
    @property
    def is_upper(self) -> bool:
        """Whether the token is uppercase."""
        return self.text.isupper()
    
    @property
    def is_lower(self) -> bool:
        """Whether the token is lowercase."""
        return self.text.islower()
    
    @property
    def is_ascii(self) -> bool:
        """Whether the token consists of ASCII characters."""
        return all(ord(char) < 128 for char in self.text)
    
    @property
    def is_bracket(self) -> bool:
        """Whether the token is a bracket."""
        return self.text in '()[]{}<>'
    
    @property
    def is_quote(self) -> bool:
        """Whether the token is a quotation mark."""
        return self.text in '\'""`''""'
    
    @property
    def is_currency(self) -> bool:
        """Whether the token is a currency symbol."""
        return self.text in '$€£¥₹₽¢'
    
    @property
    def like_url(self) -> bool:
        """Whether the token resembles a URL."""
        text_lower = self.text.lower()
        return (text_lower.startswith(('http://', 'https://', 'www.')) or
                '.' in text_lower and any(text_lower.endswith(ext) 
                for ext in ['.com', '.org', '.net', '.edu', '.gov']))
    
    @property
    def like_email(self) -> bool:
        """Whether the token resembles an email address."""
        return '@' in self.text and '.' in self.text
    
    @property
    def like_num(self) -> bool:
        """Whether the token resembles a number."""
        # Remove common number separators and check if remaining chars are digits
        cleaned = self.text.replace(',', '').replace('.', '').replace('-', '')
        return cleaned.isdigit() or self.text.replace('.', '').isdigit()
    
    # CortexOS-specific properties
    @property
    def pos_confidence(self) -> float:
        """Confidence score for POS tagging (0.0 to 1.0)."""
        return self._pos_confidence
    
    @property
    def certainty_level(self) -> str:
        """Certainty level for this token's analysis."""
        return self._certainty_level
    
    @property
    def spatial_coordinate(self) -> Optional[Any]:
        """Spatial coordinate for this token if available."""
        if hasattr(self._linguistic_token, 'spatial_coordinate'):
            return self._linguistic_token.spatial_coordinate
        return None
    
    @property
    def is_deterministic(self) -> bool:
        """Whether this token was processed deterministically."""
        return True  # CortexOS is always deterministic
    
    # Navigation methods
    def nbor(self, i: int = 1) -> 'Token':
        """
        Get neighboring token.
        
        Args:
            i: Offset from current token (default: 1 for next token)
            
        Returns:
            Token object at the specified offset
        """
        new_index = self._index + i
        if 0 <= new_index < len(self._doc.tokens):
            return self._doc.tokens[new_index]
        raise IndexError(f"Token index {new_index} out of range")
    
    def is_ancestor(self, descendant: 'Token') -> bool:
        """
        Check if this token is an ancestor of another token.
        
        Args:
            descendant: Token to check
            
        Returns:
            True if this token is an ancestor of the descendant
        """
        current = descendant
        while current.head != current:  # Not root
            if current.head == self:
                return True
            current = current.head
        return False
    
    def is_descendant(self, ancestor: 'Token') -> bool:
        """
        Check if this token is a descendant of another token.
        
        Args:
            ancestor: Token to check
            
        Returns:
            True if this token is a descendant of the ancestor
        """
        return ancestor.is_ancestor(self)
    
    def similarity(self, other: 'Token') -> float:
        """
        Calculate similarity with another token.
        
        Args:
            other: Another Token to compare with
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple similarity based on text and POS
        if self.text.lower() == other.text.lower():
            return 1.0
        elif self.pos_ == other.pos_:
            return 0.5
        else:
            return 0.0
    
    # Span creation methods
    def subtree(self) -> 'Span':
        """
        Get the syntactic subtree rooted at this token.
        
        Returns:
            Span containing this token and all its descendants
        """
        from .cortex_span import Span
        
        # Find all descendants
        descendants = []
        
        def collect_descendants(token):
            descendants.append(token)
            for child in token.children:
                collect_descendants(child)
        
        collect_descendants(self)
        
        # Find the span boundaries
        indices = [token.i for token in descendants]
        start = min(indices)
        end = max(indices) + 1
        
        return Span(self._doc, start, end, label="SUBTREE")
    
    def lefts(self) -> List['Token']:
        """Get left children of this token."""
        return [child for child in self.children if child.i < self.i]
    
    def rights(self) -> List['Token']:
        """Get right children of this token."""
        return [child for child in self.children if child.i > self.i]
    
    def ancestors(self) -> List['Token']:
        """Get all ancestors of this token."""
        ancestors = []
        current = self.head
        while current != current.head:  # Not root
            ancestors.append(current)
            current = current.head
        return ancestors
    
    # String methods
    def __str__(self) -> str:
        """String representation of the token."""
        return self.text
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Token('{self.text}', pos='{self.pos_}', dep='{self.dep_}')"
    
    def __len__(self) -> int:
        """Length of the token text."""
        return len(self.text)
    
    def __eq__(self, other) -> bool:
        """Check equality with another token or string."""
        if isinstance(other, Token):
            return (self._doc == other._doc and 
                   self._index == other._index)
        elif isinstance(other, str):
            return self.text == other
        return False
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash((id(self._doc), self._index))
    
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
    
    # Analysis methods
    def explain_pos(self) -> str:
        """
        Get explanation for POS tag assignment.
        
        Returns:
            Human-readable explanation of POS tagging
        """
        return (f"Token '{self.text}' tagged as {self.pos_} "
                f"with {self.pos_confidence:.2f} confidence "
                f"({self.certainty_level} certainty)")
    
    def explain_dependency(self) -> str:
        """
        Get explanation for dependency relation.
        
        Returns:
            Human-readable explanation of dependency relation
        """
        if self.head == self:
            return f"Token '{self.text}' is the root of the sentence"
        else:
            return (f"Token '{self.text}' is {self.dep_} of '{self.head.text}'")
    
    def to_dict(self) -> dict:
        """
        Convert token to dictionary representation.
        
        Returns:
            Dictionary containing token properties
        """
        return {
            'text': self.text,
            'pos': self.pos_,
            'tag': self.tag_,
            'dep': self.dep_,
            'lemma': self.lemma_,
            'idx': self.idx,
            'i': self.i,
            'is_alpha': self.is_alpha,
            'is_digit': self.is_digit,
            'is_punct': self.is_punct,
            'is_space': self.is_space,
            'is_stop': self.is_stop,
            'pos_confidence': self.pos_confidence,
            'certainty_level': self.certainty_level,
            'head_text': self.head.text if self.head != self else None,
            'children_count': len(self.children)
        }

