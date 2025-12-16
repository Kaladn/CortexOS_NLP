"""
CortexOS NLP - Doc Class

The Doc class represents a processed document with tokens, providing
spaCy-compatible interface while exposing deterministic processing results.
"""

import json
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from .cortex_token import Token
from .cortex_span import Span


class Doc:
    """
    A processed document containing tokens and linguistic analysis.
    
    The Doc class provides a spaCy-compatible interface to access
    the results of CortexOS deterministic NLP processing.
    
    Attributes:
        text (str): The original input text
        tokens (List[Token]): List of Token objects
        ents (List[Span]): Named entities (future implementation)
        sents (List[Span]): Sentences in the document
        noun_chunks (List[Span]): Noun phrases (future implementation)
    """
    
    def __init__(self, cortex_nlp, text: str, linguistic_doc):
        """
        Initialize a Doc object.
        
        Args:
            cortex_nlp: The CortexNLP instance that created this doc
            text: Original input text
            linguistic_doc: The processed linguistic document from Phase 2
        """
        self._cortex_nlp = cortex_nlp
        self._text = text
        self._linguistic_doc = linguistic_doc
        
        # Create Token objects
        self._tokens = []
        for i, ling_token in enumerate(linguistic_doc.tokens):
            token = Token(self, i, ling_token)
            self._tokens.append(token)
        
        # Set up token relationships
        self._setup_token_relationships()
        
        # Cache for expensive operations
        self._sents_cache = None
        self._ents_cache = None
        self._noun_chunks_cache = None
        
        # Spatial anchors from processing
        self._spatial_anchors = linguistic_doc.spatial_anchors
        
        # Processing metadata
        self._metadata = linguistic_doc.processing_metadata
    
    def _setup_token_relationships(self):
        """Set up head/children relationships between tokens."""
        # Create token lookup by ID
        token_by_id = {token._linguistic_token.token_id: token 
                      for token in self._tokens}
        
        # Set up dependency relationships
        for relationship in self._linguistic_doc.dependency_tree.relationships:
            head_id = relationship.head.token_id
            dep_id = relationship.dependent.token_id
            
            if head_id in token_by_id and dep_id in token_by_id:
                head_token = token_by_id[head_id]
                dep_token = token_by_id[dep_id]
                
                # Set head relationship
                dep_token._head = head_token
                dep_token._dep = relationship.relation
                
                # Add to children
                if not hasattr(head_token, '_children'):
                    head_token._children = []
                head_token._children.append(dep_token)
    
    @property
    def text(self) -> str:
        """The original input text."""
        return self._text
    
    @property
    def tokens(self) -> List[Token]:
        """List of Token objects in the document."""
        return self._tokens
    
    @property
    def ents(self) -> List[Span]:
        """Named entities in the document (future implementation)."""
        if self._ents_cache is None:
            # For now, return empty list - will implement NER later
            self._ents_cache = []
        return self._ents_cache
    
    @property
    def sents(self) -> List[Span]:
        """Sentences in the document."""
        if self._sents_cache is None:
            self._sents_cache = self._extract_sentences()
        return self._sents_cache
    
    @property
    def noun_chunks(self) -> List[Span]:
        """Noun phrases in the document (future implementation)."""
        if self._noun_chunks_cache is None:
            # For now, return empty list - will implement noun chunking later
            self._noun_chunks_cache = []
        return self._noun_chunks_cache
    
    def _extract_sentences(self) -> List[Span]:
        """Extract sentence spans from the document."""
        sentences = []
        current_start = 0
        
        for i, token in enumerate(self._tokens):
            # Simple sentence boundary detection
            if token.is_sent_end or i == len(self._tokens) - 1:
                # Create sentence span
                end_idx = i + 1
                if end_idx > current_start:
                    span = Span(self, current_start, end_idx, label="SENT")
                    sentences.append(span)
                current_start = end_idx
        
        return sentences
    
    def __len__(self) -> int:
        """Number of tokens in the document."""
        return len(self._tokens)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[Token, Span]:
        """Get token(s) by index or slice."""
        if isinstance(key, int):
            return self._tokens[key]
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self._tokens))
            if step != 1:
                raise ValueError("Step slicing not supported")
            return Span(self, start, stop)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __iter__(self) -> Iterator[Token]:
        """Iterate over tokens in the document."""
        return iter(self._tokens)
    
    def __str__(self) -> str:
        """String representation of the document."""
        return self._text
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Doc('{self._text[:50]}{'...' if len(self._text) > 50 else ''}')"
    
    def similarity(self, other: 'Doc') -> float:
        """
        Calculate similarity with another document.
        
        Args:
            other: Another Doc object to compare with
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        return self._cortex_nlp.similarity(self, other)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Export document to JSON format.
        
        Returns:
            Dictionary containing document data in JSON-serializable format
        """
        return {
            'text': self._text,
            'tokens': [
                {
                    'text': token.text,
                    'start': token.idx,
                    'end': token.idx + len(token.text),
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'lemma': token.lemma_,
                    'is_alpha': token.is_alpha,
                    'is_digit': token.is_digit,
                    'is_punct': token.is_punct,
                    'is_space': token.is_space,
                    'is_stop': token.is_stop
                }
                for token in self._tokens
            ],
            'dependencies': [
                {
                    'head': rel.head.text,
                    'head_idx': self._get_token_index(rel.head),
                    'dependent': rel.dependent.text,
                    'dependent_idx': self._get_token_index(rel.dependent),
                    'relation': rel.relation,
                    'confidence': rel.confidence
                }
                for rel in self._linguistic_doc.dependency_tree.relationships
            ],
            'spatial_anchors': {
                word: [coord.x1, coord.x2, coord.x3, coord.x4, coord.x5, coord.x6]
                for word, coord in self._spatial_anchors.items()
            },
            'processing_metadata': {
                'total_time': self._metadata.total_processing_time,
                'avg_pos_confidence': self._metadata.average_pos_confidence,
                'avg_dep_confidence': self._metadata.average_dependency_confidence,
                'deterministic': True
            }
        }
    
    def _get_token_index(self, linguistic_token) -> int:
        """Get the index of a linguistic token in our token list."""
        for i, token in enumerate(self._tokens):
            if token._linguistic_token.token_id == linguistic_token.token_id:
                return i
        return -1
    
    def to_conllu(self) -> str:
        """
        Export document to CoNLL-U format.
        
        Returns:
            String in CoNLL-U format
        """
        return self._linguistic_doc.to_conllu()
    
    def char_span(self, start: int, end: int, label: str = None) -> Optional[Span]:
        """
        Create a Span object from character indices.
        
        Args:
            start: Start character index
            end: End character index
            label: Optional label for the span
            
        Returns:
            Span object or None if indices are invalid
        """
        # Find tokens that overlap with the character span
        token_start = None
        token_end = None
        
        for i, token in enumerate(self._tokens):
            token_char_start = token.idx
            token_char_end = token.idx + len(token.text)
            
            # Find first token that starts at or after the start position
            if token_start is None and token_char_start >= start:
                token_start = i
            
            # Find last token that ends at or before the end position
            if token_char_end <= end:
                token_end = i + 1
        
        if token_start is not None and token_end is not None and token_start < token_end:
            return Span(self, token_start, token_end, label=label)
        
        return None
    
    def get_extension(self, name: str) -> Any:
        """
        Get custom extension attribute (spaCy compatibility).
        
        Args:
            name: Name of the extension attribute
            
        Returns:
            Extension attribute value
        """
        return getattr(self, f"_{name}", None)
    
    def set_extension(self, name: str, value: Any):
        """
        Set custom extension attribute (spaCy compatibility).
        
        Args:
            name: Name of the extension attribute
            value: Value to set
        """
        setattr(self, f"_{name}", value)
    
    def retokenize(self):
        """
        Context manager for retokenization (spaCy compatibility).
        Currently not implemented - returns self for compatibility.
        """
        return self
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    # Linguistic analysis properties
    @property
    def spatial_anchors(self) -> Dict[str, Any]:
        """Access to spatial anchor coordinates."""
        return self._spatial_anchors
    
    @property
    def processing_metadata(self) -> Any:
        """Access to processing metadata."""
        return self._metadata
    
    @property
    def is_deterministic(self) -> bool:
        """Whether this document was processed deterministically."""
        return True  # CortexOS is always deterministic
    
    @property
    def confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores for different processing steps."""
        return {
            'pos_tagging': self._metadata.average_pos_confidence,
            'dependency_parsing': self._metadata.average_dependency_confidence,
            'overall': (self._metadata.average_pos_confidence + 
                       self._metadata.average_dependency_confidence) / 2
        }
    
    def explain_processing(self) -> Dict[str, Any]:
        """
        Get detailed explanation of how this document was processed.
        
        Returns:
            Dictionary containing processing explanation
        """
        return self._cortex_nlp.explain(self._text)
    
    def get_token_by_id(self, token_id: int) -> Optional[Token]:
        """
        Get token by its linguistic token ID.
        
        Args:
            token_id: The linguistic token ID
            
        Returns:
            Token object or None if not found
        """
        for token in self._tokens:
            if token._linguistic_token.token_id == token_id:
                return token
        return None
    
    def find_tokens(self, text: str, case_sensitive: bool = True) -> List[Token]:
        """
        Find all tokens matching the given text.
        
        Args:
            text: Text to search for
            case_sensitive: Whether to match case exactly
            
        Returns:
            List of matching Token objects
        """
        matches = []
        search_text = text if case_sensitive else text.lower()
        
        for token in self._tokens:
            token_text = token.text if case_sensitive else token.text.lower()
            if token_text == search_text:
                matches.append(token)
        
        return matches
    
    def get_dependency_tree_string(self) -> str:
        """
        Get a string representation of the dependency tree.
        
        Returns:
            Human-readable dependency tree
        """
        lines = []
        lines.append("Dependency Tree:")
        lines.append("-" * 40)
        
        for rel in self._linguistic_doc.dependency_tree.relationships:
            lines.append(f"{rel.dependent.text} --[{rel.relation}]--> {rel.head.text}")
        
        return "\n".join(lines)
    
    def print_dependencies(self):
        """Print the dependency tree to console."""
        print(self.get_dependency_tree_string())

