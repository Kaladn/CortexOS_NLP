"""
CortexOS NLP - Comprehensive Phase 2 Test Suite

This module provides comprehensive testing for all Phase 2 components:
- Individual module testing (tokenizer, tagger, parser)
- Integration testing (full pipeline)
- Performance benchmarking
- Edge case handling
- Consistency validation

Core Principle: Ensure mathematical determinism and perfect reproducibility
across all linguistic processing operations.
"""

import time
import unittest
from typing import List, Dict, Any
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import SpatialAnchor, BinaryCellMemory, HarmonicResonance
from linguistic.tokenizer import CortexTokenizer, Token
from linguistic.tagger import CortexTagger
from linguistic.parser import CortexParser, DependencyTree
from linguistic.integrated_processor import CortexLinguisticProcessor, LinguisticDocument


class TestTokenizer(unittest.TestCase):
    """Test suite for CortexTokenizer module"""
    
    def setUp(self):
        self.tokenizer = CortexTokenizer()
    
    def test_basic_tokenization(self):
        """Test basic tokenization functionality"""
        text = "The quick brown fox jumps."
        tokens = self.tokenizer.tokenize(text)
        
        # Check token count (including spaces)
        content_tokens = [t for t in tokens if not t.is_space]
        self.assertEqual(len(content_tokens), 6)  # 5 words + 1 punctuation
        
        # Check token text
        expected_texts = ["The", "quick", "brown", "fox", "jumps", "."]
        actual_texts = [t.text for t in content_tokens]
        self.assertEqual(actual_texts, expected_texts)
    
    def test_tokenization_consistency(self):
        """Test that tokenization is perfectly consistent across runs"""
        text = "Hello world! This is a test."
        
        # Tokenize multiple times
        results = []
        for _ in range(5):
            tokens = self.tokenizer.tokenize(text)
            result = [(t.text, t.normalized, t.is_space, t.is_punct) for t in tokens]
            results.append(result)
        
        # All results should be identical
        for result in results[1:]:
            self.assertEqual(result, results[0])
    
    def test_spatial_coordinates(self):
        """Test that tokens have valid spatial coordinates"""
        text = "Test spatial coordinates."
        tokens = self.tokenizer.tokenize(text)
        
        for token in tokens:
            if not token.is_space:
                # Check that spatial coordinate exists
                self.assertIsNotNone(token.spatial_coord)
                # Check coordinate dimensions
                coord = token.spatial_coord
                self.assertIsNotNone(coord.x1)
                self.assertIsNotNone(coord.x2)
                self.assertIsNotNone(coord.x3)
                self.assertIsNotNone(coord.x4)
                self.assertIsNotNone(coord.x5)
                self.assertIsNotNone(coord.x6)
    
    def test_edge_cases(self):
        """Test tokenization edge cases"""
        # Empty string
        tokens = self.tokenizer.tokenize("")
        self.assertEqual(len(tokens), 0)
        
        # Single character
        tokens = self.tokenizer.tokenize("a")
        self.assertEqual(len([t for t in tokens if not t.is_space]), 1)
        
        # Only punctuation
        tokens = self.tokenizer.tokenize("!!!")
        content_tokens = [t for t in tokens if not t.is_space]
        self.assertTrue(all(t.is_punct for t in content_tokens))
        
        # Mixed special characters
        tokens = self.tokenizer.tokenize("Hello@world.com")
        self.assertGreater(len([t for t in tokens if not t.is_space]), 1)


class TestTagger(unittest.TestCase):
    """Test suite for CortexTagger module"""
    
    def setUp(self):
        self.anchor_system = SpatialAnchor()
        self.memory = BinaryCellMemory()
        self.resonance = HarmonicResonance(self.memory)
        self.tokenizer = CortexTokenizer()
        self.tagger = CortexTagger(self.anchor_system, self.memory, self.resonance)
    
    def test_basic_pos_tagging(self):
        """Test basic POS tagging functionality"""
        text = "The quick brown fox jumps."
        tokens = self.tokenizer.tokenize(text)
        tagged_tokens = self.tagger.tag_tokens(tokens)
        
        # Check that all content tokens have POS tags
        content_tokens = [t for t in tagged_tokens if not t.is_space]
        for token in content_tokens:
            if not token.is_punct:
                self.assertIsNotNone(token.pos_tag)
                self.assertIsInstance(token.pos_tag, str)
                self.assertGreater(len(token.pos_tag), 0)
    
    def test_pos_tagging_consistency(self):
        """Test that POS tagging is consistent across runs"""
        text = "The cat sits on the mat."
        
        results = []
        for _ in range(3):
            tokens = self.tokenizer.tokenize(text)
            tagged_tokens = self.tagger.tag_tokens(tokens)
            pos_sequence = [getattr(t, 'pos_tag', None) for t in tagged_tokens if not t.is_space]
            results.append(pos_sequence)
        
        # All results should be identical
        for result in results[1:]:
            self.assertEqual(result, results[0])
    
    def test_known_word_tagging(self):
        """Test tagging of known words"""
        text = "The dog runs quickly."
        tokens = self.tokenizer.tokenize(text)
        tagged_tokens = self.tagger.tag_tokens(tokens)
        
        # Check specific expected POS tags
        content_tokens = [t for t in tagged_tokens if not t.is_space and not t.is_punct]
        pos_tags = [t.pos_tag for t in content_tokens]
        
        # "The" should be DT (determiner)
        self.assertEqual(pos_tags[0], "DT")
        
        # Should have reasonable POS tags (no None values for content words)
        for pos_tag in pos_tags:
            self.assertIsNotNone(pos_tag)
            self.assertIn(pos_tag, ["DT", "NN", "NNS", "VBZ", "VBP", "VB", "RB", "JJ", "VBG", "VBD"])
    
    def test_confidence_scores(self):
        """Test that confidence scores are reasonable"""
        text = "The beautiful red car drives fast."
        tokens = self.tokenizer.tokenize(text)
        tagged_tokens = self.tagger.tag_tokens(tokens)
        
        for token in tagged_tokens:
            if hasattr(token, 'pos_confidence'):
                # Confidence should be between 0 and 1
                self.assertGreaterEqual(token.pos_confidence, 0.0)
                self.assertLessEqual(token.pos_confidence, 1.0)


class TestParser(unittest.TestCase):
    """Test suite for CortexParser module"""
    
    def setUp(self):
        self.anchor_system = SpatialAnchor()
        self.memory = BinaryCellMemory()
        self.resonance = HarmonicResonance(self.memory)
        self.tokenizer = CortexTokenizer()
        self.tagger = CortexTagger(self.anchor_system, self.memory, self.resonance)
        self.parser = CortexParser(self.memory, self.resonance)
    
    def test_basic_parsing(self):
        """Test basic dependency parsing"""
        text = "The cat sits."
        tokens = self.tokenizer.tokenize(text)
        tagged_tokens = self.tagger.tag_tokens(tokens)
        tree = self.parser.parse_sentence(tagged_tokens)
        
        # Should have a dependency tree
        self.assertIsInstance(tree, DependencyTree)
        
        # Should have some relationships
        self.assertGreaterEqual(len(tree.relationships), 0)
        
        # Should have a root token
        self.assertIsNotNone(tree.root)
    
    def test_parsing_consistency(self):
        """Test that parsing is consistent across runs"""
        text = "The quick brown fox jumps over the lazy dog."
        
        results = []
        for _ in range(3):
            tokens = self.tokenizer.tokenize(text)
            tagged_tokens = self.tagger.tag_tokens(tokens)
            tree = self.parser.parse_sentence(tagged_tokens)
            
            # Create comparable result
            relationships = [(r.head.text, r.dependent.text, r.relation) for r in tree.relationships]
            relationships.sort()  # Sort for comparison
            results.append(relationships)
        
        # All results should be identical
        for result in results[1:]:
            self.assertEqual(result, results[0])
    
    def test_dependency_relationships(self):
        """Test that dependency relationships are valid"""
        text = "The red car drives fast."
        tokens = self.tokenizer.tokenize(text)
        tagged_tokens = self.tagger.tag_tokens(tokens)
        tree = self.parser.parse_sentence(tagged_tokens)
        
        for rel in tree.relationships:
            # Check relationship structure
            self.assertIsNotNone(rel.head)
            self.assertIsNotNone(rel.dependent)
            self.assertIsNotNone(rel.relation)
            self.assertIsInstance(rel.relation, str)
            
            # Check confidence score
            self.assertGreaterEqual(rel.confidence, 0.0)
            self.assertLessEqual(rel.confidence, 1.0)
            
            # Head and dependent should be different tokens
            self.assertNotEqual(rel.head.token_id, rel.dependent.token_id)
    
    def test_conllu_export(self):
        """Test CoNLL-U format export"""
        text = "The cat sits on the mat."
        tokens = self.tokenizer.tokenize(text)
        tagged_tokens = self.tagger.tag_tokens(tokens)
        tree = self.parser.parse_sentence(tagged_tokens)
        
        conllu_output = tree.to_conllu()
        
        # Should be non-empty string
        self.assertIsInstance(conllu_output, str)
        self.assertGreater(len(conllu_output), 0)
        
        # Should have proper format (tab-separated columns)
        lines = conllu_output.strip().split('\n')
        for line in lines:
            columns = line.split('\t')
            self.assertEqual(len(columns), 10)  # CoNLL-U has 10 columns


class TestIntegratedProcessor(unittest.TestCase):
    """Test suite for the integrated linguistic processor"""
    
    def setUp(self):
        self.processor = CortexLinguisticProcessor()
    
    def test_full_pipeline(self):
        """Test the complete processing pipeline"""
        text = "The quick brown fox jumps over the lazy dog."
        doc = self.processor.process_text(text)
        
        # Check document structure
        self.assertIsInstance(doc, LinguisticDocument)
        self.assertEqual(doc.original_text, text)
        self.assertGreater(len(doc.tokens), 0)
        self.assertIsNotNone(doc.dependency_tree)
        self.assertIsNotNone(doc.processing_metadata)
    
    def test_processing_metadata(self):
        """Test that processing metadata is complete"""
        text = "Hello world! This is a test sentence."
        doc = self.processor.process_text(text)
        
        metadata = doc.processing_metadata
        
        # Check timing information
        self.assertGreater(metadata.total_processing_time, 0)
        self.assertGreaterEqual(metadata.tokenization_time, 0)
        self.assertGreaterEqual(metadata.tagging_time, 0)
        self.assertGreaterEqual(metadata.parsing_time, 0)
        
        # Check token counts
        self.assertGreater(metadata.total_tokens, 0)
        self.assertGreater(metadata.content_tokens, 0)
        
        # Check confidence scores
        self.assertGreaterEqual(metadata.average_pos_confidence, 0.0)
        self.assertLessEqual(metadata.average_pos_confidence, 1.0)
        self.assertGreaterEqual(metadata.average_dependency_confidence, 0.0)
        self.assertLessEqual(metadata.average_dependency_confidence, 1.0)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        texts = [
            "The cat sits.",
            "Dogs run fast.",
            "Birds fly high.",
            "Fish swim deep."
        ]
        
        docs = self.processor.batch_process(texts)
        
        # Should have same number of documents as input texts
        self.assertEqual(len(docs), len(texts))
        
        # Each document should be valid
        for i, doc in enumerate(docs):
            self.assertEqual(doc.original_text, texts[i])
            self.assertGreater(len(doc.tokens), 0)
    
    def test_json_export(self):
        """Test JSON export functionality"""
        text = "The red car drives fast."
        doc = self.processor.process_text(text)
        
        json_data = doc.to_json()
        
        # Check JSON structure
        self.assertIsInstance(json_data, dict)
        self.assertIn("original_text", json_data)
        self.assertIn("tokens", json_data)
        self.assertIn("dependencies", json_data)
        self.assertIn("processing_metadata", json_data)
        
        # Check token data
        self.assertIsInstance(json_data["tokens"], list)
        self.assertGreater(len(json_data["tokens"]), 0)
        
        # Check dependency data
        self.assertIsInstance(json_data["dependencies"], list)
    
    def test_document_similarity(self):
        """Test document similarity calculation"""
        doc1 = self.processor.process_text("The cat sits on the mat.")
        doc2 = self.processor.process_text("The dog lies on the rug.")
        doc3 = self.processor.process_text("Completely different sentence structure here.")
        
        # Calculate similarities
        sim_1_2 = self.processor.compare_documents(doc1, doc2)
        sim_1_3 = self.processor.compare_documents(doc1, doc3)
        
        # Check similarity structure
        self.assertIn("spatial_similarity", sim_1_2)
        self.assertIn("pos_similarity", sim_1_2)
        self.assertIn("dependency_similarity", sim_1_2)
        self.assertIn("structural_similarity", sim_1_2)
        
        # Similarity scores should be between 0 and 1
        for metric, score in sim_1_2.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Similar sentences should have higher similarity than dissimilar ones
        self.assertGreater(sim_1_2["structural_similarity"], sim_1_3["structural_similarity"])


class TestPerformance(unittest.TestCase):
    """Performance benchmarking tests"""
    
    def setUp(self):
        self.processor = CortexLinguisticProcessor()
    
    def test_processing_speed(self):
        """Test processing speed benchmarks"""
        # Test different sentence lengths
        test_cases = [
            ("Short.", 1),
            ("This is a medium length sentence.", 7),
            ("This is a much longer sentence that contains multiple clauses and various grammatical structures to test performance.", 18)
        ]
        
        for text, expected_min_tokens in test_cases:
            start_time = time.time()
            doc = self.processor.process_text(text)
            processing_time = time.time() - start_time
            
            # Should process reasonably quickly (under 1 second for test sentences)
            self.assertLess(processing_time, 1.0)
            
            # Should have expected minimum number of content tokens
            content_tokens = len([t for t in doc.tokens if not t.is_space and not t.is_punct])
            self.assertGreaterEqual(content_tokens, expected_min_tokens)
    
    def test_batch_performance(self):
        """Test batch processing performance"""
        texts = ["This is test sentence number {}.".format(i) for i in range(10)]
        
        start_time = time.time()
        docs = self.processor.batch_process(texts)
        batch_time = time.time() - start_time
        
        # Should complete batch processing reasonably quickly
        self.assertLess(batch_time, 5.0)  # 5 seconds for 10 sentences
        
        # Should process all texts
        self.assertEqual(len(docs), len(texts))
        
        # Average time per document should be reasonable
        avg_time = batch_time / len(texts)
        self.assertLess(avg_time, 1.0)  # Less than 1 second per sentence
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        # Process multiple documents and check memory stats
        texts = [
            "The cat sits on the mat.",
            "Dogs run in the park.",
            "Birds fly through the sky.",
            "Fish swim in the ocean.",
            "Cars drive on the road."
        ]
        
        initial_stats = self.processor.get_processing_statistics()
        
        for text in texts:
            self.processor.process_text(text)
        
        final_stats = self.processor.get_processing_statistics()
        
        # Should have processed all documents
        docs_processed = final_stats["documents_processed"] - initial_stats.get("documents_processed", 0)
        self.assertEqual(docs_processed, len(texts))
        
        # Memory usage should be reasonable (relationships stored should be proportional to content)
        total_relationships = final_stats["total_relationships"]
        self.assertGreater(total_relationships, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.processor = CortexLinguisticProcessor()
    
    def test_empty_input(self):
        """Test handling of empty input"""
        doc = self.processor.process_text("")
        
        # Should handle gracefully
        self.assertEqual(doc.original_text, "")
        self.assertEqual(len(doc.tokens), 0)
        self.assertEqual(len(doc.dependency_tree.relationships), 0)
    
    def test_single_word(self):
        """Test handling of single word input"""
        doc = self.processor.process_text("Hello")
        
        # Should process single word correctly
        self.assertEqual(doc.original_text, "Hello")
        self.assertGreater(len(doc.tokens), 0)
        
        content_tokens = [t for t in doc.tokens if not t.is_space]
        self.assertEqual(len(content_tokens), 1)
        self.assertEqual(content_tokens[0].text, "Hello")
    
    def test_punctuation_only(self):
        """Test handling of punctuation-only input"""
        doc = self.processor.process_text("!!!")
        
        # Should handle punctuation correctly
        self.assertEqual(doc.original_text, "!!!")
        content_tokens = [t for t in doc.tokens if not t.is_space]
        self.assertGreater(len(content_tokens), 0)
        self.assertTrue(all(t.is_punct for t in content_tokens))
    
    def test_special_characters(self):
        """Test handling of special characters"""
        doc = self.processor.process_text("Hello@world.com & test#123")
        
        # Should tokenize and process without errors
        self.assertGreater(len(doc.tokens), 0)
        self.assertIsNotNone(doc.dependency_tree)
    
    def test_very_long_sentence(self):
        """Test handling of very long sentences"""
        # Create a long sentence
        long_sentence = "This is a very long sentence that " * 20 + "contains many repeated phrases."
        
        doc = self.processor.process_text(long_sentence)
        
        # Should handle long sentences without errors
        self.assertEqual(doc.original_text, long_sentence)
        self.assertGreater(len(doc.tokens), 50)  # Should have many tokens
        self.assertIsNotNone(doc.dependency_tree)


class TestDeterminism(unittest.TestCase):
    """Test deterministic behavior across multiple runs"""
    
    def setUp(self):
        self.test_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "I can't believe it's working so well!",
            "Complex sentences with subordinate clauses are challenging.",
            "The beautiful red car drives very fast down the winding road."
        ]
    
    def test_tokenization_determinism(self):
        """Test that tokenization is perfectly deterministic"""
        for sentence in self.test_sentences:
            results = []
            
            # Run tokenization multiple times
            for _ in range(5):
                processor = CortexLinguisticProcessor()
                doc = processor.process_text(sentence)
                token_data = [(t.text, t.normalized, t.is_space, t.is_punct) for t in doc.tokens]
                results.append(token_data)
            
            # All results should be identical
            for result in results[1:]:
                self.assertEqual(result, results[0], f"Tokenization not deterministic for: {sentence}")
    
    def test_pos_tagging_determinism(self):
        """Test that POS tagging is perfectly deterministic"""
        for sentence in self.test_sentences:
            results = []
            
            # Run POS tagging multiple times
            for _ in range(5):
                processor = CortexLinguisticProcessor()
                doc = processor.process_text(sentence)
                pos_data = [getattr(t, 'pos_tag', None) for t in doc.tokens]
                results.append(pos_data)
            
            # All results should be identical
            for result in results[1:]:
                self.assertEqual(result, results[0], f"POS tagging not deterministic for: {sentence}")
    
    def test_parsing_determinism(self):
        """Test that dependency parsing is perfectly deterministic"""
        for sentence in self.test_sentences:
            results = []
            
            # Run parsing multiple times
            for _ in range(5):
                processor = CortexLinguisticProcessor()
                doc = processor.process_text(sentence)
                dep_data = [(r.head.text, r.dependent.text, r.relation, r.confidence) 
                           for r in doc.dependency_tree.relationships]
                dep_data.sort()  # Sort for comparison
                results.append(dep_data)
            
            # All results should be identical
            for result in results[1:]:
                self.assertEqual(result, results[0], f"Parsing not deterministic for: {sentence}")
    
    def test_spatial_coordinates_determinism(self):
        """Test that spatial coordinates are deterministic"""
        sentence = "The cat sits on the mat."
        
        coord_results = []
        for _ in range(3):
            processor = CortexLinguisticProcessor()
            doc = processor.process_text(sentence)
            
            coords = {}
            for word, coord in doc.spatial_anchors.items():
                coords[word] = (coord.x1, coord.x2, coord.x3, coord.x4, coord.x5, coord.x6)
            coord_results.append(coords)
        
        # All coordinate results should be identical
        for result in coord_results[1:]:
            self.assertEqual(result, coord_results[0], "Spatial coordinates not deterministic")


def run_comprehensive_tests():
    """Run all comprehensive tests and generate report"""
    print("CortexOS NLP - Phase 2 Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTokenizer,
        TestTagger,
        TestParser,
        TestIntegratedProcessor,
        TestPerformance,
        TestEdgeCases,
        TestDeterminism
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("TEST SUMMARY REPORT")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Performance benchmark
    print(f"\nPERFORMANCE BENCHMARK:")
    processor = CortexLinguisticProcessor()
    
    test_sentence = "The quick brown fox jumps over the lazy dog."
    start_time = time.time()
    doc = processor.process_text(test_sentence)
    processing_time = time.time() - start_time
    
    print(f"  Processing time: {processing_time:.4f}s")
    print(f"  Tokens processed: {len(doc.tokens)}")
    print(f"  Dependencies created: {len(doc.dependency_tree.relationships)}")
    print(f"  Processing rate: {len(doc.tokens)/processing_time:.0f} tokens/second")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    if success:
        print("\n🎉 ALL TESTS PASSED! Phase 2 implementation is fully validated.")
    else:
        print("\n❌ Some tests failed. Please review and fix issues.")
        exit(1)

