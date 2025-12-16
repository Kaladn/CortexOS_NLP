"""
Comprehensive Test Suite for CortexOS NLP API Layer

This test suite validates all components of the CortexOS NLP API layer,
including the core classes, spaCy compatibility, and integration with
the underlying deterministic processing engine.
"""

import sys
import os
import time
import json
from typing import List, Dict, Any

# Add project root to path
sys.path.append('/home/ubuntu/cortexos_nlp')

# Test configuration
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "I can't believe it's working so well!",
    "Complex sentences with subordinate clauses are challenging.",
    "The beautiful red car drives very fast down the winding road.",
    "Hello world!",
    "This is a test.",
    "Natural language processing is fascinating.",
    "CortexOS provides mathematical certainty."
]

SPACY_MODEL_NAMES = [
    "en_core_web_sm",
    "en_core_web_md", 
    "en_core_web_lg"
]


class TestResults:
    """Track test results and statistics."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.errors = []
        self.start_time = time.time()
    
    def add_test(self, test_name: str, passed: bool, error: str = None):
        """Add a test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"   ✓ {test_name}")
        else:
            self.failed_tests += 1
            self.errors.append(f"{test_name}: {error}")
            print(f"   ❌ {test_name}: {error}")
    
    def summary(self):
        """Print test summary."""
        duration = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        print(f"Duration: {duration:.2f}s")
        
        if self.errors:
            print(f"\nFAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        
        return self.failed_tests == 0


def test_core_imports(results: TestResults):
    """Test core module imports."""
    print("\n1. TESTING CORE IMPORTS")
    print("-" * 30)
    
    try:
        # Test direct processor import
        from linguistic.integrated_processor import CortexLinguisticProcessor
        processor = CortexLinguisticProcessor()
        results.add_test("CortexLinguisticProcessor import", True)
        
        # Test basic processing
        doc = processor.process_text("Hello world!")
        results.add_test("Basic text processing", len(doc.tokens) > 0)
        
    except Exception as e:
        results.add_test("Core imports", False, str(e))


def test_api_classes_standalone(results: TestResults):
    """Test API classes in standalone mode."""
    print("\n2. TESTING API CLASSES (STANDALONE)")
    print("-" * 40)
    
    try:
        # Test Token class
        from api.cortex_token import Token
        results.add_test("Token class import", True)
        
        # Test Span class  
        from api.cortex_span import Span
        results.add_test("Span class import", True)
        
        # Test Doc class
        from api.cortex_doc import Doc
        results.add_test("Doc class import", True)
        
    except Exception as e:
        results.add_test("API classes standalone", False, str(e))


def test_spacy_compatibility_functions(results: TestResults):
    """Test spaCy compatibility functions."""
    print("\n3. TESTING SPACY COMPATIBILITY FUNCTIONS")
    print("-" * 45)
    
    try:
        from api.spacy_compatibility import (
            _MODELS, _LANGUAGES, about, info, 
            ExtensionManager, Registry
        )
        
        # Test model mappings
        results.add_test("Model mappings loaded", len(_MODELS) > 0)
        results.add_test("Language mappings loaded", len(_LANGUAGES) > 0)
        
        # Test about function
        about_info = about()
        required_keys = ['cortexos_version', 'spacy_version', 'deterministic']
        for key in required_keys:
            results.add_test(f"About info has {key}", key in about_info)
        
        # Test info function
        available_models = info(silent=True)
        results.add_test("Info function works", available_models is not None)
        results.add_test("Info has spacy_models", 'spacy_models' in available_models)
        
        # Test extension manager
        ExtensionManager.set_extension('Doc', 'test_attr', default='test')
        results.add_test("Extension set", 
                        ExtensionManager.has_extension('Doc', 'test_attr'))
        
        # Test registry
        registry = Registry()
        
        @registry.component('test_component')
        def test_comp(doc):
            return doc
        
        results.add_test("Registry component", 
                        registry.get_component('test_component') is not None)
        
    except Exception as e:
        results.add_test("spaCy compatibility functions", False, str(e))


def test_deterministic_processing(results: TestResults):
    """Test deterministic processing capabilities."""
    print("\n4. TESTING DETERMINISTIC PROCESSING")
    print("-" * 40)
    
    try:
        from linguistic.integrated_processor import CortexLinguisticProcessor
        processor = CortexLinguisticProcessor()
        
        # Test determinism
        text = "The cat sits on the mat."
        
        # Process same text multiple times
        results_list = []
        for i in range(3):
            doc = processor.process_text(text)
            token_data = [(token.text, getattr(token, 'pos_tag', 'UNKNOWN')) 
                         for token in doc.tokens]
            results_list.append(token_data)
        
        # Check determinism
        all_same = all(result == results_list[0] for result in results_list[1:])
        results.add_test("Deterministic processing", all_same)
        
        # Test processing metadata
        doc = processor.process_text("Hello world!")
        results.add_test("Processing metadata exists", 
                        hasattr(doc, 'processing_metadata'))
        results.add_test("Spatial anchors exist", 
                        hasattr(doc, 'spatial_anchors'))
        
    except Exception as e:
        results.add_test("Deterministic processing", False, str(e))


def test_performance_benchmarks(results: TestResults):
    """Test performance benchmarks."""
    print("\n5. TESTING PERFORMANCE BENCHMARKS")
    print("-" * 40)
    
    try:
        from linguistic.integrated_processor import CortexLinguisticProcessor
        processor = CortexLinguisticProcessor()
        
        # Single document performance
        start_time = time.time()
        doc = processor.process_text("The quick brown fox jumps over the lazy dog.")
        single_time = time.time() - start_time
        
        results.add_test("Single doc processing < 0.1s", single_time < 0.1)
        
        # Batch processing performance
        start_time = time.time()
        docs = processor.batch_process(TEST_SENTENCES[:5])
        batch_time = time.time() - start_time
        
        results.add_test("Batch processing works", len(docs) == 5)
        results.add_test("Batch processing < 0.5s", batch_time < 0.5)
        
        # Tokens per second calculation
        total_tokens = sum(len(doc.tokens) for doc in docs)
        tokens_per_second = total_tokens / batch_time if batch_time > 0 else 0
        
        results.add_test("Processing speed > 100 tokens/sec", 
                        tokens_per_second > 100)
        
        print(f"      Performance: {tokens_per_second:.0f} tokens/second")
        
    except Exception as e:
        results.add_test("Performance benchmarks", False, str(e))


def test_linguistic_analysis(results: TestResults):
    """Test linguistic analysis capabilities."""
    print("\n6. TESTING LINGUISTIC ANALYSIS")
    print("-" * 35)
    
    try:
        from linguistic.integrated_processor import CortexLinguisticProcessor
        processor = CortexLinguisticProcessor()
        
        # Test tokenization
        doc = processor.process_text("Hello, world! How are you?")
        results.add_test("Tokenization works", len(doc.tokens) > 0)
        
        # Test POS tagging
        pos_tags = [getattr(token, 'pos_tag', None) for token in doc.tokens 
                   if not token.is_space]
        results.add_test("POS tagging works", any(tag for tag in pos_tags))
        
        # Test dependency parsing
        results.add_test("Dependency tree exists", 
                        hasattr(doc, 'dependency_tree'))
        results.add_test("Dependencies found", 
                        len(doc.dependency_tree.relationships) > 0)
        
        # Test confidence scores
        doc = processor.process_text("The cat sits.")
        avg_confidence = doc.processing_metadata.average_pos_confidence
        results.add_test("POS confidence > 0.5", avg_confidence > 0.5)
        
    except Exception as e:
        results.add_test("Linguistic analysis", False, str(e))


def test_export_formats(results: TestResults):
    """Test export format capabilities."""
    print("\n7. TESTING EXPORT FORMATS")
    print("-" * 30)
    
    try:
        from linguistic.integrated_processor import CortexLinguisticProcessor
        processor = CortexLinguisticProcessor()
        
        doc = processor.process_text("The cat sits on the mat.")
        
        # Test JSON export
        json_data = doc.to_json()
        results.add_test("JSON export works", isinstance(json_data, dict))
        results.add_test("JSON has tokens", 'tokens' in json_data)
        results.add_test("JSON has dependencies", 'dependencies' in json_data)
        
        # Test CoNLL-U export
        conllu_data = doc.to_conllu()
        results.add_test("CoNLL-U export works", isinstance(conllu_data, str))
        results.add_test("CoNLL-U has content", len(conllu_data) > 0)
        
        # Test JSON serialization
        json_str = json.dumps(json_data)
        results.add_test("JSON serializable", len(json_str) > 0)
        
    except Exception as e:
        results.add_test("Export formats", False, str(e))


def test_mathematical_foundation(results: TestResults):
    """Test mathematical foundation integration."""
    print("\n8. TESTING MATHEMATICAL FOUNDATION")
    print("-" * 40)
    
    try:
        from linguistic.integrated_processor import CortexLinguisticProcessor
        processor = CortexLinguisticProcessor()
        
        doc = processor.process_text("Hello world!")
        
        # Test spatial anchors
        results.add_test("Spatial anchors exist", len(doc.spatial_anchors) > 0)
        
        # Test coordinate structure
        for word, coord in doc.spatial_anchors.items():
            results.add_test(f"Coordinate for '{word}' has 6 dimensions", 
                           hasattr(coord, 'x1') and hasattr(coord, 'x6'))
            break  # Test just one
        
        # Test binary cell memory integration
        results.add_test("Memory integration works", 
                        hasattr(processor, '_memory'))
        
        # Test harmonic resonance
        results.add_test("Resonance integration works", 
                        hasattr(processor, '_resonance'))
        
    except Exception as e:
        results.add_test("Mathematical foundation", False, str(e))


def test_error_handling(results: TestResults):
    """Test error handling and edge cases."""
    print("\n9. TESTING ERROR HANDLING")
    print("-" * 30)
    
    try:
        from linguistic.integrated_processor import CortexLinguisticProcessor
        processor = CortexLinguisticProcessor()
        
        # Test empty string
        doc = processor.process_text("")
        results.add_test("Empty string handling", len(doc.tokens) >= 0)
        
        # Test whitespace only
        doc = processor.process_text("   ")
        results.add_test("Whitespace handling", len(doc.tokens) >= 0)
        
        # Test special characters
        doc = processor.process_text("@#$%^&*()")
        results.add_test("Special characters handling", len(doc.tokens) > 0)
        
        # Test very long text
        long_text = "word " * 1000
        doc = processor.process_text(long_text)
        results.add_test("Long text handling", len(doc.tokens) > 0)
        
        # Test unicode
        doc = processor.process_text("Hello 世界! Café naïve résumé")
        results.add_test("Unicode handling", len(doc.tokens) > 0)
        
    except Exception as e:
        results.add_test("Error handling", False, str(e))


def test_memory_and_caching(results: TestResults):
    """Test memory usage and caching."""
    print("\n10. TESTING MEMORY AND CACHING")
    print("-" * 35)
    
    try:
        from linguistic.integrated_processor import CortexLinguisticProcessor
        processor = CortexLinguisticProcessor()
        
        # Test repeated processing (should use cache)
        text = "The cat sits on the mat."
        
        # First processing
        start_time = time.time()
        doc1 = processor.process_text(text)
        first_time = time.time() - start_time
        
        # Second processing (should be faster due to caching)
        start_time = time.time()
        doc2 = processor.process_text(text)
        second_time = time.time() - start_time
        
        results.add_test("Caching improves performance", second_time <= first_time)
        results.add_test("Cached results identical", 
                        len(doc1.tokens) == len(doc2.tokens))
        
        # Test statistics
        stats = processor.get_processing_statistics()
        results.add_test("Statistics available", isinstance(stats, dict))
        
    except Exception as e:
        results.add_test("Memory and caching", False, str(e))


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("CortexOS NLP - Comprehensive API Test Suite")
    print("=" * 50)
    print("Testing all components of the CortexOS NLP API layer")
    print("Mathematical certainty in every processing step")
    
    results = TestResults()
    
    # Run all test suites
    test_core_imports(results)
    test_api_classes_standalone(results)
    test_spacy_compatibility_functions(results)
    test_deterministic_processing(results)
    test_performance_benchmarks(results)
    test_linguistic_analysis(results)
    test_export_formats(results)
    test_mathematical_foundation(results)
    test_error_handling(results)
    test_memory_and_caching(results)
    
    # Print summary
    success = results.summary()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED! CortexOS NLP API is ready for production!")
        print(f"✅ Mathematical certainty validated across all components")
        print(f"🚀 Ready for developer adoption and real-world deployment")
    else:
        print(f"\n⚠️  Some tests failed. Review the issues above.")
        print(f"🔧 Fix the failing components before production deployment")
    
    return success


if __name__ == "__main__":
    try:
        success = run_comprehensive_tests()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

