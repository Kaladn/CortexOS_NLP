"""
CortexOS NLP - Simplified Phase 2 Test Script

This script tests all Phase 2 components by running them directly
without complex import dependencies.
"""

import sys
import os
import time

# Add the project root to Python path
sys.path.append('/home/ubuntu/cortexos_nlp')

def test_integrated_processor():
    """Test the integrated processor which includes all components"""
    print("Testing CortexOS NLP Integrated Processor")
    print("=" * 50)
    
    # Import and initialize
    from linguistic.integrated_processor import CortexLinguisticProcessor
    processor = CortexLinguisticProcessor()
    
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe it's working so well!",
        "Complex sentences with subordinate clauses are challenging.",
        "The beautiful red car drives very fast down the winding road.",
        "Running quickly, the athlete finished the race successfully."
    ]
    
    print("1. Testing individual sentence processing:")
    all_docs = []
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n   {i}. Processing: '{sentence}'")
        
        start_time = time.time()
        doc = processor.process_text(sentence)
        processing_time = time.time() - start_time
        
        all_docs.append(doc)
        
        # Validate results
        assert doc.original_text == sentence, "Original text mismatch"
        assert len(doc.tokens) > 0, "No tokens generated"
        assert doc.dependency_tree is not None, "No dependency tree"
        assert doc.processing_metadata is not None, "No processing metadata"
        
        # Display results
        content_tokens = len([t for t in doc.tokens if not t.is_space])
        print(f"      ✓ Tokens: {len(doc.tokens)} (content: {content_tokens})")
        print(f"      ✓ Dependencies: {len(doc.dependency_tree.relationships)}")
        print(f"      ✓ Processing time: {processing_time:.4f}s")
        print(f"      ✓ Avg POS confidence: {doc.processing_metadata.average_pos_confidence:.3f}")
        print(f"      ✓ Avg dep confidence: {doc.processing_metadata.average_dependency_confidence:.3f}")
    
    print("\n2. Testing batch processing:")
    start_time = time.time()
    batch_docs = processor.batch_process(test_sentences)
    batch_time = time.time() - start_time
    
    assert len(batch_docs) == len(test_sentences), "Batch processing count mismatch"
    print(f"   ✓ Processed {len(batch_docs)} documents in {batch_time:.4f}s")
    print(f"   ✓ Average: {batch_time/len(batch_docs):.4f}s per document")
    
    print("\n3. Testing document similarity:")
    if len(all_docs) >= 2:
        similarity = processor.compare_documents(all_docs[0], all_docs[1])
        
        # Validate similarity metrics
        expected_metrics = ["spatial_similarity", "pos_similarity", "dependency_similarity", "structural_similarity"]
        for metric in expected_metrics:
            assert metric in similarity, f"Missing similarity metric: {metric}"
            score = similarity[metric]
            assert 0.0 <= score <= 1.0, f"Invalid similarity score: {score}"
        
        print(f"   ✓ Document 1 vs Document 2:")
        for metric, score in similarity.items():
            print(f"     - {metric}: {score:.3f}")
    
    print("\n4. Testing JSON export:")
    json_data = all_docs[0].to_json()
    
    # Validate JSON structure
    expected_keys = ["original_text", "tokens", "dependencies", "processing_metadata"]
    for key in expected_keys:
        assert key in json_data, f"Missing JSON key: {key}"
    
    assert isinstance(json_data["tokens"], list), "Tokens not a list in JSON"
    assert isinstance(json_data["dependencies"], list), "Dependencies not a list in JSON"
    
    print(f"   ✓ JSON export successful")
    print(f"   ✓ JSON keys: {list(json_data.keys())}")
    print(f"   ✓ Token count: {len(json_data['tokens'])}")
    print(f"   ✓ Dependency count: {len(json_data['dependencies'])}")
    
    print("\n5. Testing CoNLL-U export:")
    conllu_output = all_docs[0].to_conllu()
    
    assert isinstance(conllu_output, str), "CoNLL-U output not a string"
    assert len(conllu_output) > 0, "Empty CoNLL-U output"
    
    # Check basic CoNLL-U format
    lines = conllu_output.strip().split('\n')
    for line in lines:
        columns = line.split('\t')
        assert len(columns) == 10, f"Invalid CoNLL-U format: {len(columns)} columns"
    
    print(f"   ✓ CoNLL-U export successful")
    print(f"   ✓ Lines generated: {len(lines)}")
    
    print("\n6. Testing processing statistics:")
    stats = processor.get_processing_statistics()
    
    # Validate statistics
    assert stats["documents_processed"] > 0, "No documents processed"
    assert stats["total_tokens"] > 0, "No tokens processed"
    assert stats["total_relationships"] >= 0, "Invalid relationship count"
    
    print(f"   ✓ Documents processed: {stats['documents_processed']}")
    print(f"   ✓ Total tokens: {stats['total_tokens']}")
    print(f"   ✓ Total relationships: {stats['total_relationships']}")
    if stats.get('average_processing_time'):
        print(f"   ✓ Average processing time: {stats['average_processing_time']:.4f}s")
    
    return True


def test_determinism():
    """Test that processing is deterministic across multiple runs"""
    print("\nTesting Deterministic Behavior")
    print("=" * 50)
    
    from linguistic.integrated_processor import CortexLinguisticProcessor
    
    test_sentence = "The cat sits on the mat."
    
    print(f"Testing determinism with: '{test_sentence}'")
    
    # Run processing multiple times
    results = []
    for i in range(5):
        processor = CortexLinguisticProcessor()
        doc = processor.process_text(test_sentence)
        
        # Extract comparable data
        token_data = [(t.text, t.normalized, getattr(t, 'pos_tag', None)) for t in doc.tokens]
        dep_data = [(r.head.text, r.dependent.text, r.relation) for r in doc.dependency_tree.relationships]
        dep_data.sort()  # Sort for comparison
        
        results.append((token_data, dep_data))
    
    # Check that all results are identical
    for i, result in enumerate(results[1:], 2):
        assert result[0] == results[0][0], f"Tokenization not deterministic (run {i})"
        assert result[1] == results[0][1], f"Parsing not deterministic (run {i})"
    
    print("   ✓ Tokenization is deterministic across 5 runs")
    print("   ✓ POS tagging is deterministic across 5 runs")
    print("   ✓ Dependency parsing is deterministic across 5 runs")
    
    return True


def test_performance():
    """Test performance benchmarks"""
    print("\nTesting Performance Benchmarks")
    print("=" * 50)
    
    from linguistic.integrated_processor import CortexLinguisticProcessor
    processor = CortexLinguisticProcessor()
    
    # Test different sentence lengths
    test_cases = [
        ("Short.", "short sentence"),
        ("This is a medium length sentence with several words.", "medium sentence"),
        ("This is a much longer sentence that contains multiple clauses and various grammatical structures to test the performance of our deterministic NLP engine.", "long sentence")
    ]
    
    print("Performance by sentence length:")
    
    for sentence, description in test_cases:
        # Run multiple times for average
        times = []
        for _ in range(3):
            start_time = time.time()
            doc = processor.process_text(sentence)
            processing_time = time.time() - start_time
            times.append(processing_time)
        
        avg_time = sum(times) / len(times)
        content_tokens = len([t for t in doc.tokens if not t.is_space])
        
        print(f"   ✓ {description.capitalize()}: {avg_time:.4f}s avg ({content_tokens} content tokens)")
        
        # Performance assertions
        assert avg_time < 1.0, f"Processing too slow for {description}: {avg_time:.4f}s"
        assert content_tokens > 0, f"No content tokens for {description}"
    
    # Test batch performance
    batch_sentences = ["This is test sentence number {}.".format(i) for i in range(10)]
    
    start_time = time.time()
    batch_docs = processor.batch_process(batch_sentences)
    batch_time = time.time() - start_time
    
    assert len(batch_docs) == len(batch_sentences), "Batch processing failed"
    avg_per_doc = batch_time / len(batch_sentences)
    
    print(f"   ✓ Batch processing: {batch_time:.4f}s total, {avg_per_doc:.4f}s per document")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting Edge Cases")
    print("=" * 50)
    
    from linguistic.integrated_processor import CortexLinguisticProcessor
    processor = CortexLinguisticProcessor()
    
    edge_cases = [
        ("", "empty string"),
        ("Hello", "single word"),
        ("!!!", "punctuation only"),
        ("Hello@world.com", "special characters"),
        ("   ", "spaces only"),
        ("123", "numbers only"),
        ("Hello\nWorld", "newlines"),
        ("Very " * 50 + "long sentence", "very long sentence")
    ]
    
    print("Testing edge cases:")
    
    for text, description in edge_cases:
        try:
            doc = processor.process_text(text)
            
            # Basic validations
            assert doc.original_text == text, f"Original text mismatch for {description}"
            assert doc.dependency_tree is not None, f"No dependency tree for {description}"
            assert doc.processing_metadata is not None, f"No metadata for {description}"
            
            print(f"   ✓ {description}: processed successfully")
            
        except Exception as e:
            print(f"   ❌ {description}: failed with error: {str(e)}")
            return False
    
    return True


def main():
    """Run all tests"""
    print("CortexOS NLP - Phase 2 Validation Suite")
    print("=" * 60)
    print("Testing the complete deterministic NLP pipeline...")
    print()
    
    tests = [
        ("Integrated Processor", test_integrated_processor),
        ("Deterministic Behavior", test_determinism),
        ("Performance Benchmarks", test_performance),
        ("Edge Cases", test_edge_cases)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"RUNNING: {test_name}")
            print(f"{'='*60}")
            
            success = test_func()
            if success:
                print(f"\n✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"\n❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR - {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Phase 2 implementation is fully validated and ready!")
        print("✅ Mathematical determinism confirmed across all components!")
        print("✅ Performance benchmarks meet requirements!")
        print("✅ Edge cases handled gracefully!")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

