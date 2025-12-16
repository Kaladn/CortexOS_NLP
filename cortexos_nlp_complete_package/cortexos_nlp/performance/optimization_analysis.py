"""
CortexOS NLP - Performance Optimization Analysis

This module analyzes the performance characteristics of the CortexOS NLP engine
and provides optimization recommendations for production deployment.
"""

import time
import sys
import os
import gc
import psutil
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path
sys.path.append('/home/ubuntu/cortexos_nlp')

@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""
    operation: str
    execution_time: float
    memory_usage: int
    tokens_processed: int
    tokens_per_second: float
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PerformanceAnalyzer:
    """Analyzes and optimizes CortexOS NLP performance."""
    
    def __init__(self):
        self.metrics = []
        self.baseline_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss // 1024 // 1024
    
    def _count_tokens(self, text: str) -> int:
        """Simple token counting for performance metrics."""
        return len(text.split())
    
    def measure_operation(self, operation_name: str, func, *args, **kwargs) -> Any:
        """
        Measure the performance of an operation.
        
        Args:
            operation_name: Name of the operation being measured
            func: Function to execute and measure
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        # Force garbage collection for accurate memory measurement
        gc.collect()
        
        # Record initial state
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Execute operation
        result = func(*args, **kwargs)
        
        # Record final state
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Estimate tokens processed (rough approximation)
        tokens_processed = 0
        if args and isinstance(args[0], str):
            tokens_processed = self._count_tokens(args[0])
        elif args and isinstance(args[0], list):
            tokens_processed = sum(self._count_tokens(text) for text in args[0] if isinstance(text, str))
        
        tokens_per_second = tokens_processed / execution_time if execution_time > 0 else 0
        
        # Store metrics
        metrics = PerformanceMetrics(
            operation=operation_name,
            execution_time=execution_time,
            memory_usage=memory_delta,
            tokens_processed=tokens_processed,
            tokens_per_second=tokens_per_second
        )
        
        self.metrics.append(metrics)
        
        return result
    
    def benchmark_core_operations(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark core CortexOS operations."""
        print("🔍 Benchmarking Core Operations")
        print("-" * 35)
        
        benchmarks = {}
        
        try:
            # Test mathematical foundation components
            from core.spatial_anchor import SpatialAnchor
            from core.binary_cell_memory import BinaryCellMemory, RelationshipType
            from core.harmonic_resonance import HarmonicResonance
            
            # Benchmark spatial anchor generation
            anchor = SpatialAnchor()
            result = self.measure_operation(
                "spatial_anchor_generation",
                lambda: [anchor.get_coordinate(f"word_{i}") for i in range(100)]
            )
            benchmarks["spatial_anchor"] = self.metrics[-1]
            print(f"   ✓ Spatial Anchor: {self.metrics[-1].tokens_per_second:.0f} ops/sec")
            
            # Benchmark memory operations
            memory = BinaryCellMemory()
            coords = [anchor.get_coordinate(f"word_{i}") for i in range(10)]
            
            result = self.measure_operation(
                "memory_storage",
                lambda: [memory.store_relationship(coords[i], coords[(i+1)%10], 
                                                 RelationshipType.SYNONYM, 0.8) 
                        for i in range(10)]
            )
            benchmarks["memory_storage"] = self.metrics[-1]
            print(f"   ✓ Memory Storage: {self.metrics[-1].execution_time*1000:.2f}ms for 10 ops")
            
            # Benchmark harmonic resonance
            resonance = HarmonicResonance(memory)
            result = self.measure_operation(
                "harmonic_resonance",
                lambda: [resonance.calculate_similarity(coords[0], coords[i]) 
                        for i in range(1, 10)]
            )
            benchmarks["harmonic_resonance"] = self.metrics[-1]
            print(f"   ✓ Harmonic Resonance: {self.metrics[-1].execution_time*1000:.2f}ms for 9 ops")
            
        except Exception as e:
            print(f"   ❌ Core operations benchmark failed: {str(e)}")
        
        return benchmarks
    
    def benchmark_linguistic_processing(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark linguistic processing components."""
        print("\n🔍 Benchmarking Linguistic Processing")
        print("-" * 40)
        
        benchmarks = {}
        
        try:
            from linguistic.integrated_processor import CortexLinguisticProcessor
            processor = CortexLinguisticProcessor()
            
            # Test sentences of varying complexity
            test_sentences = [
                "Hello world!",
                "The quick brown fox jumps over the lazy dog.",
                "Complex sentences with subordinate clauses require more processing time.",
                "The beautiful red car drives very fast down the winding mountain road while the sun sets behind the distant hills."
            ]
            
            # Benchmark single document processing
            for i, sentence in enumerate(test_sentences):
                result = self.measure_operation(
                    f"single_doc_processing_{i+1}",
                    processor.process_text,
                    sentence
                )
                benchmarks[f"single_doc_{i+1}"] = self.metrics[-1]
                print(f"   ✓ Doc {i+1} ({len(sentence.split())} tokens): "
                      f"{self.metrics[-1].execution_time*1000:.2f}ms, "
                      f"{self.metrics[-1].tokens_per_second:.0f} tokens/sec")
            
            # Benchmark batch processing
            result = self.measure_operation(
                "batch_processing",
                processor.batch_process,
                test_sentences
            )
            benchmarks["batch_processing"] = self.metrics[-1]
            print(f"   ✓ Batch Processing: {self.metrics[-1].execution_time*1000:.2f}ms, "
                  f"{self.metrics[-1].tokens_per_second:.0f} tokens/sec")
            
            # Benchmark repeated processing (caching test)
            repeated_sentence = "The cat sits on the mat."
            
            # First processing (cache miss)
            result = self.measure_operation(
                "first_processing",
                processor.process_text,
                repeated_sentence
            )
            first_time = self.metrics[-1].execution_time
            
            # Second processing (cache hit)
            result = self.measure_operation(
                "cached_processing",
                processor.process_text,
                repeated_sentence
            )
            cached_time = self.metrics[-1].execution_time
            
            benchmarks["caching_improvement"] = cached_time / first_time if first_time > 0 else 1.0
            print(f"   ✓ Caching Improvement: {(1 - cached_time/first_time)*100:.1f}% faster")
            
        except Exception as e:
            print(f"   ❌ Linguistic processing benchmark failed: {str(e)}")
        
        return benchmarks
    
    def benchmark_api_layer(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark API layer performance."""
        print("\n🔍 Benchmarking API Layer")
        print("-" * 30)
        
        benchmarks = {}
        
        try:
            # Test API class instantiation and usage
            from api.cortex_doc import Doc
            from api.cortex_token import Token
            from api.cortex_span import Span
            
            # Create mock data for API testing
            mock_tokens = [
                {"text": "The", "pos_tag": "DET", "start": 0, "end": 3},
                {"text": "cat", "pos_tag": "NOUN", "start": 4, "end": 7},
                {"text": "sits", "pos_tag": "VERB", "start": 8, "end": 12}
            ]
            
            # Benchmark Doc creation
            result = self.measure_operation(
                "doc_creation",
                lambda: [Doc("The cat sits.", mock_tokens, {}, {}) for _ in range(100)]
            )
            benchmarks["doc_creation"] = self.metrics[-1]
            print(f"   ✓ Doc Creation: {self.metrics[-1].execution_time*1000:.2f}ms for 100 docs")
            
            # Benchmark Token operations
            doc = Doc("The cat sits.", mock_tokens, {}, {})
            result = self.measure_operation(
                "token_operations",
                lambda: [token.text + token.pos_ for token in doc for _ in range(10)]
            )
            benchmarks["token_operations"] = self.metrics[-1]
            print(f"   ✓ Token Operations: {self.metrics[-1].execution_time*1000:.2f}ms")
            
            # Benchmark export operations
            result = self.measure_operation(
                "json_export",
                lambda: [doc.to_json() for _ in range(50)]
            )
            benchmarks["json_export"] = self.metrics[-1]
            print(f"   ✓ JSON Export: {self.metrics[-1].execution_time*1000:.2f}ms for 50 exports")
            
        except Exception as e:
            print(f"   ❌ API layer benchmark failed: {str(e)}")
        
        return benchmarks
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        print("\n🔍 Analyzing Memory Usage")
        print("-" * 30)
        
        memory_analysis = {}
        
        try:
            # Get current process info
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            memory_analysis.update({
                "rss_mb": memory_info.rss // 1024 // 1024,
                "vms_mb": memory_info.vms // 1024 // 1024,
                "peak_memory_mb": max(m.memory_usage for m in self.metrics) if self.metrics else 0,
                "average_memory_delta": sum(m.memory_usage for m in self.metrics) / len(self.metrics) if self.metrics else 0
            })
            
            print(f"   ✓ Current RSS: {memory_analysis['rss_mb']} MB")
            print(f"   ✓ Current VMS: {memory_analysis['vms_mb']} MB")
            print(f"   ✓ Peak Delta: {memory_analysis['peak_memory_mb']} MB")
            print(f"   ✓ Avg Delta: {memory_analysis['average_memory_delta']:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ Memory analysis failed: {str(e)}")
        
        return memory_analysis
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on benchmarks."""
        recommendations = []
        
        if not self.metrics:
            return ["No performance data available for analysis."]
        
        # Analyze execution times
        avg_time = sum(m.execution_time for m in self.metrics) / len(self.metrics)
        slow_operations = [m for m in self.metrics if m.execution_time > avg_time * 2]
        
        if slow_operations:
            recommendations.append(
                f"🐌 Optimize slow operations: {', '.join(op.operation for op in slow_operations[:3])}"
            )
        
        # Analyze memory usage
        high_memory_ops = [m for m in self.metrics if m.memory_usage > 10]  # >10MB
        if high_memory_ops:
            recommendations.append(
                f"🧠 Reduce memory usage in: {', '.join(op.operation for op in high_memory_ops[:3])}"
            )
        
        # Analyze throughput
        low_throughput_ops = [m for m in self.metrics if m.tokens_per_second < 100 and m.tokens_processed > 0]
        if low_throughput_ops:
            recommendations.append(
                f"⚡ Improve throughput for: {', '.join(op.operation for op in low_throughput_ops[:3])}"
            )
        
        # General recommendations
        recommendations.extend([
            "🔄 Implement connection pooling for concurrent processing",
            "💾 Add persistent caching for frequently accessed data",
            "🎯 Use lazy loading for large linguistic resources",
            "🔧 Consider C++ extensions for critical path operations",
            "📊 Implement batch processing for high-volume scenarios",
            "🗜️ Add compression for spatial coordinate storage",
            "⚡ Use memory mapping for large binary cell memory files",
            "🎨 Implement progressive loading for API responses"
        ])
        
        return recommendations
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        report_lines = []
        
        report_lines.extend([
            "CortexOS NLP - Performance Analysis Report",
            "=" * 50,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Operations Measured: {len(self.metrics)}",
            ""
        ])
        
        if self.metrics:
            # Summary statistics
            total_time = sum(m.execution_time for m in self.metrics)
            total_tokens = sum(m.tokens_processed for m in self.metrics)
            avg_throughput = total_tokens / total_time if total_time > 0 else 0
            
            report_lines.extend([
                "PERFORMANCE SUMMARY",
                "-" * 20,
                f"Total Execution Time: {total_time:.3f}s",
                f"Total Tokens Processed: {total_tokens}",
                f"Average Throughput: {avg_throughput:.0f} tokens/second",
                f"Memory Baseline: {self.baseline_memory} MB",
                ""
            ])
            
            # Top performers
            fastest_ops = sorted(self.metrics, key=lambda m: m.execution_time)[:3]
            slowest_ops = sorted(self.metrics, key=lambda m: m.execution_time, reverse=True)[:3]
            
            report_lines.extend([
                "FASTEST OPERATIONS",
                "-" * 18,
            ])
            for op in fastest_ops:
                report_lines.append(f"  {op.operation}: {op.execution_time*1000:.2f}ms")
            
            report_lines.extend([
                "",
                "SLOWEST OPERATIONS", 
                "-" * 18,
            ])
            for op in slowest_ops:
                report_lines.append(f"  {op.operation}: {op.execution_time*1000:.2f}ms")
            
            report_lines.append("")
        
        # Optimization recommendations
        recommendations = self.generate_optimization_recommendations()
        report_lines.extend([
            "OPTIMIZATION RECOMMENDATIONS",
            "-" * 30,
        ])
        for rec in recommendations:
            report_lines.append(f"  {rec}")
        
        return "\n".join(report_lines)


def run_performance_analysis():
    """Run comprehensive performance analysis."""
    print("CortexOS NLP - Performance Optimization Analysis")
    print("=" * 50)
    print("Analyzing performance characteristics and optimization opportunities")
    
    analyzer = PerformanceAnalyzer()
    
    # Run benchmarks
    core_benchmarks = analyzer.benchmark_core_operations()
    linguistic_benchmarks = analyzer.benchmark_linguistic_processing()
    api_benchmarks = analyzer.benchmark_api_layer()
    memory_analysis = analyzer.analyze_memory_usage()
    
    # Generate report
    print("\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 50)
    
    report = analyzer.generate_performance_report()
    print(report)
    
    # Save report to file
    report_path = "/home/ubuntu/cortexos_nlp/performance/performance_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Full report saved to: {report_path}")
    
    return {
        "core_benchmarks": core_benchmarks,
        "linguistic_benchmarks": linguistic_benchmarks,
        "api_benchmarks": api_benchmarks,
        "memory_analysis": memory_analysis,
        "report_path": report_path
    }


if __name__ == "__main__":
    try:
        results = run_performance_analysis()
        print("\n🎯 Performance analysis complete!")
        print("🚀 Ready for production optimization!")
    except Exception as e:
        print(f"\n💥 Performance analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

