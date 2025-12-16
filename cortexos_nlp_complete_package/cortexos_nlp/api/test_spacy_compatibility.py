"""
Standalone test for spaCy compatibility layer
"""

import sys
import os

# Add project root to path
sys.path.append('/home/ubuntu/cortexos_nlp')

def test_spacy_compatibility():
    """Test spaCy compatibility features."""
    print("Testing spaCy Compatibility Layer")
    print("=" * 40)
    
    try:
        # Test 1: Import compatibility layer
        print("\n1. Testing imports:")
        from api.spacy_compatibility import load, info, about, _MODELS
        print("   ✓ Successfully imported compatibility functions")
        print(f"   ✓ Available models: {list(_MODELS.keys())}")
        
        # Test 2: Model mapping
        print("\n2. Testing model mapping:")
        model_mappings = [
            ('en_core_web_sm', 'cortex_en_core'),
            ('en_core_web_md', 'cortex_en_core'),
            ('en_core_web_lg', 'cortex_en_core')
        ]
        
        for spacy_model, expected_cortex in model_mappings:
            if spacy_model in _MODELS:
                cortex_model = _MODELS[spacy_model]
                assert cortex_model == expected_cortex, f"Mapping failed: {spacy_model} -> {cortex_model}"
                print(f"   ✓ {spacy_model} -> {cortex_model}")
        
        # Test 3: About function
        print("\n3. Testing about function:")
        about_info = about()
        required_keys = ['cortexos_version', 'spacy_version', 'deterministic', 'mathematical_certainty']
        
        for key in required_keys:
            assert key in about_info, f"Missing key: {key}"
            print(f"   ✓ {key}: {about_info[key]}")
        
        # Test 4: Info function
        print("\n4. Testing info function:")
        available_models = info(silent=True)
        assert 'spacy_models' in available_models, "Missing spacy_models"
        assert 'cortex_models' in available_models, "Missing cortex_models"
        print(f"   ✓ spaCy models: {len(available_models['spacy_models'])}")
        print(f"   ✓ CortexOS models: {len(available_models['cortex_models'])}")
        
        # Test 5: Extension system
        print("\n5. Testing extension system:")
        from api.spacy_compatibility import ExtensionManager
        
        # Test extension management
        ExtensionManager.set_extension('Doc', 'test_attr', default='test_value')
        assert ExtensionManager.has_extension('Doc', 'test_attr'), "Extension not found"
        
        ext_config = ExtensionManager.get_extension('Doc', 'test_attr')
        assert ext_config is not None, "Extension config not found"
        assert ext_config['default'] == 'test_value', "Extension default value incorrect"
        
        print("   ✓ Extension set and retrieved successfully")
        
        # Test 6: Registry system
        print("\n6. Testing registry system:")
        from api.spacy_compatibility import registry
        
        # Test component registration
        @registry.component('test_component')
        def test_component(doc):
            return doc
        
        component = registry.get_component('test_component')
        assert component is not None, "Component not registered"
        print("   ✓ Component registration working")
        
        print("\n✅ spaCy Compatibility Layer Test: SUCCESS!")
        print("🎉 All compatibility features working correctly!")
        print("🔧 Ready for comprehensive testing!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_spacy_compatibility()
    if success:
        print("\n🚀 spaCy compatibility layer is ready!")
        print("📋 Next: Build comprehensive test suite")
    else:
        print("\n💥 Compatibility layer needs fixes")

