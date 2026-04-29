"""
Test script to verify the bias detection system works correctly.
Run this before running the full Streamlit app.
"""

import sys
import os

print("="*70)
print("BIAS & FAIRNESS ANALYZER - INSTALLATION TEST")
print("="*70)

# Test 1: Check Python version
print("\n[1/6] Checking Python version...")
print(f"Python {sys.version}")
if sys.version_info >= (3, 8):
    print("✓ Python version OK")
else:
    print("✗ Python 3.8+ required")
    sys.exit(1)

# Test 2: Import standard libraries
print("\n[2/6] Checking standard dependencies...")
try:
    import pandas
    import numpy
    print(f"✓ pandas {pandas.__version__}")
    print(f"✓ numpy {numpy.__version__}")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Import ML libraries
print("\n[3/6] Checking ML dependencies...")
try:
    import torch
    import transformers
    print(f"✓ torch {torch.__version__}")
    print(f"✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 4: Import Streamlit
print("\n[4/6] Checking Streamlit...")
try:
    import streamlit
    print(f"✓ streamlit {streamlit.__version__}")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 5: Import project modules
print("\n[5/6] Checking project modules...")
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from modules.bias_detector import BiasDetector
    from modules.fairness_metrics import FairnessMetrics
    from modules.data_loader import DataLoader
    from modules.evaluator import Evaluator
    print("✓ All project modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import project modules: {e}")
    sys.exit(1)

# Test 6: Quick functionality test
print("\n[6/6] Running quick functionality test...")
try:
    print("Loading English BiasDetector (this may take a minute on first run)...")
    detector = BiasDetector(language='english')
    print("✓ BiasDetector loaded")
    
    # Test analysis
    test_text = "The doctor asked the nurse to help him."
    print(f"\nTest text: '{test_text}'")
    result = detector.analyze_text(test_text)
    
    print(f"\nResults:")
    print(f"  Overall Bias Score: {result['overall_bias_score']:.3f}")
    print(f"  Is Biased: {result['is_biased']}")
    print(f"  Gender Bias Direction: {result['gender_bias']['bias_direction']}")
    print(f"  Gender Bias Score: {result['gender_bias']['bias_score']:.3f}")
    
    print("\n✓ Functionality test passed!")
    
except Exception as e:
    print(f"✗ Functionality test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("SUCCESS! All tests passed.")
print("="*70)
print("\nYou can now run the Streamlit app:")
print("  streamlit run app.py")
print("\n" + "="*70)
