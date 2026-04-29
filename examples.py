"""
Simple Example - Using the Bias Detector Directly

This script shows how to use the bias detection modules
without running the full Streamlit dashboard.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from modules.bias_detector import BiasDetector
from modules.fairness_metrics import FairnessMetrics, get_default_weat_word_sets
from modules.data_loader import DataLoader


def example_1_single_text_english():
    """Example 1: Analyze a single English text."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single English Text Analysis")
    print("="*70)
    
    # Initialize detector
    detector = BiasDetector(language='english')
    
    # Analyze text
    text = "The doctor asked the nurse to help him with the procedure."
    print(f"\nText: {text}")
    
    result = detector.analyze_text(text)
    
    print(f"\nResults:")
    print(f"  Overall Bias Score: {result['overall_bias_score']:.3f}")
    print(f"  Is Biased: {result['is_biased']}")
    print(f"  Gender Bias: {result['gender_bias']['bias_direction']} ({result['gender_bias']['severity']})")
    print(f"  Gender Score: {result['gender_bias']['bias_score']:.3f}")


def example_2_single_text_arabic():
    """Example 2: Analyze a single Arabic text."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Single Arabic Text Analysis")
    print("="*70)
    
    # Initialize detector for Arabic
    detector = BiasDetector(language='arabic')
    
    # Analyze Arabic text
    text = "طلب الطبيب من الممرضة أن تساعده في الإجراء."
    print(f"\nText: {text}")
    
    result = detector.analyze_text(text)
    
    print(f"\nResults:")
    print(f"  Overall Bias Score: {result['overall_bias_score']:.3f}")
    print(f"  Is Biased: {result['is_biased']}")
    print(f"  Gender Bias: {result['gender_bias']['bias_direction']}")


def example_3_batch_analysis():
    """Example 3: Batch analysis on dataset."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Analysis on English Dataset")
    print("="*70)
    
    # Initialize components
    detector = BiasDetector(language='english')
    fairness_metrics = FairnessMetrics(detector)
    data_loader = DataLoader()
    
    # Load dataset
    dataset = data_loader.load_dataset('english')
    print(f"\nLoaded {len(dataset)} sentences")
    
    # Analyze all texts
    results = []
    for item in dataset:
        result = detector.analyze_text(item['text'])
        results.append(result)
    
    # Calculate fairness score
    fairness = fairness_metrics.calculate_fairness_score(results)
    
    print(f"\nFairness Metrics:")
    print(f"  Overall Fairness Score: {fairness['overall_fairness_score']:.2f}/100")
    print(f"  Grade: {fairness['grade']}")
    print(f"  Biased Texts: {fairness['biased_count']}/{fairness['total_count']}")
    print(f"  Bias Percentage: {fairness['bias_percentage']:.1f}%")


def example_4_weat_analysis():
    """Example 4: WEAT analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 4: WEAT Analysis")
    print("="*70)
    
    # Initialize components
    detector = BiasDetector(language='english')
    fairness_metrics = FairnessMetrics(detector)
    
    # Get word sets
    word_sets = get_default_weat_word_sets('english')
    
    print("\nTarget Words:")
    print(f"  Male names: {', '.join(word_sets['male_names'][:4])}...")
    print(f"  Female names: {', '.join(word_sets['female_names'][:4])}...")
    
    print("\nAttribute Words:")
    print(f"  Career: {', '.join(word_sets['career_words'][:4])}...")
    print(f"  Family: {', '.join(word_sets['family_words'][:4])}...")
    
    # Calculate WEAT
    weat = fairness_metrics.calculate_weat(
        word_sets['male_names'],
        word_sets['female_names'],
        word_sets['career_words'],
        word_sets['family_words']
    )
    
    print(f"\nWEAT Results:")
    print(f"  WEAT Score: {weat['weat_score']:.4f}")
    print(f"  Effect Size: {weat['effect_size']:.4f}")
    print(f"  Interpretation: {weat['interpretation']}")
    print(f"  Statistically Significant: {weat['is_significant']}")


def example_5_filtered_comparison():
    """Example 5: Compare filtered vs unfiltered."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Filtered vs Unfiltered Comparison")
    print("="*70)
    
    # Initialize components
    detector = BiasDetector(language='english')
    fairness_metrics = FairnessMetrics(detector)
    data_loader = DataLoader()
    
    # Load dataset
    dataset = data_loader.load_dataset('english')
    
    # Analyze unfiltered
    print("\nAnalyzing unfiltered texts...")
    unfiltered_results = []
    for item in dataset:
        result = detector.analyze_text(item['text'])
        unfiltered_results.append(result)
    
    # Create filtered version
    print("Creating filtered versions...")
    filtered_dataset = data_loader.create_filtered_version(dataset, detector)
    
    # Analyze filtered
    print("Analyzing filtered texts...")
    filtered_results = []
    for item in filtered_dataset:
        result = detector.analyze_text(item['text'])
        filtered_results.append(result)
    
    # Compare
    comparison = fairness_metrics.compare_filtered_unfiltered(
        unfiltered_results,
        filtered_results
    )
    
    print("\n" + "-"*70)
    print("COMPARISON RESULTS")
    print("-"*70)
    print(f"\nUnfiltered:")
    print(f"  Fairness Score: {comparison['unfiltered']['overall_fairness_score']:.2f}/100")
    print(f"  Grade: {comparison['unfiltered']['grade']}")
    
    print(f"\nFiltered:")
    print(f"  Fairness Score: {comparison['filtered']['overall_fairness_score']:.2f}/100")
    print(f"  Grade: {comparison['filtered']['grade']}")
    
    print(f"\nImprovement:")
    print(f"  Absolute: +{comparison['improvement']:.2f} points")
    print(f"  Relative: {comparison['improvement_percentage']:.1f}%")
    print(f"  Bias Reduction: {comparison['bias_reduction']:.3f}")
    print(f"  Status: {comparison['status'].upper()}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "BIAS DETECTOR - USAGE EXAMPLES" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nThis script demonstrates how to use the bias detection modules.")
    print("Choose an example to run:")
    print("\n1. Single English Text Analysis")
    print("2. Single Arabic Text Analysis")
    print("3. Batch Analysis on Dataset")
    print("4. WEAT Analysis")
    print("5. Filtered vs Unfiltered Comparison")
    print("6. Run All Examples")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-6): ").strip()
    
    examples = {
        '1': example_1_single_text_english,
        '2': example_2_single_text_arabic,
        '3': example_3_batch_analysis,
        '4': example_4_weat_analysis,
        '5': example_5_filtered_comparison,
    }
    
    if choice == '0':
        print("\nGoodbye!")
        return
    elif choice == '6':
        # Run all examples
        for func in examples.values():
            func()
    elif choice in examples:
        examples[choice]()
    else:
        print("\nInvalid choice!")
        return
    
    print("\n" + "="*70)
    print("Done! For the full dashboard, run: streamlit run app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
