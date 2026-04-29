"""
Evaluator Module
Comprehensive evaluation framework for bias detection system.
"""

import os
from typing import List, Dict
from datetime import datetime
import json


class Evaluator:
    """
    Evaluate bias detection system performance.
    """
    
    def __init__(self, bias_detector, fairness_metrics, data_loader):
        """
        Initialize evaluator.
        
        Args:
            bias_detector: BiasDetector instance
            fairness_metrics: FairnessMetrics instance
            data_loader: DataLoader instance
        """
        self.bias_detector = bias_detector
        self.fairness_metrics = fairness_metrics
        self.data_loader = data_loader
    
    def run_benchmark(self, language='english') -> Dict:
        """
        Run comprehensive benchmark on dataset.
        
        Args:
            language: 'english' or 'arabic'
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Running {language.upper()} Benchmark")
        print(f"{'='*60}\n")
        
        # Load dataset
        print("Loading dataset...")
        dataset = self.data_loader.load_dataset(language)
        print(f"Loaded {len(dataset)} sentences\n")
        
        # Analyze unfiltered data
        print("Analyzing unfiltered text...")
        unfiltered_results = []
        for item in dataset:
            result = self.bias_detector.analyze_text(item['text'])
            result['original_label'] = item.get('label', 'unknown')
            result['stereotype_type'] = item.get('stereotype', 'unknown')
            unfiltered_results.append(result)
        
        # Create and analyze filtered data
        print("Creating filtered versions...")
        filtered_dataset = self.data_loader.create_filtered_version(dataset, self.bias_detector)
        
        print("Analyzing filtered text...")
        filtered_results = []
        for item in filtered_dataset:
            result = self.bias_detector.analyze_text(item['text'])
            result['original_label'] = item.get('label', 'unknown')
            result['stereotype_type'] = item.get('stereotype', 'unknown')
            result['original_text'] = item.get('original_text', '')
            filtered_results.append(result)
        
        # Calculate fairness metrics
        print("\nCalculating fairness metrics...")
        unfiltered_fairness = self.fairness_metrics.calculate_fairness_score(unfiltered_results)
        filtered_fairness = self.fairness_metrics.calculate_fairness_score(filtered_results)
        
        # Compare filtered vs unfiltered
        comparison = self.fairness_metrics.compare_filtered_unfiltered(
            unfiltered_results, 
            filtered_results
        )
        
        # Calculate StereoSet score
        stereoset_unfiltered = self.fairness_metrics.calculate_stereoset_score(dataset)
        
        # Compile results
        benchmark_results = {
            'language': language,
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(dataset),
            'unfiltered_analysis': {
                'results': unfiltered_results,
                'fairness_metrics': unfiltered_fairness,
                'stereoset_metrics': stereoset_unfiltered
            },
            'filtered_analysis': {
                'results': filtered_results,
                'fairness_metrics': filtered_fairness
            },
            'comparison': comparison,
            'summary': self._generate_summary(comparison, unfiltered_fairness, filtered_fairness)
        }
        
        print("\n" + "="*60)
        print("Benchmark Complete!")
        print("="*60)
        
        return benchmark_results
    
    def evaluate_weat(self, language='english') -> Dict:
        """
        Run WEAT evaluation.
        
        Args:
            language: 'english' or 'arabic'
            
        Returns:
            WEAT results
        """
        from modules.fairness_metrics import get_default_weat_word_sets
        
        print(f"\nRunning WEAT for {language}...")
        
        word_sets = get_default_weat_word_sets(language)
        
        weat_result = self.fairness_metrics.calculate_weat(
            target_words_1=word_sets['male_names'],
            target_words_2=word_sets['female_names'],
            attribute_words_1=word_sets['career_words'],
            attribute_words_2=word_sets['family_words']
        )
        
        print(f"WEAT Score: {weat_result['weat_score']:.4f}")
        print(f"Effect Size: {weat_result['effect_size']:.4f}")
        print(f"Interpretation: {weat_result['interpretation']}")
        print(f"Statistically Significant: {weat_result['is_significant']}")
        
        return weat_result
    
    def _generate_summary(self, comparison: Dict, 
                         unfiltered_fairness: Dict, 
                         filtered_fairness: Dict) -> Dict:
        """Generate human-readable summary."""
        return {
            'original_fairness_score': unfiltered_fairness['overall_fairness_score'],
            'filtered_fairness_score': filtered_fairness['overall_fairness_score'],
            'improvement': comparison['improvement'],
            'improvement_percentage': comparison['improvement_percentage'],
            'bias_reduction': comparison['bias_reduction'],
            'original_grade': unfiltered_fairness['grade'],
            'filtered_grade': filtered_fairness['grade'],
            'status': comparison['status'],
            'recommendation': self._get_recommendation(comparison)
        }
    
    def _get_recommendation(self, comparison: Dict) -> str:
        """Generate recommendation based on results."""
        improvement = comparison['improvement']
        
        if improvement > 10:
            return "Excellent improvement! The filtering significantly reduced bias."
        elif improvement > 5:
            return "Good improvement. The filtering reduced bias noticeably."
        elif improvement > 0:
            return "Modest improvement. Consider additional filtering strategies."
        elif improvement == 0:
            return "No improvement detected. Review filtering approach."
        else:
            return "Warning: Filtering may have introduced new biases."
    
    def generate_report(self, benchmark_results: Dict, output_dir='results') -> str:
        """
        Generate detailed evaluation report.
        
        Args:
            benchmark_results: Results from run_benchmark
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        language = benchmark_results['language']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_path = os.path.join(output_dir, f'benchmark_{language}_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
        
        # Generate text report
        report_path = os.path.join(output_dir, f'report_{language}_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"BIAS DETECTION EVALUATION REPORT - {language.upper()}\n")
            f.write("="*70 + "\n\n")
            
            summary = benchmark_results['summary']
            
            f.write("SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Dataset Size: {benchmark_results['dataset_size']} sentences\n")
            f.write(f"Language: {language}\n")
            f.write(f"Timestamp: {benchmark_results['timestamp']}\n\n")
            
            f.write("FAIRNESS SCORES\n")
            f.write("-"*70 + "\n")
            f.write(f"Original (Unfiltered):\n")
            f.write(f"  - Fairness Score: {summary['original_fairness_score']:.2f}/100\n")
            f.write(f"  - Grade: {summary['original_grade']}\n\n")
            
            f.write(f"Filtered:\n")
            f.write(f"  - Fairness Score: {summary['filtered_fairness_score']:.2f}/100\n")
            f.write(f"  - Grade: {summary['filtered_grade']}\n\n")
            
            f.write("IMPROVEMENT METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Absolute Improvement: {summary['improvement']:.2f} points\n")
            f.write(f"Relative Improvement: {summary['improvement_percentage']:.2f}%\n")
            f.write(f"Bias Reduction: {summary['bias_reduction']:.3f}\n")
            f.write(f"Status: {summary['status'].upper()}\n\n")
            
            f.write("RECOMMENDATION\n")
            f.write("-"*70 + "\n")
            f.write(f"{summary['recommendation']}\n\n")
            
            # StereoSet metrics
            stereoset = benchmark_results['unfiltered_analysis']['stereoset_metrics']
            f.write("STEREOSET METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Stereotype Count: {stereoset['stereotype_count']}\n")
            f.write(f"Anti-Stereotype Count: {stereoset['anti_stereotype_count']}\n")
            f.write(f"Unrelated Count: {stereoset['unrelated_count']}\n")
            f.write(f"StereoSet Score: {stereoset['stereoset_score']:.2f}/100\n")
            f.write(f"Interpretation: {stereoset['interpretation']}\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"\nReport saved to: {report_path}")
        print(f"JSON data saved to: {json_path}")
        
        return report_path
    
    def quick_test(self, text: str) -> Dict:
        """
        Quick test on a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        result = self.bias_detector.analyze_text(text)
        
        print("\n" + "="*60)
        print("BIAS ANALYSIS RESULTS")
        print("="*60)
        print(f"\nText: {text}")
        print(f"\nLanguage: {result['language']}")
        print(f"Overall Bias Score: {result['overall_bias_score']:.3f}")
        print(f"Is Biased: {result['is_biased']}")
        
        print(f"\nGender Bias:")
        gb = result['gender_bias']
        print(f"  - Score: {gb['bias_score']:.3f}")
        print(f"  - Direction: {gb['bias_direction']}")
        print(f"  - Severity: {gb['severity']}")
        
        print(f"\nSentiment Bias:")
        sb = result['sentiment_bias']
        print(f"  - Score: {sb['sentiment_score']:.3f}")
        print(f"  - Type: {sb['bias_type']}")
        
        print("\n" + "="*60)
        
        return result
