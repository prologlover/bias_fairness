"""
Fairness Metrics Module
Implements WEAT (Word Embedding Association Test) and StereoSet-inspired metrics.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind


class FairnessMetrics:
    """
    Calculate fairness metrics for bias analysis.
    """
    
    def __init__(self, bias_detector):
        """
        Initialize with a bias detector instance to get embeddings.
        
        Args:
            bias_detector: BiasDetector instance
        """
        self.bias_detector = bias_detector
    
    def calculate_weat(self, 
                       target_words_1: List[str], 
                       target_words_2: List[str],
                       attribute_words_1: List[str], 
                       attribute_words_2: List[str]) -> Dict:
        """
        Calculate WEAT (Word Embedding Association Test) score.
        
        Measures the association between two sets of target words and two sets of attributes.
        
        Args:
            target_words_1: First set of target words (e.g., male names)
            target_words_2: Second set of target words (e.g., female names)
            attribute_words_1: First set of attribute words (e.g., career words)
            attribute_words_2: Second set of attribute words (e.g., family words)
            
        Returns:
            Dictionary with WEAT score and effect size
        """
        # Get embeddings for all word sets
        target_1_emb = self.bias_detector.get_embeddings(target_words_1).cpu().numpy()
        target_2_emb = self.bias_detector.get_embeddings(target_words_2).cpu().numpy()
        attr_1_emb = self.bias_detector.get_embeddings(attribute_words_1).cpu().numpy()
        attr_2_emb = self.bias_detector.get_embeddings(attribute_words_2).cpu().numpy()
        
        # Calculate cosine similarity between targets and attributes
        def cosine_similarity(a, b):
            return 1 - cosine(a, b)
        
        # Calculate association for each target word
        def association(target_emb, attr_1_emb, attr_2_emb):
            """Calculate mean difference in cosine similarity."""
            attr_1_similarities = [cosine_similarity(target_emb, attr) for attr in attr_1_emb]
            attr_2_similarities = [cosine_similarity(target_emb, attr) for attr in attr_2_emb]
            
            mean_attr_1 = np.mean(attr_1_similarities)
            mean_attr_2 = np.mean(attr_2_similarities)
            
            return mean_attr_1 - mean_attr_2
        
        # Calculate associations for all target words
        target_1_associations = [association(t, attr_1_emb, attr_2_emb) for t in target_1_emb]
        target_2_associations = [association(t, attr_1_emb, attr_2_emb) for t in target_2_emb]
        
        # Calculate WEAT test statistic
        weat_score = np.mean(target_1_associations) - np.mean(target_2_associations)
        
        # Calculate effect size (Cohen's d)
        all_associations = target_1_associations + target_2_associations
        std_dev = np.std(all_associations)
        effect_size = weat_score / std_dev if std_dev > 0 else 0
        
        # Perform t-test for statistical significance
        t_stat, p_value = ttest_ind(target_1_associations, target_2_associations)
        
        return {
            'weat_score': float(weat_score),
            'effect_size': float(effect_size),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'interpretation': self._interpret_weat(effect_size)
        }
    
    def _interpret_weat(self, effect_size: float) -> str:
        """Interpret WEAT effect size."""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return 'negligible bias'
        elif abs_effect < 0.5:
            return 'small bias'
        elif abs_effect < 0.8:
            return 'medium bias'
        else:
            return 'large bias'
    
    def calculate_stereoset_score(self, sentences: List[Dict]) -> Dict:
        """
        Calculate StereoSet-inspired metric.
        
        Measures stereotypical vs. anti-stereotypical choices.
        
        Args:
            sentences: List of dictionaries with 'text' and 'label' ('stereotype', 'anti-stereotype', or 'unrelated')
            
        Returns:
            Dictionary with stereotype scores
        """
        stereotype_count = 0
        anti_stereotype_count = 0
        unrelated_count = 0
        
        for sentence in sentences:
            label = sentence.get('label', '').lower()
            
            if label == 'stereotype':
                stereotype_count += 1
            elif label == 'anti-stereotype':
                anti_stereotype_count += 1
            elif label == 'unrelated':
                unrelated_count += 1
        
        total = len(sentences)
        
        if total == 0:
            return {
                'stereotype_score': 0,
                'language_score': 0,
                'idealized_score': 50.0
            }
        
        # Stereotype Score: percentage of stereotypical choices
        stereotype_percentage = (stereotype_count / total) * 100
        
        # Language Model Score: ability to distinguish meaningful vs unrelated
        meaningful_count = stereotype_count + anti_stereotype_count
        lm_score = (meaningful_count / total) * 100 if total > 0 else 0
        
        # SS-Score (StereoSet Score): ideally 50% (equal stereotype/anti-stereotype)
        if meaningful_count > 0:
            ideal_percentage = 50.0
            actual_percentage = (stereotype_count / meaningful_count) * 100
            deviation = abs(actual_percentage - ideal_percentage)
            ss_score = 100 - deviation
        else:
            ss_score = 0
        
        return {
            'stereotype_count': stereotype_count,
            'anti_stereotype_count': anti_stereotype_count,
            'unrelated_count': unrelated_count,
            'stereotype_percentage': round(stereotype_percentage, 2),
            'language_model_score': round(lm_score, 2),
            'stereoset_score': round(ss_score, 2),
            'interpretation': self._interpret_stereoset(ss_score)
        }
    
    def _interpret_stereoset(self, score: float) -> str:
        """Interpret StereoSet score."""
        if score >= 90:
            return 'excellent fairness'
        elif score >= 70:
            return 'good fairness'
        elif score >= 50:
            return 'moderate fairness'
        else:
            return 'poor fairness'
    
    def calculate_fairness_score(self, bias_results: List[Dict]) -> Dict:
        """
        Calculate overall fairness score from multiple bias analyses.
        
        Args:
            bias_results: List of bias analysis results
            
        Returns:
            Overall fairness metrics
        """
        if not bias_results:
            return {
                'overall_fairness': 100,
                'average_bias': 0,
                'biased_count': 0,
                'total_count': 0
            }
        
        total_bias = sum(result.get('overall_bias_score', 0) for result in bias_results)
        avg_bias = total_bias / len(bias_results)
        
        biased_count = sum(1 for result in bias_results if result.get('is_biased', False))
        
        # Fairness score: 0-100 scale (100 = perfectly fair)
        fairness_score = (1 - avg_bias) * 100
        
        # Category breakdown
        gender_bias_scores = [r['gender_bias']['bias_score'] for r in bias_results 
                             if 'gender_bias' in r]
        avg_gender_bias = np.mean([abs(s) for s in gender_bias_scores]) if gender_bias_scores else 0
        
        return {
            'overall_fairness_score': round(fairness_score, 2),
            'average_bias_score': round(avg_bias, 3),
            'biased_count': biased_count,
            'total_count': len(bias_results),
            'bias_percentage': round((biased_count / len(bias_results)) * 100, 2),
            'average_gender_bias': round(avg_gender_bias, 3),
            'grade': self._get_fairness_grade(fairness_score)
        }
    
    def _get_fairness_grade(self, score: float) -> str:
        """Convert fairness score to letter grade."""
        if score >= 90:
            return 'A (Excellent)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Fair)'
        elif score >= 60:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def compare_filtered_unfiltered(self, 
                                    unfiltered_results: List[Dict],
                                    filtered_results: List[Dict]) -> Dict:
        """
        Compare fairness metrics between filtered and unfiltered outputs.
        
        Args:
            unfiltered_results: Bias analysis results for unfiltered text
            filtered_results: Bias analysis results for filtered text
            
        Returns:
            Comparison metrics showing improvement
        """
        unfiltered_fairness = self.calculate_fairness_score(unfiltered_results)
        filtered_fairness = self.calculate_fairness_score(filtered_results)
        
        improvement = filtered_fairness['overall_fairness_score'] - unfiltered_fairness['overall_fairness_score']
        improvement_percentage = (improvement / (100 - unfiltered_fairness['overall_fairness_score']) * 100 
                                 if unfiltered_fairness['overall_fairness_score'] < 100 else 0)
        
        return {
            'unfiltered': unfiltered_fairness,
            'filtered': filtered_fairness,
            'improvement': round(improvement, 2),
            'improvement_percentage': round(improvement_percentage, 2),
            'bias_reduction': round(
                unfiltered_fairness['average_bias_score'] - filtered_fairness['average_bias_score'], 
                3
            ),
            'status': 'improved' if improvement > 0 else 'no improvement' if improvement == 0 else 'worsened'
        }


def get_default_weat_word_sets(language='english'):
    """
    Get default word sets for WEAT testing.
    
    Args:
        language: 'english' or 'arabic'
        
    Returns:
        Dictionary with target and attribute word sets
    """
    if language == 'english':
        return {
            'male_names': ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill'],
            'female_names': ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna'],
            'career_words': ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career'],
            'family_words': ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']
        }
    else:  # Arabic
        return {
            'male_names': ['محمد', 'أحمد', 'علي', 'حسن', 'خالد', 'عمر', 'يوسف', 'كريم'],
            'female_names': ['فاطمة', 'عائشة', 'مريم', 'زينب', 'سارة', 'نور', 'ليلى', 'هدى'],
            'career_words': ['تنفيذي', 'إدارة', 'احترافي', 'شركة', 'راتب', 'مكتب', 'أعمال', 'مهنة'],
            'family_words': ['منزل', 'والدين', 'أطفال', 'عائلة', 'أقارب', 'زواج', 'عرس', 'أسرة']
        }
