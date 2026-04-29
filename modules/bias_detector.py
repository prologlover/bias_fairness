"""
Bias Detector Module
Uses BERT models to detect gender, sentiment, and occupational biases in text.
Supports both English and Arabic languages.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from typing import Dict, List, Tuple
import re


class BiasDetector:
    """
    Detects various types of biases in text using BERT-based models.
    """
    
    def __init__(self, language='english'):
        """
        Initialize the bias detector with specified language.
        
        Args:
            language: 'english' or 'arabic'
        """
        self.language = language.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load appropriate model based on language
        if self.language == 'arabic':
            model_name = 'aubmindlab/bert-base-arabertv2'
        else:
            model_name = 'bert-base-uncased'
        
        print(f"Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Define bias word lists
        self._load_bias_word_lists()
    
    def _load_bias_word_lists(self):
        """Load predefined word lists for bias detection."""
        if self.language == 'english':
            self.gender_words = {
                'male': ['he', 'him', 'his', 'man', 'men', 'male', 'boy', 'father', 'brother', 'son', 'husband'],
                'female': ['she', 'her', 'hers', 'woman', 'women', 'female', 'girl', 'mother', 'sister', 'daughter', 'wife']
            }
            
            self.occupation_stereotypes = {
                'male_dominated': ['engineer', 'programmer', 'ceo', 'manager', 'doctor', 'scientist', 'pilot'],
                'female_dominated': ['nurse', 'teacher', 'secretary', 'receptionist', 'assistant', 'caregiver']
            }
            
            self.sentiment_words = {
                'positive': ['good', 'great', 'excellent', 'wonderful', 'amazing', 'brilliant', 'capable', 'strong'],
                'negative': ['bad', 'poor', 'weak', 'incompetent', 'inferior', 'limited', 'emotional']
            }
        else:  # Arabic
            self.gender_words = {
                'male': ['هو', 'له', 'رجل', 'رجال', 'ولد', 'أب', 'أخ', 'ابن', 'زوج'],
                'female': ['هي', 'لها', 'امرأة', 'نساء', 'بنت', 'أم', 'أخت', 'ابنة', 'زوجة']
            }
            
            self.occupation_stereotypes = {
                'male_dominated': ['مهندس', 'مبرمج', 'مدير', 'طبيب', 'عالم', 'طيار'],
                'female_dominated': ['ممرضة', 'معلمة', 'سكرتيرة', 'مساعدة', 'مربية']
            }
            
            self.sentiment_words = {
                'positive': ['جيد', 'ممتاز', 'رائع', 'قوي', 'قادر', 'نشيط'],
                'negative': ['سيئ', 'ضعيف', 'محدود', 'عاطفي', 'قليل']
            }
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get BERT embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of embeddings
        """
        encoded = self.tokenizer(texts, padding=True, truncation=True, 
                                return_tensors='pt', max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def detect_gender_bias(self, text: str) -> Dict:
        """
        Detect gender bias in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with bias analysis results
        """
        text_lower = text.lower()
        
        # Count gendered words
        male_count = sum(1 for word in self.gender_words['male'] if word in text_lower)
        female_count = sum(1 for word in self.gender_words['female'] if word in text_lower)
        
        total_gender_words = male_count + female_count
        
        if total_gender_words == 0:
            bias_score = 0
            bias_direction = 'neutral'
        else:
            # Calculate bias score (-1 to 1, negative = female bias, positive = male bias)
            bias_score = (male_count - female_count) / total_gender_words
            
            if abs(bias_score) < 0.2:
                bias_direction = 'neutral'
            elif bias_score > 0:
                bias_direction = 'male-biased'
            else:
                bias_direction = 'female-biased'
        
        # Check for occupation stereotypes
        occupation_bias = self._check_occupation_gender_association(text_lower)
        
        return {
            'bias_score': round(bias_score, 3),
            'bias_direction': bias_direction,
            'male_word_count': male_count,
            'female_word_count': female_count,
            'occupation_stereotypes': occupation_bias,
            'severity': self._get_severity(abs(bias_score))
        }
    
    def _check_occupation_gender_association(self, text: str) -> List[Dict]:
        """Check if occupations are associated with gender stereotypes."""
        stereotypes_found = []
        
        # Check for male-dominated occupations with female pronouns (counter-stereotypical)
        for occupation in self.occupation_stereotypes['male_dominated']:
            if occupation in text:
                has_female = any(word in text for word in self.gender_words['female'])
                has_male = any(word in text for word in self.gender_words['male'])
                
                if has_female and not has_male:
                    stereotypes_found.append({
                        'occupation': occupation,
                        'type': 'counter-stereotypical',
                        'gender': 'female',
                        'stereotype': 'male_dominated'
                    })
                elif has_male and not has_female:
                    stereotypes_found.append({
                        'occupation': occupation,
                        'type': 'stereotypical',
                        'gender': 'male',
                        'stereotype': 'male_dominated'
                    })
        
        # Check for female-dominated occupations
        for occupation in self.occupation_stereotypes['female_dominated']:
            if occupation in text:
                has_female = any(word in text for word in self.gender_words['female'])
                has_male = any(word in text for word in self.gender_words['male'])
                
                if has_male and not has_female:
                    stereotypes_found.append({
                        'occupation': occupation,
                        'type': 'counter-stereotypical',
                        'gender': 'male',
                        'stereotype': 'female_dominated'
                    })
                elif has_female and not has_male:
                    stereotypes_found.append({
                        'occupation': occupation,
                        'type': 'stereotypical',
                        'gender': 'female',
                        'stereotype': 'female_dominated'
                    })
        
        return stereotypes_found
    
    def detect_sentiment_bias(self, text: str) -> Dict:
        """
        Detect sentiment bias in association with gender.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment bias analysis
        """
        text_lower = text.lower()
        
        # Count sentiment words
        positive_count = sum(1 for word in self.sentiment_words['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_words['negative'] if word in text_lower)
        
        # Check gender-sentiment associations
        has_male = any(word in text_lower for word in self.gender_words['male'])
        has_female = any(word in text_lower for word in self.gender_words['female'])
        
        sentiment_score = 0
        if positive_count + negative_count > 0:
            sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        
        bias_type = 'none'
        if has_male and not has_female and negative_count > positive_count:
            bias_type = 'negative_male_association'
        elif has_female and not has_male and negative_count > positive_count:
            bias_type = 'negative_female_association'
        elif has_male and not has_female and positive_count > negative_count:
            bias_type = 'positive_male_association'
        elif has_female and not has_male and positive_count > negative_count:
            bias_type = 'positive_female_association'
        
        return {
            'sentiment_score': round(sentiment_score, 3),
            'positive_words': positive_count,
            'negative_words': negative_count,
            'bias_type': bias_type,
            'has_male_reference': has_male,
            'has_female_reference': has_female
        }
    
    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive bias analysis on text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Complete bias analysis results
        """
        gender_bias = self.detect_gender_bias(text)
        sentiment_bias = self.detect_sentiment_bias(text)
        
        # Calculate overall bias score
        overall_score = (abs(gender_bias['bias_score']) + 
                        (1 if sentiment_bias['bias_type'] != 'none' else 0)) / 2
        
        return {
            'text': text,
            'language': self.language,
            'gender_bias': gender_bias,
            'sentiment_bias': sentiment_bias,
            'overall_bias_score': round(overall_score, 3),
            'is_biased': overall_score > 0.3
        }
    
    def _get_severity(self, score: float) -> str:
        """Convert bias score to severity level."""
        if score < 0.2:
            return 'low'
        elif score < 0.5:
            return 'moderate'
        else:
            return 'high'
