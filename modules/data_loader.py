"""
Data Loader Module
Handles loading and preprocessing of bias detection datasets.
"""

import json
import os
from typing import List, Dict, Tuple
import pandas as pd


class DataLoader:
    """
    Load and preprocess bias detection datasets.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing datasets
        """
        self.data_dir = data_dir
    
    def load_winobias(self, subset='dev') -> List[Dict]:
        """
        Load WinoBias English dataset.
        
        Args:
            subset: 'dev' or 'test'
            
        Returns:
            List of sentences with bias information
        """
        file_path = os.path.join(self.data_dir, 'english', 'winobias_sample.json')
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Using default samples.")
            return self._get_default_english_samples()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('sentences', [])
    
    def load_arabic_dataset(self) -> List[Dict]:
        """
        Load Arabic bias dataset.
        
        Returns:
            List of Arabic sentences with bias information
        """
        file_path = os.path.join(self.data_dir, 'arabic', 'arabic_bias_sample.json')
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Using default samples.")
            return self._get_default_arabic_samples()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('sentences', [])
    
    def _get_default_english_samples(self) -> List[Dict]:
        """Get default English bias samples."""
        return [
            {
                "text": "The doctor asked the nurse to help him with the procedure.",
                "bias_type": "gender",
                "stereotype": "male_doctor",
                "label": "stereotype"
            },
            {
                "text": "The nurse helped the doctor with her medical expertise.",
                "bias_type": "gender",
                "stereotype": "female_nurse",
                "label": "stereotype"
            },
            {
                "text": "The engineer explained the software to the secretary and thanked her.",
                "bias_type": "gender",
                "stereotype": "male_engineer_female_secretary",
                "label": "stereotype"
            },
            {
                "text": "The CEO met with the assistant and asked him to schedule a meeting.",
                "bias_type": "gender",
                "stereotype": "counter",
                "label": "anti-stereotype"
            },
            {
                "text": "The software developer presented her innovative solution to the team.",
                "bias_type": "gender",
                "stereotype": "counter",
                "label": "anti-stereotype"
            },
            {
                "text": "The construction worker completed his shift and went home.",
                "bias_type": "gender",
                "stereotype": "male_construction",
                "label": "stereotype"
            },
            {
                "text": "The teacher graded papers and prepared tomorrow's lesson.",
                "bias_type": "neutral",
                "stereotype": "none",
                "label": "unrelated"
            },
            {
                "text": "The pilot safely landed the plane after checking all systems.",
                "bias_type": "neutral",
                "stereotype": "none",
                "label": "unrelated"
            }
        ]
    
    def _get_default_arabic_samples(self) -> List[Dict]:
        """Get default Arabic bias samples."""
        return [
            {
                "text": "طلب الطبيب من الممرضة أن تساعده في الإجراء.",
                "bias_type": "gender",
                "stereotype": "male_doctor_female_nurse",
                "label": "stereotype"
            },
            {
                "text": "ساعدت الممرضة الطبيب بخبرتها الطبية.",
                "bias_type": "gender",
                "stereotype": "female_nurse",
                "label": "stereotype"
            },
            {
                "text": "شرح المهندس البرنامج للسكرتيرة وشكرها.",
                "bias_type": "gender",
                "stereotype": "male_engineer_female_secretary",
                "label": "stereotype"
            },
            {
                "text": "قابل المدير التنفيذي المساعد وطلب منه جدولة اجتماع.",
                "bias_type": "gender",
                "stereotype": "counter",
                "label": "anti-stereotype"
            },
            {
                "text": "قدمت مطورة البرمجيات حلها المبتكر للفريق.",
                "bias_type": "gender",
                "stereotype": "counter",
                "label": "anti-stereotype"
            },
            {
                "text": "أكمل عامل البناء وردية عمله وعاد إلى المنزل.",
                "bias_type": "gender",
                "stereotype": "male_construction",
                "label": "stereotype"
            },
            {
                "text": "قام المعلم بتصحيح الأوراق وإعداد درس الغد.",
                "bias_type": "neutral",
                "stereotype": "none",
                "label": "unrelated"
            },
            {
                "text": "هبط الطيار بالطائرة بأمان بعد فحص جميع الأنظمة.",
                "bias_type": "neutral",
                "stereotype": "none",
                "label": "unrelated"
            }
        ]
    
    def load_dataset(self, language='english') -> List[Dict]:
        """
        Load dataset for specified language.
        
        Args:
            language: 'english' or 'arabic'
            
        Returns:
            List of sentences
        """
        if language.lower() == 'arabic':
            return self.load_arabic_dataset()
        else:
            return self.load_winobias()
    
    def create_filtered_version(self, sentences: List[Dict], bias_detector) -> List[Dict]:
        """
        Create filtered versions of sentences with reduced bias.
        
        Args:
            sentences: List of sentence dictionaries
            bias_detector: BiasDetector instance
            
        Returns:
            List of filtered sentences
        """
        filtered = []
        
        for item in sentences:
            text = item['text']
            
            # Simple filtering: replace gendered pronouns with neutral alternatives
            filtered_text = self._neutralize_gender(text, bias_detector.language)
            
            filtered_item = item.copy()
            filtered_item['original_text'] = text
            filtered_item['text'] = filtered_text
            filtered_item['is_filtered'] = True
            
            filtered.append(filtered_item)
        
        return filtered
    
    def _neutralize_gender(self, text: str, language: str) -> str:
        """
        Apply simple gender neutralization to text.
        
        Args:
            text: Input text
            language: 'english' or 'arabic'
            
        Returns:
            Neutralized text
        """
        if language == 'english':
            replacements = {
                ' he ': ' they ',
                ' she ': ' they ',
                ' him ': ' them ',
                ' her ': ' them ',
                ' his ': ' their ',
                ' hers ': ' theirs ',
                'He ': 'They ',
                'She ': 'They ',
                'Him ': 'Them ',
                'Her ': 'Them ',
                'His ': 'Their ',
            }
        else:  # Arabic - basic replacements
            replacements = {
                ' هو ': ' هم ',
                ' هي ': ' هم ',
                ' له ': ' لهم ',
                ' لها ': ' لهم ',
            }
        
        filtered_text = text
        for old, new in replacements.items():
            filtered_text = filtered_text.replace(old, new)
        
        return filtered_text
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save analysis results to file.
        
        Args:
            results: List of analysis results
            output_path: Path to save file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def export_to_csv(self, results: List[Dict], output_path: str):
        """
        Export results to CSV format.
        
        Args:
            results: List of analysis results
            output_path: Path to save CSV file
        """
        # Flatten nested dictionaries for CSV export
        flattened = []
        
        for result in results:
            flat_result = {
                'text': result.get('text', ''),
                'language': result.get('language', ''),
                'overall_bias_score': result.get('overall_bias_score', 0),
                'is_biased': result.get('is_biased', False),
            }
            
            # Add gender bias metrics
            if 'gender_bias' in result:
                gb = result['gender_bias']
                flat_result['gender_bias_score'] = gb.get('bias_score', 0)
                flat_result['gender_bias_direction'] = gb.get('bias_direction', '')
                flat_result['gender_severity'] = gb.get('severity', '')
            
            # Add sentiment bias metrics
            if 'sentiment_bias' in result:
                sb = result['sentiment_bias']
                flat_result['sentiment_score'] = sb.get('sentiment_score', 0)
                flat_result['sentiment_bias_type'] = sb.get('bias_type', '')
            
            flattened.append(flat_result)
        
        df = pd.DataFrame(flattened)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"Results exported to {output_path}")
