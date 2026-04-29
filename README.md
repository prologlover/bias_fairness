# ⚖️ Bias & Fairness Analyzer for LLM Outputs

A graduation project for automatic bias detection in Large Language Model outputs, supporting both **English and Arabic** languages.

## 📋 Project Overview

This framework analyzes LLM outputs (summaries, answers, generated text) to detect potential biases in terms of:
- **Gender bias**
- **Occupational stereotypes**
- **Sentiment bias**

The system provides **real-time** interactive analysis through a Streamlit dashboard and implements industry-standard fairness metrics.

## ✨ Features

- 🌍 **Bilingual Support**: English and Arabic language analysis
- 🤖 **BERT-based Detection**: Uses pre-trained transformer models
- 📊 **Fairness Metrics**: WEAT and StereoSet implementation
- 📈 **Comprehensive Evaluation**: Compare filtered vs unfiltered outputs
- 💻 **Interactive Dashboard**: User-friendly Streamlit interface
- 📁 **Multiple Analysis Modes**:
  - Single text analysis
  - Batch dataset analysis
  - WEAT (Word Embedding Association Test)
  - Full benchmark evaluation
- 📊 **Rich Visualizations**: Charts, metrics, and comparison views
- 💾 **Export Results**: Download analysis as CSV/JSON

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **Framework**: Streamlit
- **ML Models**: 
  - BERT (bert-base-uncased) for English
  - AraBERT (aubmindlab/bert-base-arabertv2) for Arabic
- **Libraries**: 
  - transformers, torch
  - pandas, numpy, scipy
  - plotly, matplotlib

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Setup Instructions

1. **Clone or download the project**

```bash
cd "project of graduation"
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

This will install all required packages including PyTorch, Transformers, and Streamlit.

3. **Verify installation**

```bash
python -c "import streamlit; import torch; import transformers; print('All dependencies installed successfully!')"
```

## 🚀 Usage

### Running the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your web browser at `http://localhost:8501`

### Analysis Modes

#### 1. Single Text Analysis
- Enter any text in English or Arabic
- Get immediate bias detection results
- View detailed metrics and visualizations

#### 2. Batch Dataset Analysis
- Analyze entire dataset (WinoBias or Arabic samples)
- Calculate overall fairness scores
- Export results as CSV

#### 3. WEAT Analysis
- Test for implicit associations in word embeddings
- Compare male/female names with career/family attributes
- Statistical significance testing

#### 4. Full Benchmark
- Comprehensive evaluation on entire dataset
- Compare unfiltered vs filtered outputs
- Generate detailed reports
- Measure bias reduction

### Example Usage

```python
# Quick command-line test
from modules.bias_detector import BiasDetector

detector = BiasDetector(language='english')
result = detector.analyze_text("The doctor asked the nurse to help him.")

print(f"Gender Bias: {result['gender_bias']['bias_direction']}")
print(f"Overall Score: {result['overall_bias_score']}")
```

## 📂 Project Structure

```
project of graduation/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── modules/
│   ├── __init__.py
│   ├── bias_detector.py       # BERT-based bias detection
│   ├── fairness_metrics.py    # WEAT & StereoSet metrics
│   ├── data_loader.py         # Dataset management
│   └── evaluator.py           # Evaluation framework
├── data/
│   ├── english/
│   │   └── winobias_sample.json    # English test dataset
│   └── arabic/
│       └── arabic_bias_sample.json # Arabic test dataset
├── models/                    # Cached models (auto-created)
└── results/                   # Analysis results (auto-created)
```

## 📊 Datasets

### English: WinoBias
- 15 sample sentences with gender bias annotations
- Includes stereotypical, counter-stereotypical, and neutral examples
- Focuses on occupational gender stereotypes

### Arabic: Custom Bias Dataset
- 15 sample sentences in Arabic
- Parallel structure to WinoBias
- Covers common gender stereotypes in Arabic context

## 🔬 Methodology

### Bias Detection

1. **Tokenization**: Text is tokenized using BERT tokenizer
2. **Embedding**: Get contextualized embeddings from BERT
3. **Analysis**: 
   - Count gendered words and references
   - Detect occupational stereotypes
   - Measure sentiment associations
4. **Scoring**: Calculate bias scores on normalized scale

### Fairness Metrics

#### WEAT (Word Embedding Association Test)
- Measures association between target words (male/female names) and attributes (career/family)
- Effect size indicates strength of bias
- Statistical significance via t-test

#### StereoSet-inspired Metric
- Counts stereotypical vs anti-stereotypical choices
- Ideal score: 50% (no preference)
- Measures both bias and language modeling ability

### Filtering
- Simple gender-neutral pronoun replacement
- Converts gendered pronouns (he/she) to neutral (they)
- Demonstrates bias reduction potential

## 📈 Evaluation Results

The system is evaluated on:
- **Accuracy**: Detection of known biases
- **Fairness Score**: 0-100 scale (higher = more fair)
- **Improvement**: Filtered vs unfiltered comparison
- **Effect Size**: Statistical measure of bias strength

## 🎓 Academic Context

This project addresses the gap in **interactive, real-time bias detection tools** for LLM outputs. While bias in AI has been widely studied, few tools exist for educators and researchers to audit and analyze model outputs in production.

### Novelty
- Bilingual support (English + Arabic)
- Interactive dashboard for non-technical users
- Real-time analysis capability
- Comprehensive fairness metrics

## 📝 References

- **WinoBias**: Zhao et al. (2018) - "Gender Bias in Coreference Resolution"
- **WEAT**: Caliskan et al. (2017) - "Semantics derived automatically from language corpora contain human-like biases"
- **StereoSet**: Nadeem et al. (2020) - "StereoSet: Measuring stereotypical bias in pretrained language models"
- **AraBERT**: Antoun et al. (2020) - "AraBERT: Transformer-based Model for Arabic Language Understanding"

## 🤝 Contributing

This is a graduation project. For questions or suggestions, please contact the project author.

## 📄 License

This project is developed for academic purposes as part of a graduation project.

## 👤 Author

**Abubaker S. Issa**  
Graduation Project 2026  
Project: Bias & Fairness Analyzer for LLM Outputs

## 🙏 Acknowledgments

- BERT and AraBERT model developers
- WinoBias dataset creators
- Streamlit framework
- Hugging Face Transformers library

---

**Note**: This project requires internet connection on first run to download pre-trained models (~400MB for BERT, ~500MB for AraBERT). Models are cached locally after first download.
