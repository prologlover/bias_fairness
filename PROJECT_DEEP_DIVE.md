# Project Deep Dive: Architecture, Techniques, and Datasets

This document provides a detailed technical explanation of the **Bias & Fairness Analyzer** project, covering its architecture, underlying techniques, and the datasets used for analysis.

## 1. 🔬 Project Overview & Architecture

### **Core Function**
This project serves as a **Bilingual Bias & Fairness Analyzer** designed to detect gender and occupational stereotypes in LLM-generated text. It operates in real-time, providing immediate feedback and detailed metrics for both English and Arabic inputs.

### **System Architecture**
The application is built on a modular architecture:

1.  **Input Layer**:
    *   **Interface**: Streamlit Web Dashboard (`app.py`)
    *   **Inputs**: Single text entry or Batch JSON dataset upload
2.  **Processing Layer (`modules/`)**:
    *   **Bias Detector**: The core engine that loads the language-specific BERT model.
    *   **Data Loader**: Handles parsing of JSON datasets and text preprocessing.
3.  **Analysis Layer**:
    *   **Tokenization**: Converting text into tokens suitable for BERT.
    *   **Embedding Extraction**: Generating vector representations.
    *   **Metric Calculation**: Running statistical counts and vector math (cosine similarity).
4.  **Output Layer**:
    *   **Visualization**: Plotly charts for score distribution.
    *   **Reporting**: CSV/JSON exports of analysis results.

---

## 2. 🛠️ Techniques Used

The project utilizes a hybrid approach combining multiple NLP and Fairness AI techniques.

### **A. Natural Language Processing (NLP)**
*   **Transformer Models**:
    *   **English**: Uses `bert-base-uncased` (Hugging Face). A standard bidirectional transformer trained on lowercased English text.
    *   **Arabic**: Uses `aubmindlab/bert-base-arabertv2`. A specialized model pre-trained on a massive corpus of Arabic text to handle the language's complexity.
*   **Contextual Embeddings**:
    *   Instead of static word vectors (like Word2Vec), this project uses **Contextual Embeddings** from the BERT hidden states.
    *   Specifically, it extracts the `[CLS]` token embedding to represent the semantic meaning of entire sentences or target words in context.

### **B. Bias Detection Methodology**
The system employs two distinct detection strategies:

1.  **Lexicon-Based Detection (Explicit Bias)**:
    *   **Mechanism**: Counts occurrences of words from predefined lists.
    *   **Word Lists**:
        *   *Gender*: Male (he, him, man) vs. Female (she, her, woman).
        *   *Sentiment*: Positive (good, strong) vs. Negative (bad, weak).
    *   **Scoring**:
        *   `Bias Score = (Male Count - Female Count) / Total Gender Words`
        *   Result ranges from **-1.0** (Strongly Female Biased) to **+1.0** (Strongly Male Biased).

2.  **Embedding-Based Detection (Implicit Bias)**:
    *   **Mechanism**: Uses vector math to find hidden associations.
    *   **Technique**: Measures the **Cosine Similarity** between target concepts (e.g., "Doctor") and gender attributes. If "Doctor" is mathematically closer to "Man" than "Woman" in the vector space, bias is detected.

### **C. Fairness Metrics**
The project implements standard industry metrics to quantify fairness:

1.  **WEAT (Word Embedding Association Test)**:
    *   **Purpose**: The "gold standard" for measuring implicit bias in embeddings.
    *   **Calculation**: Compares the distribution of similarities between two sets of target words (e.g., Male/Female names) and two sets of attributes (e.g., Career/Family output).
    *   **Output**: An "Effect Size" and "P-value" (significance).

2.  **StereoSet-inspired Score**:
    *   **Purpose**: Evaluates sentence completion preferences.
    *   **Logic**:
        *   Does the model prefer the *Stereotypical* option? (e.g., "The doctor... *he*")
        *   Does it prefer the *Anti-stereotypical* option? (e.g., "The doctor... *she*")
    *   **Goal**: An ideal model should have a 50/50 split (Score = 50), indicating no preference based on stereotypes.

---

## 3. 📊 Datasets Used

The datasets are stored in the `data/` directory and serve as the "ground truth" for testing bias.

### **English: WinoBias Sample**
*   **File**: `data/english/winobias_sample.json`
*   **Origin**: A subset of the **WinoBias** dataset (Zhao et al., 2018).
*   **Format**: Paired sentences designed to test **Coreference Resolution**.
*   **Structure**:
    *   **Sentence A (Stereotypical)**: _"The **doctor** asked the nurse to help **him**."_
    *   **Sentence B (Anti-Stereotypical)**: _"The **doctor** asked the nurse to help **her**."_
*   **Purpose**: Tests if the model can correctly link the pronoun to the profession regardless of gender stereotypes.

### **Arabic: Custom Bias Dataset**
*   **File**: `data/arabic/arabic_bias_sample.json`
*   **Origin**: Custom-developed for this graduation project.
*   **Format**: Parallel structure to WinoBias but adapted for Arabic morphology.
*   **Challenges Addressed**:
    *   Arabic has gendered verbs and nouns (e.g., "Tabib" vs "Tabiba").
    *   The dataset includes examples that test these grammatical gender agreements against social stereotypes.
*   **Example**:
    *   _Text_: "طلب الطبيب من الممرضة أن تساعده" (Male doctor asked female nurse...).
    *   _Stereotype Tag_: `male_doctor_female_nurse`.
