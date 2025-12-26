# Mental Health Detection using Machine Learning

A machine learning project that classifies mental health conditions from text statements using Natural Language Processing (NLP) and traditional ML algorithms.

## 📊 Project Overview

This project uses the **Kaggle Mental Health Sentiment Analysis dataset** to classify text statements into 7 different mental health categories:
- Normal
- Depression
- Suicidal
- Anxiety
- Bipolar
- Stress
- Personality Disorder

**Achieved Accuracy: 80.06%** using Logistic Regression with optimized TF-IDF features.

## 🎯 Key Features

- **Advanced Text Preprocessing**: Regex-based cleaning, URL/email removal, short word filtering
- **Dual TF-IDF Vectorization**: 
  - Word-level n-grams (1-3 grams, 15,000 features)
  - Character-level n-grams (2-4 grams, 5,000 features)
  - Total: 20,000 combined features
- **Multiple ML Models**: Logistic Regression, SVM
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualizations**: Confusion matrices, comparison charts, class distribution

## 📁 Dataset

**Source**: [Kaggle - Mental Health Sentiment Analysis](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

**Statistics**:
- Original samples: 53,043
- After cleaning: 51,060
- Training samples: 40,848 (80%)
- Testing samples: 10,212 (20%)

**Class Distribution**:
- Normal: 30.8%
- Depression: 29.0%
- Suicidal: 20.1%
- Anxiety: 7.3%
- Bipolar: 5.4%
- Stress: 5.0%
- Personality Disorder: 2.3%

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- Jupyter Notebook

### Install Dependencies
```bash
pip install matplotlib seaborn scikit-learn pandas numpy scipy
```

## 💻 Usage

1. **Download the dataset**:
   - Get `Combined Data.csv` from [Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
   - Place it in the same directory as the notebook

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook mental_health_detection.ipynb
   ```

3. **Run the cells in order**:
   - Cell 1: Install packages
   - Cell 4: Import libraries
   - Cell 6: Load dataset
   - Cell 12: Clean data
   - Cell 13: Text preprocessing
   - Cell 15: Data preparation
   - Cell 16: TF-IDF feature extraction
   - Cell 18: Initialize results
   - Cell 20: Train Logistic Regression
   - Cell 22: Train SVM
   - Cell 24+: Visualizations and analysis

## 📈 Model Performance

### Best Model: Logistic Regression

| Metric | Score |
|--------|-------|
| **Accuracy** | **80.06%** |
| **Precision** | **79.87%** |
| **Recall** | **80.00%** |
| **F1-Score** | **79.88%** |

### Model Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 80.06% |
| SVM | 74.26% |

### Performance Context
- **Random Baseline**: 14.3% (1 in 7 classes)
- **Improvement**: 5.6× better than random guessing
- **Traditional ML Benchmark**: 75-80% for 7-class text classification problems
- **Deep Learning Potential**: Transformer models (BERT, RoBERTa) could achieve 85-92%

## 🛠️ Technical Details

### Text Preprocessing
```python
- Lowercase conversion
- URL removal
- Email removal
- Special character removal (keeping punctuation)
- Short word removal (1-2 characters)
- Whitespace normalization
```

### Feature Engineering
- **Word TF-IDF**: 15,000 features, 1-3 grams
- **Character TF-IDF**: 5,000 features, 2-4 char grams
- **Total Features**: 20,000 (sparse matrix)

### Model Configuration
**Logistic Regression**:
- Solver: liblinear
- C: 5.0
- Multi-class: One-vs-Rest
- Class weight: Balanced
- Max iterations: 500

**SVM**:
- C: 5.0
- Loss: Squared hinge
- Max iterations: 2000
- Class weight: Balanced

## 📊 Project Structure

```
ml/
├── mental_health_detection.ipynb    # Main Jupyter notebook
├── Combined Data.csv                # Dataset (download separately)
└── README.md                        # This file
```

## 🔍 Key Insights

### Model Performance Analysis

This 7-class classification problem presents several inherent challenges:
- **Class Imbalance**: Distribution ranges from 2.3% to 30.8%
- **Overlapping Linguistic Patterns**: Mental health conditions share similar vocabulary and expressions
- **Limited Contextual Understanding**: Traditional ML approaches lack semantic comprehension

Achieving 80% accuracy represents strong performance for traditional machine learning methods on this complex multi-class problem.

### Technical Challenges
1. **Class Imbalance**: Minority classes represent less than 3% of the dataset
2. **Lexical Overlap**: Mental health conditions exhibit similar linguistic patterns
3. **Negation and Context**: Traditional bag-of-words approaches cannot distinguish contextual negation (e.g., "not suicidal" vs "suicidal")

### Future Research Directions
To achieve 85-92% accuracy:
- Implement transformer-based models (BERT, RoBERTa, GPT)
- Explore domain-specific pre-trained models (e.g., Mental-BERT)
- Develop ensemble architectures combining multiple deep learning approaches
- Apply data augmentation techniques to address class imbalance
- Investigate attention mechanisms for interpretability

## 📝 Notebook Sections

1. **Package Installation** - Install required libraries
2. **Data Loading** - Load and preview dataset
3. **Data Overview** - Class distribution and statistics
4. **Data Cleaning** - Remove duplicates, handle missing values
5. **Text Preprocessing** - Advanced regex-based cleaning
6. **Data Preparation** - Train/test split, label encoding
7. **Feature Extraction** - Dual TF-IDF vectorization
8. **Model Training** - Logistic Regression, SVM
9. **Model Comparison** - Performance metrics and charts
10. **Detailed Analysis** - Classification report, confusion matrix
11. **Summary** - Final results and conclusions

## Contributing

This project serves as an educational implementation. Potential enhancements include:
- Implementation of deep learning architectures (LSTM, GRU, Transformer)
- Advanced feature engineering techniques
- K-fold cross-validation for robust performance estimation
- Systematic hyperparameter optimization using grid search or Bayesian methods

## Disclaimer

- This project is intended for **educational and research purposes only**
- This model is **not suitable for clinical diagnosis** or medical decision-making
- Mental health assessment requires evaluation by qualified healthcare professionals
- Model performance may vary significantly on out-of-distribution data

## References

- **Dataset**: Sarkar, S. (2023). Mental Health Sentiment Analysis Dataset. Kaggle. https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Methodologies**: TF-IDF vectorization, Logistic Regression, Support Vector Machines

## License

This project is released for educational purposes. Please cite the original dataset source when using this work in research or publications.

## Results Summary

```
==================================================
MENTAL HEALTH DETECTION - FINAL RESULTS
==================================================

Dataset: 51,060 samples across 7 classes
Best Model: Logistic Regression
Accuracy: 80.06%
Improvement over Random Baseline: 5.6×
Performance: Strong results for traditional ML approaches

==================================================
```

---

**Project Type**: Machine Learning Classification  
**Date**: December 2025  
**Status**: Complete
