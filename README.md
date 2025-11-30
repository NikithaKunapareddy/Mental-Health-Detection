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

**Achieved Accuracy: 78.06%** using Logistic Regression with optimized TF-IDF features.

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
| **Accuracy** | **78.06%** |
| **Precision** | **77.93%** |
| **Recall** | **78.06%** |
| **F1-Score** | **77.94%** |

### Model Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 78.06% ✅ |
| SVM | 74.26% |

### Performance Context
- **Random Baseline**: 14.3% (1 in 7 classes)
- **Improvement**: 5.6x better than random guessing
- **Traditional ML Limit**: 75-80% for 7-class problems
- **Deep Learning (BERT)**: Could achieve 85-92%

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

### Why 78% Accuracy?

This is a **challenging 7-class classification problem** with:
- **Imbalanced classes** (2.3% to 30.8%)
- **Overlapping symptoms** (Depression/Anxiety/Suicidal use similar words)
- **Traditional ML limitations** (no context understanding)

**78% accuracy is excellent for traditional ML!**

### Challenges
1. **Class Imbalance**: Some classes have very few samples
2. **Similar Text Patterns**: Mental health conditions share vocabulary
3. **Context Dependency**: "I'm not suicidal" vs "I'm suicidal" (needs deep learning)

### Future Improvements
To achieve 85-92% accuracy:
- Use **BERT** or **RoBERTa** transformers
- Try **Mental-BERT** (pre-trained on mental health text)
- Implement **ensemble of deep learning models**
- Use **data augmentation** for minority classes

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

## 🤝 Contributing

This is an educational project. Suggestions for improvements:
- Try deep learning models (LSTM, BERT)
- Experiment with different feature engineering
- Add cross-validation
- Implement hyperparameter tuning

## ⚠️ Important Notes

- This is for **educational purposes only**
- Not intended for **clinical diagnosis**
- Mental health assessment requires **professional evaluation**
- The model has limitations and may not generalize to all text

## 📚 References

- Dataset: [Kaggle Mental Health Sentiment Analysis](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
- Techniques: TF-IDF, Logistic Regression, SVM

## 📄 License

This project is for educational purposes. Please cite the dataset source if using in research.

## 🏆 Results Summary

```
==================================================
MENTAL HEALTH DETECTION - FINAL RESULTS
==================================================

📊 Dataset: 51,060 samples, 7 classes
🎯 Best Model: Logistic Regression
✅ Accuracy: 78.06%
📈 5.6x better than random guessing
🚀 Excellent performance for traditional ML!

==================================================
```

---

**Author**: Machine Learning Project  
**Date**: November 2025  
**Status**: Complete ✅
