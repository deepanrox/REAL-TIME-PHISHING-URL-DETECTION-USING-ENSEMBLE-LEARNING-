# Real-Time Phishing URL detection using ensemble learning

## 🎯 Project Overview

This project implements a machine learning-based system to detect phishing websites using various URL-based features. The system analyzes URLs and classifies them as either legitimate (good) or phishing (bad) websites.

## 🚀 Features

- **URL Feature Extraction**: Extracts various features from URLs including:
  - Domain length, path length, and total URL length
  - Number of digits and letters in domain, path, and URL
  - Top-level domain analysis
  - URL structure analysis

- **Multiple ML Algorithms**: Implements and compares various machine learning algorithms:
  - Random Forest Classifier
  - AdaBoost Classifier
  - Gradient Boosting Classifier
  - Extra Trees Classifier
  - Multi-layer Perceptron (MLP)
  - Linear Discriminant Analysis
  - Logistic Regression

- **Ensemble Learning**: Uses Voting Classifier to combine multiple models for better performance
- **Comprehensive Analysis**: Includes data visualization, correlation analysis, and model evaluation

## 📊 Performance Metrics

The best performing model achieves:
- **Accuracy**: 91.0%
- **F1 Score**: 0.91
- **Matthews Correlation Coefficient**: 0.83

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see `requirements.txt`)

### Setup

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Place your `Dataset.csv` file in the project root directory
   - The dataset should contain columns: `URL` and `Label` (where Label is 'good' or 'bad')


### Using the Prediction Function

```python
# Example usage
from phishing_detector import predict_url

# Test a suspicious URL
result = predict_url("http://login-paypal.secureverify.com")
print(result)  # Output: "Phishing (bad)"

# Test a legitimate URL
result = predict_url("https://www.google.com/")
print(result)  # Output: "Legit (good)"
```

## 📈 Key Findings

1. **Feature Importance**: URL length, domain characteristics, and digit/letter ratios are strong indicators of phishing attempts
2. **Model Performance**: Ensemble methods (Voting Classifier) perform better than individual models
3. **Data Distribution**: The dataset shows clear patterns between legitimate and phishing URLs

## 🔍 Data Analysis

The project includes comprehensive data analysis:
- Histogram plots showing feature distributions
- Correlation heatmaps
- Top-level domain analysis
- Feature importance analysis



## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



