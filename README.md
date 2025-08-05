# Phishing Websites Detection using Machine Learning

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

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phishing-websites-detection.git
cd phishing-websites-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Place your `Dataset.csv` file in the project root directory
   - The dataset should contain columns: `URL` and `Label` (where Label is 'good' or 'bad')

## 📁 Project Structure

```
phishing-websites-detection/
├── Phishing_Websites_EL_updated.ipynb  # Main Jupyter notebook
├── Dataset.csv                         # Dataset file (not included in repo)
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── .gitignore                         # Git ignore file
└── images/                            # Generated visualizations
    ├── heatmap.png
    ├── heatmapfeaturecorr.png
    └── RFLearningPlot.png
```

## 🎮 Usage

### Running the Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Phishing_Websites_EL_updated.ipynb`

3. Run all cells to:
   - Load and preprocess the data
   - Extract features from URLs
   - Train multiple machine learning models
   - Evaluate model performance
   - Generate visualizations

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset sources and contributors
- Machine learning community for algorithms and techniques
- Open-source libraries used in this project

## 📞 Contact

For questions or suggestions, please open an issue on GitHub or contact the project maintainer.

---

**Note**: This project is for educational and research purposes. Always verify URLs through multiple sources before making security decisions. 