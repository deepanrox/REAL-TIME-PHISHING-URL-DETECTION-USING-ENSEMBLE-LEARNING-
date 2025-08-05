"""
Phishing Websites Detection Module

This module provides functionality to detect phishing websites using machine learning.
It includes feature extraction from URLs and a trained model for classification.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import tldextract
import warnings
import re
warnings.filterwarnings('ignore')

class PhishingDetector:
    """
    A class to detect phishing websites using machine learning.
    """
    
    def __init__(self):
        """Initialize the PhishingDetector with trained models."""
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self.is_trained = False
        
    def count_digits(self, string):
        """Count the number of digits in a string."""
        return sum(c.isdigit() for c in string)
    
    def count_letter(self, string):
        """Count the number of letters in a string."""
        return sum(c.isalpha() for c in string)
    
    def extract_features(self, url):
        """
        Extract features from a URL for phishing detection.
        
        Args:
            url (str): The URL to analyze
            
        Returns:
            dict: Dictionary containing extracted features
        """
        # Input validation
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string")
        
        # Clean URL
        url = url.strip()
        
        # Split URL into domain and path
        url_split = url.split("/", 1)
        domain, path = url_split[0], url_split[1] if len(url_split) == 2 else ""
        
        features = {
            "total_digits_domain": self.count_digits(domain),
            "total_digits_path": self.count_digits(path),
            "total_digits_url": self.count_digits(url),
            "total_letter_domain": self.count_letter(domain),
            "total_letter_path": self.count_letter(path),
            "total_letter_url": self.count_letter(url),
            "len_domain": len(domain),
            "len_path": len(path),
            "len_url": len(url)
        }
        
        return features
    
    def train(self, data_path):
        """
        Train the phishing detection model.
        
        Args:
            data_path (str): Path to the CSV file containing training data
        """
        # Load data
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")
        
        # Validate data structure
        required_columns = ['URL', 'Label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Extract features
        data['features'] = data['URL'].apply(self.extract_features)
        X = pd.DataFrame(list(data['features']))
        
        # Encode labels
        y = self.label_encoder.fit_transform(data["Label"])
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define and train the VotingClassifier
        ada = AdaBoostClassifier(n_estimators=100, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        self.classifier = VotingClassifier(
            estimators=[('ada', ada), ('rf', rf), ('gb', gb)], 
            voting='soft'
        )
        
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training results
        predictions = self.classifier.predict(X_test)
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
        
        print(f"Training completed!")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print(f"F1 Score: {f1_score(y_test, predictions, average='weighted'):.3f}")
        print(f"Matthews Correlation: {matthews_corrcoef(y_test, predictions):.3f}")
    
    def predict(self, url):
        """
        Predict whether a URL is phishing or legitimate.
        
        Args:
            url (str): The URL to classify
            
        Returns:
            str: "Legit (good)" or "Phishing (bad)"
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
        
        features = self.extract_features(url)
        features_df = pd.DataFrame([features])
        result = self.classifier.predict(features_df)[0]
        label = self.label_encoder.inverse_transform([result])[0]
        
        return "Legit (good)" if label == "good" else "Phishing (bad)"
    
    def predict_batch(self, urls):
        """
        Predict multiple URLs at once.
        
        Args:
            urls (list): List of URLs to classify
            
        Returns:
            list: List of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
        
        if not isinstance(urls, list):
            raise ValueError("URLs must be provided as a list")
        
        features_list = [self.extract_features(url) for url in urls]
        features_df = pd.DataFrame(features_list)
        results = self.classifier.predict(features_df)
        labels = self.label_encoder.inverse_transform(results)
        
        return ["Legit (good)" if label == "good" else "Phishing (bad)" for label in labels]

def predict_url(url, model_path=None):
    """
    Convenience function to predict a single URL.
    
    Args:
        url (str): The URL to classify
        model_path (str, optional): Path to a saved model file
        
    Returns:
        str: "Legit (good)" or "Phishing (bad)"
    """
    detector = PhishingDetector()
    
    if model_path:
        # Load pre-trained model (if available)
        import pickle
        try:
            with open(model_path, 'rb') as f:
                detector = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        # Train a new model (requires Dataset.csv in current directory)
        try:
            detector.train('Dataset.csv')
        except FileNotFoundError:
            raise FileNotFoundError("Dataset.csv not found. Please ensure the dataset is in the current directory.")
    
    return detector.predict(url)

# Example usage
if __name__ == "__main__":
    # Example URLs for testing
    test_urls = [
        "https://www.google.com/",
        "http://login-paypal.secureverify.com",
        "https://www.github.com/",
        "http://fake-bank-login.xyz"
    ]
    
    print("Testing phishing detection...")
    print("=" * 50)
    
    for url in test_urls:
        try:
            result = predict_url(url)
            print(f"URL: {url}")
            print(f"Result: {result}")
            print("-" * 30)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            print("-" * 30) 