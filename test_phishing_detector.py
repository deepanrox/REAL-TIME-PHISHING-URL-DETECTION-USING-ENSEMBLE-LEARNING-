"""
Test file for the phishing detector module.
"""

import unittest
import pandas as pd
import numpy as np
from phishing_detector import PhishingDetector

class TestPhishingDetector(unittest.TestCase):
    """Test cases for PhishingDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = PhishingDetector()
        
    def test_count_digits(self):
        """Test digit counting functionality."""
        self.assertEqual(self.detector.count_digits("abc123def456"), 6)
        self.assertEqual(self.detector.count_digits("no_digits"), 0)
        self.assertEqual(self.detector.count_digits("12345"), 5)
        
    def test_count_letter(self):
        """Test letter counting functionality."""
        self.assertEqual(self.detector.count_letter("abc123def456"), 6)
        self.assertEqual(self.detector.count_letter("12345"), 0)
        self.assertEqual(self.detector.count_letter("abcdef"), 6)
        
    def test_extract_features(self):
        """Test feature extraction from URLs."""
        url = "https://www.example.com/path123"
        features = self.detector.extract_features(url)
        
        expected_keys = [
            "total_digits_domain", "total_digits_path", "total_digits_url",
            "total_letter_domain", "total_letter_path", "total_letter_url",
            "len_domain", "len_path", "len_url"
        ]
        
        for key in expected_keys:
            self.assertIn(key, features)
            self.assertIsInstance(features[key], int)
            
    def test_extract_features_no_path(self):
        """Test feature extraction from URLs without path."""
        url = "https://www.example.com"
        features = self.detector.extract_features(url)
        
        self.assertEqual(features["len_path"], 0)
        self.assertEqual(features["total_digits_path"], 0)
        self.assertEqual(features["total_letter_path"], 0)
        
    def test_model_not_trained_error(self):
        """Test that predict raises error when model is not trained."""
        with self.assertRaises(ValueError):
            self.detector.predict("https://www.example.com")

if __name__ == "__main__":
    unittest.main() 