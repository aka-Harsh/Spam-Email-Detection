# spam_detector.py - Module containing EmailSpamDetector class

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Make sure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

class EmailSpamDetector:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.vectorizer = None
    
    def load_data(self, file_path):
        """
        Load the dataset. Expected format is CSV with 'text' and 'label' columns.
        If you have the SMS Spam Collection Dataset, you can use:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_text(self, text):
        """Preprocess the text data"""
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize text
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and stem words
        cleaned_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Join tokens back to string
        cleaned_text = ' '.join(cleaned_tokens)
        
        return cleaned_text
    
    def prepare_data(self, df):
        """Prepare the data for training"""
        # Apply preprocessing to text data
        print("Preprocessing text...")
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        # Split the data into training and testing sets
        X = df['cleaned_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def extract_features(self, X_train, X_test, method='tfidf'):
        """Extract features from text using specified vectorizer"""
        if method == 'count':
            self.vectorizer = CountVectorizer(max_features=5000)
        else:  # tfidf
            self.vectorizer = TfidfVectorizer(max_features=5000)
        
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        
        print(f"Feature extraction completed using {method} vectorizer.")
        print(f"Training features shape: {X_train_features.shape}")
        print(f"Testing features shape: {X_test_features.shape}")
        
        return X_train_features, X_test_features
    
    def train_model(self, X_train_features, y_train, model_type='naive_bayes'):
        """Train the specified model"""
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_features, y_train)
        print(f"Model training completed using {model_type}.")
        
        return self.model
    
    def evaluate_model(self, X_test_features, y_test):
        """Evaluate the trained model"""
        y_pred = self.model.predict(X_test_features)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        return accuracy, report, conf_matrix
    
    def save_model(self, model_path='spam_detector_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Save the trained model and vectorizer"""
        if self.model and self.vectorizer:
            with open(model_path, 'wb') as model_file:
                pickle.dump(self.model, model_file)
            
            with open(vectorizer_path, 'wb') as vec_file:
                pickle.dump(self.vectorizer, vec_file)
            
            print(f"Model saved to {model_path}")
            print(f"Vectorizer saved to {vectorizer_path}")
        else:
            print("Error: Model or vectorizer not initialized. Train the model first.")
    
    def load_model(self, model_path='spam_detector_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load a trained model and vectorizer"""
        try:
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
            
            with open(vectorizer_path, 'rb') as vec_file:
                self.vectorizer = pickle.load(vec_file)
            
            print("Model and vectorizer loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_email(self, email_text):
        """Predict if an email is spam or ham"""
        if not self.model or not self.vectorizer:
            print("Error: Model not loaded. Please train or load a model first.")
            return None
        
        # Preprocess the email text
        preprocessed_text = self.preprocess_text(email_text)
        
        # Extract features
        email_features = self.vectorizer.transform([preprocessed_text])
        
        # Make prediction
        prediction = self.model.predict(email_features)[0]
        probability = None
        
        # Get prediction probability if the model supports it
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(email_features)[0]
        
        result = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': probability[1] if probability is not None else None
        }
        
        return result