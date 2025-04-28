# main.py - Script to train and save the spam detection model

import os
import pandas as pd
import matplotlib.pyplot as plt
from spam_detector import EmailSpamDetector

def create_example_dataset():
    """Create a simple example dataset if the real dataset is not available"""
    print("Creating a simple example dataset for demonstration...")
    example_data = {
        'text': [
            "Congratulations! You've won a $1000 gift card. To claim call now!",
            "Free entry to win $1000000. Text YES to 12345 now!",
            "Meeting scheduled for tomorrow at 2pm.",
            "Hi, how are you doing? Let's catch up soon.",
            "URGENT: Your account has been suspended. Call this number immediately!",
            "Your package will be delivered today between 2-4pm.",
            "Click here to claim your prize! Limited time offer!",
            "Remember to pick up milk on your way home.",
            "You have won a free iPhone! Click here to claim now!",
            "The project deadline has been extended to next Friday.",
            "WINNER!! As a valued network customer you have been selected to receive a $900 prize reward!",
            "Please call our customer service regarding your unpaid invoices",
            "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575",
            "I'll be there in 10 minutes",
            "Reminder: Your appointment is scheduled for tomorrow at 3:00 PM",
            "Congrats! 1 year special cinema pass for 2 is yours. call 09061209465 now!",
            "Hey, what's up? Want to grab lunch later?",
            "Dear customer, your invoice is ready to view at myaccount.example.com",
            "URGENT! Your Mobile number has been awarded $2000, call 8001816552",
            "Could you please send me the report by end of day?"
        ],
        'label': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # 1 for spam, 0 for ham
    }
    df = pd.DataFrame(example_data)
    df.to_csv('example_spam.csv', index=False)
    return 'example_spam.csv'

def main():
    # Initialize spam detector
    spam_detector = EmailSpamDetector()
    
    # Check if the SMS Spam Collection Dataset exists
    dataset_path = 'spam.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("You can download the SMS Spam Collection Dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
        print("Or use the UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
        
        # Create a simple example dataset for demonstration
        dataset_path = create_example_dataset()
        print(f"Example dataset created at {dataset_path}")
    else:
        # Process the actual SMS Spam Collection Dataset
        print(f"Found dataset at {dataset_path}, processing...")
        df = pd.read_csv(dataset_path, encoding='latin-1')
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df.to_csv('processed_spam.csv', index=False)
        dataset_path = 'processed_spam.csv'
        print(f"Original dataset processed and saved to {dataset_path}")
    
    # Load data
    df = spam_detector.load_data(dataset_path)
    
    if df is not None:
        # Prepare data
        X_train, X_test, y_train, y_test = spam_detector.prepare_data(df)
        
        # Extract features
        X_train_features, X_test_features = spam_detector.extract_features(X_train, X_test, method='tfidf')
        
        # Train model
        print("\nTraining models...")
        print("\n1. Naive Bayes Model")
        spam_detector.train_model(X_train_features, y_train, model_type='naive_bayes')
        accuracy_nb, report_nb, _ = spam_detector.evaluate_model(X_test_features, y_test)
        spam_detector.save_model('nb_model.pkl', 'tfidf_vectorizer.pkl')
        
        print("\n2. SVM Model")
        spam_detector.train_model(X_train_features, y_train, model_type='svm')
        accuracy_svm, report_svm, _ = spam_detector.evaluate_model(X_test_features, y_test)
        spam_detector.save_model('svm_model.pkl', 'tfidf_vectorizer.pkl')
        
        print("\n3. Random Forest Model")
        spam_detector.train_model(X_train_features, y_train, model_type='random_forest')
        accuracy_rf, report_rf, _ = spam_detector.evaluate_model(X_test_features, y_test)
        spam_detector.save_model('rf_model.pkl', 'tfidf_vectorizer.pkl')
        
        # Compare model performances
        models = ['Naive Bayes', 'SVM', 'Random Forest']
        accuracies = [accuracy_nb, accuracy_svm, accuracy_rf]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.ylim(0.8, 1.0)  # Adjust as needed
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        # Choose the best model
        best_model_index = accuracies.index(max(accuracies))
        best_model = models[best_model_index]
        
        print(f"\nBest model: {best_model} with accuracy: {max(accuracies):.4f}")
        print(f"Model comparison chart saved to model_comparison.png")
        
        # Save the best model as the default
        best_model_file = ['nb_model.pkl', 'svm_model.pkl', 'rf_model.pkl'][best_model_index]
        if os.path.exists(best_model_file):
            import shutil
            shutil.copy(best_model_file, 'spam_detector_model.pkl')
            print(f"Best model ({best_model}) saved as spam_detector_model.pkl")
        
        # Test with sample emails
        print("\nTesting with sample emails:")
        
        # Load the best model
        spam_detector.load_model('spam_detector_model.pkl', 'tfidf_vectorizer.pkl')
        
        sample_emails = [
            "Congratulations! You've won a free vacation. Call now to claim your prize!",
            "Hi, just checking in to see how you're doing. Let's meet for coffee next week.",
            "URGENT: Your bank account has been compromised. Click here to verify your details immediately!",
            "The meeting has been rescheduled to 3 PM tomorrow. Please bring your presentation.",
            "FREE ENTRY! WIN $5000 CASH! Call 1-800-555-1234 now!"
        ]
        
        for i, email in enumerate(sample_emails):
            result = spam_detector.predict_email(email)
            print(f"\nEmail {i+1}: {email[:50]}...")
            print(f"Prediction: {result['prediction'].upper()}")
            if result['confidence'] is not None:
                print(f"Confidence: {result['confidence']:.4f}")
            print("-" * 50)
        
        print("\nModel training and evaluation completed successfully!")
        print("You can now run the Flask application (app.py) to use the web interface.")

if __name__ == "__main__":
    main()