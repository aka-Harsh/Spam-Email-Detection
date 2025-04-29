# 🛡️ Email Spam Detection with Machine Learning
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Status](https://img.shields.io/badge/status-active-green.svg)

A powerful email spam detection system that uses machine learning algorithms to classify emails as spam or legitimate (ham). This tool delivers high-accuracy classification through multiple algorithms, making it perfect for filtering unwanted emails, improving inbox management, and enhancing email security.

## ✨ Features
- **Machine Learning Classification**: Uses multiple algorithms (Naive Bayes, SVM, Random Forest) to analyze and classify emails
- **Text Processing**: Implements NLP techniques to process email content
- **Model Comparison**: Automatically compares multiple ML models and selects the best performer
- **Web Interface**: User-friendly interface built with Flask
- **Visualizations**: Includes performance metrics and visualizations

---

## 🛠️ Prerequisites
- Python 3.7 or higher
- Required packages (listed in requirements.txt)
- Dataset for training (provided or downloadable)

---

## 🚀 Deployment
Follow these steps to deploy the project on your local system:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection
```

### 2. Set Up Python Environment
```bash
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
# Install Python dependencies
pip install pandas numpy scikit-learn nltk flask matplotlib seaborn
```

### 3. Download NLTK Resources
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 4. Train the Model
```bash
python main.py
```

### 5. Start the Application
```bash
python app.py
```

### 6. Open the Application
Open your web browser and navigate to: http://127.0.0.1:5000/

---
## 🔭 Project Outlook

![Image](https://github.com/user-attachments/assets/0950018c-e51b-4ac1-8107-58440dd47738)
![Image](https://github.com/user-attachments/assets/6b02098a-368c-49dc-b9f7-430e705f8d5e)
![Image](https://github.com/user-attachments/assets/723f352e-a67c-4805-a3ae-ed842b2ffab7)

---

## 🙏 Acknowledgements
- UCI Machine Learning Repository - SMS Spam Collection Dataset
