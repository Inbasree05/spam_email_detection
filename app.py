from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Ensure correct columns
df = df.iloc[:, [0, 1]]
df.columns = ['label', 'message']
df['label'] = df['label'].str.strip().str.lower()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Track accuracy dynamically
total_predictions = 0
correct_predictions = 0

@app.route('/')
def index():
    return render_template('index.html', accuracy=0)  # Start with 0% accuracy

@app.route('/predict', methods=['POST'])
def predict():
    global total_predictions, correct_predictions  # Allow modifying global variables

    email_text = request.form['email_text'].strip()
    
    if not email_text:
        return render_template('index.html', accuracy=(correct_predictions / total_predictions * 100 if total_predictions else 0), result="Please enter some text!")

    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)[0]
    confidence = model.predict_proba(email_tfidf)[0]  # Get confidence scores

    spam_confidence = round(confidence[1] * 100, 2)  # Convert to percentage

    # Determine if it's spam or not
    if prediction == 1:
        result = f"Spam (Confidence: {spam_confidence}%)"
    else:
        result = f"Not Spam (Confidence: {100 - spam_confidence}%)"

    # Update dynamic accuracy (assume user knows actual label)
    total_predictions += 1
    if (prediction == 1 and "spam" in email_text.lower()) or (prediction == 0 and "spam" not in email_text.lower()):
        correct_predictions += 1

    dynamic_accuracy = round((correct_predictions / total_predictions) * 100, 2) if total_predictions > 0 else 0

    return render_template('index.html', accuracy=dynamic_accuracy, result=result)

if __name__ == '__main__':
    app.run(debug=True)

#"Congratulations! You've won a lottery of $1,000,000. Claim now! Ref: 2227"