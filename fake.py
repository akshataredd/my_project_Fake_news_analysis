import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake['label'] = 0
real['label'] = 1

# Combine and shuffle
data = pd.concat([fake, real], axis=0)
data = data.sample(frac=1).reset_index(drop=True)
data = data.drop(['subject', 'date'], axis=1)

# ✅ Download NLTK resources (run once)
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# ✅ Define cleaning function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ✅ Apply the cleaning function
data['clean_text'] = data['text'].apply(clean_text)
print("✅ Text cleaned")
print(data[['text', 'clean_text']].head())

# ✅ Preview the cleaned data
print("\n✅ Sample cleaned text:")
print(data[['text', 'clean_text']].head())
# Create the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the clean text
X = vectorizer.fit_transform(data['clean_text'])

# Get the labels (0 for fake, 1 for real)
y = data['label']

print("\n✅ TF-IDF Vectorization completed.")
print("Shape of X:", X.shape)  # (rows, features)
 #Step 6: Split the data into training and testing
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# Step 7: Train the Logistic Regression model
# -----------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)
# Save the trained model
joblib.dump(model, "fake_news_model.pkl")

# Save the vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")

# -----------------------------------------------
# Step 8: Make predictions and evaluate
# -----------------------------------------------
y_pred = model.predict(X_test)

# Evaluate the model
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#streamlit run app.py

