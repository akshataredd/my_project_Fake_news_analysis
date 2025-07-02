import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Sidebar
with st.sidebar:

    st.title("🧠 AI News Checker")
    st.markdown("Check if news is **Real or Fake** using Machine Learning.\n\nBuilt with ❤️ using Python, NLTK, and Streamlit.")

# Main title
st.markdown("<h2 style='text-align: center; color: #4B8BBE;'>📰 Fake News Detection App</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a news article or sentence to verify its authenticity.</p>", unsafe_allow_html=True)

# Input box
news_input = st.text_area("✏️ Enter News Text Below:", height=200)

# Button
if st.button("🔍 Check News Authenticity"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        st.markdown("---")
        if prediction[0] == 1:
            st.success("✅ This looks like **Real News**. Stay informed! 🧠")
        else:
            st.error("🚨 This seems like **Fake News**. Be cautious! ❌")
        st.markdown("---")
