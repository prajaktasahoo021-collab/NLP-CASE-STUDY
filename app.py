import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main {
    padding: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load dataset
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/SMSSpamCollection",
        sep="\t",
        names=["label", "message"]
    )

    return df


df = load_data()


# -----------------------------
# Text preprocessing
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    return text


df["clean_message"] = df["message"].apply(clean_text)


# -----------------------------
# Train model
# -----------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    df["clean_message"]
)

y = df["label"]


model = MultinomialNB()

model.fit(X, y)


# -----------------------------
# Website header
# -----------------------------

st.markdown(
    '<div class="title">📱 SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered SMS classification using NLP'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# -----------------------------
# Message input
# -----------------------------

message = st.text_area(
    "Enter your SMS message:",
    height=150,
    placeholder="Example: Congratulations! You have won a prize..."
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Analyze Message", use_container_width=True):

    if not message.strip():

        st.warning("Please enter an SMS message.")

    else:

        cleaned = clean_text(message)

        transformed = vectorizer.transform(
            [cleaned]
        )

        prediction = model.predict(
            transformed
        )[0]

        probability = model.predict_proba(
            transformed
        ).max()


        if prediction == "spam":

            st.error("🚨 SPAM MESSAGE")

            st.write(
                f"Confidence: {probability * 100:.2f}%"
            )

        else:

            st.success("✅ HAM / SAFE MESSAGE")

            st.write(
                f"Confidence: {probability * 100:.2f}%"
            )


# -----------------------------
# Project information
# -----------------------------

st.divider()

st.subheader("📊 Project Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Messages",
        len(df)
    )

with col2:

    st.metric(
        "Spam Messages",
        (df["label"] == "spam").sum()
    )

with col3:

    st.metric(
        "Ham Messages",
        (df["label"] == "ham").sum()
    )


st.divider()

st.caption(
    "SMS Spam Detection using NLP • "
    "TF-IDF • Multinomial Naive Bayes"
)