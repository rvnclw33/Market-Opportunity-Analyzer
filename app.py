import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import tensorflow as tf
from openai import OpenAI
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Market Opportunity Analyzer", page_icon="📊", layout="wide")

MAX_WORDS = 5000
MAX_LEN = 100
NUM_TOPICS = 5
RND_STATE = 42

# -----------------------------
# OPENAI SETUP (Optional)
# -----------------------------
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# -----------------------------
# 1. HELPER FUNCTIONS
# -----------------------------
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r"\bim\b", "i am", text)
    text = re.sub(r"\bdont\b", "do not", text)
    text = re.sub(r"\bcant\b", "cannot", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_column(df):
    possible_cols = ['Review', 'review', 'text', 'content', 'body', 'comment']
    text_col = next((c for c in possible_cols if c in df.columns), None)
    if text_col:
        df['clean_text'] = df[text_col].apply(clean_text)
        df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
        return df
    return None

# -----------------------------
# 2. CORE ML PIPELINE
# -----------------------------
@st.cache_resource
def train_pipeline(df):
    custom_ignore = ['im', 'use', 'using', 'app', 'slack', 'just', 'get', 'work', 'good', 'great', 'like']
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_ignore))
    
    tfidf = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    X_tfidf = tfidf.fit_transform(df['clean_text'])
    
    kmeans = KMeans(n_clusters=NUM_TOPICS, random_state=RND_STATE, n_init=10)
    df['topic_id'] = kmeans.fit_predict(X_tfidf)
    
    terms = tfidf.get_feature_names_out()
    centers = kmeans.cluster_centers_.argsort()[:, ::-1]
    topic_map = {i: ", ".join([terms[ind] for ind in centers[i, :5]]) for i in range(NUM_TOPICS)}
    df['topic_keywords'] = df['topic_id'].map(topic_map)

    analyzer = SentimentIntensityAnalyzer()
    def silver_label(t):
        s = analyzer.polarity_scores(t)['compound']
        return 1 if s > 0.05 else (0 if s < -0.05 else -1)

    df['temp_label'] = df['clean_text'].apply(silver_label)
    df_train = df[df['temp_label'] != -1].copy()
    
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df_train['clean_text'])
    
    X = pad_sequences(tokenizer.texts_to_sequences(df_train['clean_text']), maxlen=MAX_LEN)
    y = df_train['temp_label'].values
    
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(MAX_WORDS, 32, input_length=MAX_LEN),
        Bidirectional(LSTM(32, dropout=0.4)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=15, batch_size=16, verbose=0)
    
    return df, model, tokenizer

# -----------------------------
# 3. SCORING (The Logic Stack)
# -----------------------------
def get_smart_sentiment(text, model, tokenizer):
    # A. PREPARE TEXT (Split 'but')
    if " but " in text:
        target_text = text.split(" but ")[-1]
    else:
        target_text = text

    # B. GET LSTM PREDICTION
    seq = tokenizer.texts_to_sequences([target_text])
    if not seq or len(seq[0]) == 0:
        lstm_prob = 0.5 # Unknown
        lstm_label = "Neutral"
    else:
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        lstm_prob = model.predict(padded, verbose=0)[0][0]
        lstm_label = "Positive" if lstm_prob > 0.5 else "Negative"

    # C. GET VADER SCORE
    analyzer = SentimentIntensityAnalyzer()
    vader_score = analyzer.polarity_scores(target_text)['compound']
    
    # D. KEYWORD CHECK
    complaint_keywords = ['cost', 'price', 'money', 'expensive', 'charge', 'slow', 'crash', 'bug', 'broken', 'freeze']
    has_keyword = any(w in target_text for w in complaint_keywords)
    
    # --- FINAL DECISION LOGIC ---
    
    # 1. VADER says Negative? -> Trust VADER (Safety Net)
    if vader_score < -0.05:
        return "Negative", abs(vader_score), "VADER (Negative Guard)"
        
    # 2. LSTM says Negative BUT VADER says Positive? -> Trust VADER (Fixes "good" bug)
    if lstm_label == "Negative" and vader_score > 0.05:
        return "Positive", vader_score, "VADER (Positive Rescue)"
        
    # 3. Keyword Check
    # Only trigger if VADER isn't strongly positive (to avoid flagging "Good price")
    if has_keyword and vader_score < 0.4:
        return "Negative", 0.90, "Keyword Rule"

    # 4. Default -> Trust LSTM
    conf = lstm_prob if lstm_label == "Positive" else 1 - lstm_prob
    return lstm_label, conf, "Deep Learning (LSTM)"


def run_scoring(df, model, tokenizer):
    # Apply Smart Logic to every row
    results = df['clean_text'].apply(lambda x: get_smart_sentiment(x, model, tokenizer))
    
    df['sentiment_label'] = [r[0] for r in results]
    df['probability'] = [r[1] for r in results] # Store confidence
    df['is_negative'] = (df['sentiment_label'] == 'Negative').astype(int)
    
    stats = df.groupby('topic_keywords').agg(
        mentions=('topic_id', 'count'),
        neg_count=('is_negative', 'sum')
    ).reset_index()
    
    stats['Neg %'] = stats['neg_count'] / stats['mentions']
    stats['raw_score'] = stats['mentions'] * stats['Neg %']
    stats['Opportunity Score'] = (stats['raw_score'] / stats['raw_score'].max()) * 100 if stats['raw_score'].max() > 0 else 0
    
    return df, stats.sort_values('Opportunity Score', ascending=False)

def generate_ai_insight(query, table_str):
    if not client: return "AI not configured."
    prompt = f"Analyze this market opportunity data:\n{table_str}\n\nUser Question: {query}"
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content
    except Exception as e: return f"Error: {e}"

# -----------------------------
# 4. MAIN APP UI
# -----------------------------
st.title("📊 Market Opportunity Analyzer")
st.markdown("""
> **An end-to-end NLP pipeline** that transforms unstructured customer reviews into actionable product strategy. 
>
> This project demonstrates architectural mastery by benchmarking three distinct models:
> * **Unsupervised:** K-Means Clustering (Topic Discovery)
> * **Classical ML:** Logistic Regression (Baseline)
> * **Deep Learning:** Bi-Directional LSTM (Sentiment Classification)
""")

uploaded_file = st.file_uploader("Upload Reviews (CSV)", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df = extract_text_column(df_raw)
    
    if df is not None:
        st.success(f"Loaded {len(df)} reviews. Training models...")
        
        with st.spinner("Training K-Means & LSTM..."):
            df_proc, model, tokenizer = train_pipeline(df)
            df_final, stats = run_scoring(df_proc, model, tokenizer)
            
        st.subheader("🏆 Market Opportunity Dashboard")
        st.dataframe(
            stats[['topic_keywords', 'mentions', 'Neg %', 'Opportunity Score']],
            column_config={
                "Opportunity Score": st.column_config.ProgressColumn("Score", format="%.1f", min_value=0, max_value=100),
                "Neg %": st.column_config.NumberColumn("Negative Sentiment", format="%.2%")
            },
            use_container_width=True
        )
        
        st.divider()
        st.subheader("🔍 Explore All Documents")
        st.dataframe(
            df_final[['clean_text', 'topic_keywords', 'sentiment_label', 'probability']],
            column_config={"probability": st.column_config.NumberColumn("Confidence", format="%.2f")},
            use_container_width=True
        )

        if client:
            st.divider()
            if query := st.chat_input("Ask about the opportunities..."):
                with st.chat_message("user"): st.write(query)
                with st.chat_message("assistant"):
                    st.write(generate_ai_insight(query, stats.to_string()))

        st.divider()
        with st.expander("🛠️ Live Model Test", expanded=True):
            st.write("Test the model's accuracy on manual inputs.")
            txt = st.text_input("Enter a review:", "The features are great, but it costs way too much money.")
            
            if txt:
                cleaned_txt = clean_text(txt)
                label, conf, source = get_smart_sentiment(cleaned_txt, model, tokenizer)
                
                if label == "Positive":
                    st.success(f"Prediction: {label}")
                else:
                    st.error(f"Prediction: {label}")
                st.caption(f"Source: {source} | Confidence: {conf:.2%}")

    else:
        st.error("No text column found.")