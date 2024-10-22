import streamlit as st
import torch
import re
import gdown  # Import gdown for Google Drive downloads
from transformers import BertTokenizerFast, BertForSequenceClassification
from nltk.corpus import stopwords
import nltk
import os

nltk.download('stopwords')

file_id = '18IHEKb43uakOFB1VMH6XLnkldBAZJgfi'
model_path = 'sentiment_model.pkl'

def download_model_from_drive():
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

download_model_from_drive()

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
device = torch.device("cpu")
model = None

try:
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join(text.split())
    
    return text

# Predict sentiment if model is loaded successfully
if model is not None:
    def predict_sentiment(text):
        preprocessed_text = preprocess_text(text)
        inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).item()
        
        return predictions

    # Streamlit UI
    st.title("Sentiment Analysis with BERT")
    user_input = st.text_area("Enter text for sentiment analysis:")

    if st.button("Analyze"):
        if user_input:
            sentiment = predict_sentiment(user_input)
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Please enter some text.")
else:
    st.write("The model could not be loaded. Please check the model file.")
