import streamlit as st
import torch
import pickle
from transformers import BertTokenizerFast

# Load the model from the pickle file
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the tokenizer from Hugging Face
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# Set the model to evaluation mode
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
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
