import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model_path = './results/final_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Define label meanings
label_meanings = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Streamlit app code
st.title("Sentiment Analysis App")

text = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if text:
        result = sentiment_pipeline(text)
        # Add label meaning and color to the result
        label_colors = {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'orange',
            4: 'purple',
            5: 'pink'
        }
        for res in result:
            label_index = int(res['label'].split('_')[-1])
            res['label_meaning'] = label_meanings[label_index]
            res['color'] = label_colors[label_index]
        
        # Display the result with color
        for res in result:
            st.markdown(f"<div style='color:{res['color']}'>Label: {res['label']}<br>Meaning: {res['label_meaning']}<br>Score: {res['score']:.2f}</div>", unsafe_allow_html=True)
    else:
        st.write("Please enter some text for analysis.")