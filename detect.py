

import streamlit as st
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load model and tokenizer once
model_name = "tuner007/pegasus_paraphrase"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Load bad words once
df = pd.read_csv('cursing_lexicon.csv')
bad_words = df['word'].tolist()

def paraphrase_text(text):
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    paraphrased = model.generate(**tokens)
    paraphrased_text = tokenizer.decode(paraphrased[0], skip_special_tokens=True)
    return paraphrased_text

def profanity_detect(user_input):
    # Check for profanity and replace with asterisks
    contains_bad_word = False
    #cleaned_text = user_input
    for word in bad_words:
        if word in user_input:
            cleaned_text = user_input.replace(word, "*****")
            contains_bad_word = True

    if contains_bad_word:
        st.warning("Bad word detected! Cleaned message: " + cleaned_text + "\n Message will be sent after paraphrasing")
        paraphrased_text = paraphrase_text(user_input)
        return paraphrased_text
    else:
        return user_input
