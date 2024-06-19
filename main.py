import streamlit as st
import pandas as pd
from dataclasses import dataclass
import detect
import Semantic
st.title("Safe Chat Hub")

@dataclass
class Message:
    actor: str
    payload: str

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)
# Input text box
user_input: str = st.chat_input("Enter your message here")

if user_input:
    mod_input = detect.profanity_detect(user_input)
    st.session_state[MESSAGES].append(Message(actor=USER, payload=mod_input))
    st.chat_message(USER).write(mod_input)
    response: str = f"You wrote {mod_input}"
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)
    similarity_score = Semantic.calculate_semantic_similarity(user_input, mod_input)
    st.info("The similarity score between original message and sent message is: "+str(similarity_score))
