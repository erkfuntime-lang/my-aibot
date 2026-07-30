import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

SYSTEM_PROMPT = "You are a the best at competitive programming, and use c++. Your goal is to teach, and help people solve competitive programming problems"

st.title("My Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages
    )
    reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
