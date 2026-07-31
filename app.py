import streamlit as st
from groq import Groq
from supabase import create_client

# --- Clients ---
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

st.title("My Chatbot")

# --- Personas ---
PERSONAS = {
    "Friendly Helper": "You are a friendly, helpful assistant who explains things simply.",
    "Math Tutor": "You are a patient, encouraging math tutor for beginners. Explain step by step.",
    "D&D Game Master": "You are a creative Dungeons & Dragons game master narrating an adventure.",
    "Code Reviewer": "You are a senior software engineer giving direct, constructive code feedback.",
    "Custom": ""  # leave blank so the user can type their own below
}

# --- Supabase helpers ---
def save_chat(name, persona, messages):
    supabase.table("chats").upsert({
        "name": name,
        "persona": persona,
        "messages": messages
    }, on_conflict="name").execute()

def load_chat_names():
    result = supabase.table("chats").select("name").order("updated_at", desc=True).execute()
    return [row["name"] for row in result.data]

def load_chat(name):
    result = supabase.table("chats").select("*").eq("name", name).single().execute()
    return result.data

def delete_chat(name):
    supabase.table("chats").delete().eq("name", name).execute()

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    persona_choice = st.selectbox("Choose a persona", list(PERSONAS.keys()))

    if persona_choice == "Custom":
        system_prompt = st.text_area("Write your own system prompt", height=200)
    else:
        system_prompt = st.text_area(
            "System prompt (editable)",
            value=PERSONAS[persona_choice],
            height=200
        )

    st.divider()
    st.header("Saved Chats")

    saved_names = load_chat_names()
    selected_chat = st.selectbox("Load a saved chat", ["(none)"] + saved_names)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load") and selected_chat != "(none)":
            chat_data = load_chat(selected_chat)
            st.session_state.messages = chat_data["messages"]
            st.rerun()
    with col2:
        if st.button("Delete") and selected_chat != "(none)":
            delete_chat(selected_chat)
            st.rerun()

    new_chat_name = st.text_input("Save current chat as")
    if st.button("Save chat") and new_chat_name:
        save_chat(new_chat_name, persona_choice, st.session_state.messages)
        st.success(f"Saved as '{new_chat_name}'")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []

# --- Render existing conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Handle new input ---
user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages
    )
    reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
