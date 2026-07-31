import streamlit as st
import re
from groq import Groq
from supabase import create_client

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
db = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

st.set_page_config(layout="wide")

ARTIFACT_INSTRUCTIONS = """
When the user asks you to build something visual or interactive (a webpage, a game, a diagram, a small app, an SVG graphic), respond with the code wrapped EXACTLY like this:

<artifact type="html" title="Short title here">
...complete, self-contained HTML/CSS/JS code here...
</artifact>

Rules:
- Put ALL CSS and JS inside this single HTML block (no external files).
- Only use this tag when producing something meant to be viewed/run, not for regular code snippets or explanations.
- You can still write normal text before or after the artifact block to explain it.
"""

PERSONAS = {
    "Friendly Helper": "You are a friendly, helpful assistant who explains things simply.",
    "Math Tutor": "You are a patient, encouraging math tutor for beginners. Explain step by step.",
    "D&D Game Master": "You are a creative Dungeons & Dragons game master narrating an adventure.",
    "Code Reviewer": "You are a senior software engineer giving direct, constructive code feedback.",
    "Custom": ""
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_artifact" not in st.session_state:
    st.session_state.current_artifact = None

def extract_artifact(text):
    """Pulls out the artifact block if present. Returns (clean_text, artifact_dict_or_None)."""
    match = re.search(r'<artifact type="(.*?)" title="(.*?)">(.*?)</artifact>', text, re.DOTALL)
    if not match:
        return text, None
    art_type, title, code = match.groups()
    clean_text = text[:match.start()] + text[match.end():]
    return clean_text.strip(), {"type": art_type, "title": title, "code": code.strip()}

with st.sidebar:
    st.header("Settings")
    persona_choice = st.selectbox("Choose a persona", list(PERSONAS.keys()))
    if persona_choice == "Custom":
        system_prompt = st.text_area("Write your own system prompt", height=150)
    else:
        system_prompt = st.text_area("System prompt (editable)", value=PERSONAS[persona_choice], height=150)

    st.divider()
    st.subheader("Save current chat")
    save_title = st.text_input("Chat title")
    if st.button("Save chat") and save_title and st.session_state.messages:
        db.table("chats").insert({
            "title": save_title,
            "persona": system_prompt,
            "messages": st.session_state.messages
        }).execute()
        st.success("Saved!")

    st.divider()
    st.subheader("Load a saved chat")
    saved = db.table("chats").select("id, title, created_at").order("created_at", desc=True).execute()
    saved_options = {f"{row['title']} ({row['created_at'][:10]})": row["id"] for row in saved.data}
    if saved_options:
        chosen = st.selectbox("Pick a chat", list(saved_options.keys()))
        if st.button("Load chat"):
            chat_id = saved_options[chosen]
            full = db.table("chats").select("*").eq("id", chat_id).single().execute()
            st.session_state.messages = full.data["messages"]
            st.session_state.current_artifact = None
            st.rerun()

    if st.button("Clear current chat"):
        st.session_state.messages = []
        st.session_state.current_artifact = None

# --- Two-column layout: chat on left, artifact panel on right ---
chat_col, artifact_col = st.columns([1, 1])

with chat_col:
    st.subheader("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Say something...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        full_system_prompt = system_prompt + "\n\n" + ARTIFACT_INSTRUCTIONS
        api_messages = [{"role": "system", "content": full_system_prompt}] + st.session_state.messages

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages
        )
        raw_reply = response.choices[0].message.content

        clean_reply, artifact = extract_artifact(raw_reply)

        st.session_state.messages.append({"role": "assistant", "content": clean_reply})
        with st.chat_message("assistant"):
            st.write(clean_reply)

        if artifact:
            st.session_state.current_artifact = artifact
            st.rerun()

with artifact_col:
    st.subheader("Artifact Preview")
    art = st.session_state.current_artifact
    if art:
        st.caption(art["title"])
        tab1, tab2 = st.tabs(["Preview", "Code"])
        with tab1:
            st.components.v1.html(art["code"], height=500, scrolling=True)
        with tab2:
            st.code(art["code"], language="html")
    else:
        st.info("Ask the assistant to build something visual, and it'll show up here.")
