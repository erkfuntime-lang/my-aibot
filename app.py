import streamlit as st
from groq import Groq
import requests
import re

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.title("My Chatbot")

PERSONAS = {
    "Friendly Helper": "You are a friendly, helpful assistant who explains things simply.",
    "Math Tutor": "You are a patient, encouraging math tutor for beginners. Explain step by step.",
    "D&D Game Master": "You are a creative Dungeons & Dragons game master narrating an adventure.",
    "Code Reviewer": "You are a senior software engineer giving direct, constructive code feedback.",
    "Custom": ""
}

with st.sidebar:
    st.header("Settings")
    persona_choice = st.selectbox("Choose a persona", list(PERSONAS.keys()))
    if persona_choice == "Custom":
        system_prompt = st.text_area("Write your own system prompt", height=200)
    else:
        system_prompt = st.text_area("System prompt (editable)", value=PERSONAS[persona_choice], height=200)
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Maps Piston's language names to what markdown code fences usually say
LANGUAGE_MAP = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "java": "java",
    "c": "c", "cpp": "cpp", "c++": "cpp",
    "bash": "bash", "sh": "bash",
    "ruby": "ruby",
    "go": "go",
}

@st.cache_data(ttl=3600)
def get_piston_runtimes():
    try:
        resp = requests.get("https://emkc.org/api/v2/piston/runtimes", timeout=10)
        return resp.json()
    except Exception:
        return []

def run_code_with_piston(language, code):
    runtimes = get_piston_runtimes()
    version = None
    for rt in runtimes:
        if rt["language"] == language or language in rt.get("aliases", []):
            version = rt["version"]
            break

    if not version:
        return f"'{language}' isn't available on the code runner right now."

    try:
        resp = requests.post(
            "https://emkc.org/api/v2/piston/execute",
            json={
                "language": language,
                "version": version,
                "files": [{"content": code}]
            },
            timeout=15
        )
        result = resp.json()

        if "run" not in result:
            # Piston returned an error instead of run results — show it plainly
            return f"Error from code runner: {result}"

        run = result["run"]
        stdout = run.get("stdout", "")
        stderr = run.get("stderr", "")
        combined = stdout
        if stderr:
            combined += f"\n--- stderr ---\n{stderr}"
        return combined if combined.strip() else "(Code ran successfully with no output)"
    except Exception as e:
        return f"Error running code: {e}"
def render_message_with_code_blocks(content, key_prefix):
    # Split the message into text and code blocks
    pattern = r"```(\w+)?\n(.*?)```"
    parts = re.split(pattern, content, flags=re.DOTALL)
    # parts alternates: [text, lang, code, text, lang, code, ...]
    i = 0
    block_index = 0
    while i < len(parts):
        text_part = parts[i]
        if text_part.strip():
            st.markdown(text_part)
        if i + 2 < len(parts):
            lang = (parts[i+1] or "text").lower()
            code = parts[i+2]
            st.code(code, language=lang)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    "Download file",
                    data=code,
                    file_name=f"snippet_{block_index}.{ 'py' if lang=='python' else 'txt'}",
                    key=f"{key_prefix}_dl_{block_index}"
                )
            with col2:
                piston_lang = LANGUAGE_MAP.get(lang)
                if piston_lang:
                    if st.button("Run this code", key=f"{key_prefix}_run_{block_index}"):
                        with st.spinner("Running..."):
                            output = run_code_with_piston(piston_lang, code)
                        st.text_area("Output", output, height=150, key=f"{key_prefix}_out_{block_index}")
                else:
                    st.caption("(Run not supported for this language)")
            block_index += 1
        i += 3

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_message_with_code_blocks(msg["content"], key_prefix=f"msg{idx}")
        else:
            st.write(msg["content"])

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
        render_message_with_code_blocks(reply, key_prefix=f"msg{len(st.session_state.messages)}")
