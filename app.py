import streamlit as st
import re
import base64
import requests
import numpy as np
from groq import Groq
from supabase import create_client
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
db = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

st.set_page_config(layout="wide")

MODEL_NAME = "qwen/qwen3.6-27b"  # multimodal: handles text + image understanding

ARTIFACT_INSTRUCTIONS = """
When the user asks you to build something visual or interactive (a webpage, a game, a diagram, a small app, an SVG graphic), respond with the code wrapped EXACTLY like this:

<artifact type="html" title="Short title here">
...complete, self-contained HTML/CSS/JS code here...
</artifact>

When the user asks you to generate, draw, or create a NEW image (no image attached, or not asking to change an existing one), respond with:

<artifact type="image" title="Short title here">
a detailed, vivid description of the image to generate
</artifact>

When the user has attached an image AND asks you to edit, modify, transform, or change it (e.g. "make this snowy", "turn this into a cartoon"), respond with:

<artifact type="edit_image" title="Short title here">
a short prompt describing the transformation to apply, written for an image editing AI (e.g. "add falling snow, winter atmosphere")
</artifact>

When the user asks you to run, execute, or test a piece of Python code, or wants to see the actual output of a script (not just read the code), respond with:

<artifact type="python" title="Short title here">
complete Python code to execute
</artifact>

If the user attaches an image and just asks you to describe or answer questions about it, respond normally in plain text with no artifact tag.
"""

PERSONAS = {
    "Friendly Helper": "You are a friendly, helpful assistant who explains things simply.",
    "Math Tutor": "You are a patient, encouraging math tutor for beginners. Explain step by step.",
    "D&D Game Master": "You are a creative Dungeons & Dragons game master narrating an adventure.",
    "Code Reviewer": "You are a senior software engineer giving direct, constructive code feedback.",
    "Custom": ""
}

for key, default in [
    ("messages", []), ("current_artifact", None),
    ("doc_chunks", None), ("doc_vectorizer", None),
    ("doc_matrix", None), ("doc_name", None),
    ("artifact_visible", True), ("pending_image", None),
    ("last_attached_image_bytes", None), ("suggested_title", "")
]:
    if key not in st.session_state:
        st.session_state[key] = default


def extract_artifact(text):
    match = re.search(r'<artifact type="(.*?)" title="(.*?)">(.*?)</artifact>', text, re.DOTALL)
    if not match:
        return text, None
    art_type, title, code = match.groups()
    clean_text = text[:match.start()] + text[match.end():]
    return clean_text.strip(), {"type": art_type, "title": title, "code": code.strip()}


def generate_image_cloudflare(prompt):
    account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
    token = st.secrets["CLOUDFLARE_API_TOKEN"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    try:
        response = requests.post(
            url, headers={"Authorization": f"Bearer {token}"},
            json={"prompt": prompt}, timeout=60
        )
    except Exception as e:
        return None, f"Request failed: {e}"
    if response.status_code != 200:
        return None, f"Status {response.status_code}: {response.text[:300]}"
    try:
        data = response.json()
        image_bytes = base64.b64decode(data["result"]["image"])
        return image_bytes, None
    except Exception as e:
        return None, f"Unexpected response format: {response.text[:300]}"


def edit_image_cloudflare(image_bytes, prompt, strength=0.7):
    account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
    token = st.secrets["CLOUDFLARE_API_TOKEN"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/runwayml/stable-diffusion-v1-5-img2img"

    b64_input = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"prompt": prompt, "image_b64": b64_input, "strength": strength},
            timeout=60
        )
    except Exception as e:
        return None, f"Request failed: {e}"

    if response.status_code != 200:
        return None, f"Status {response.status_code}: {response.text[:300]}"
    content_type = response.headers.get("content-type", "")
    if "image" in content_type:
        return response.content, None
    try:
        data = response.json()
        return base64.b64decode(data["result"]["image"]), None
    except Exception:
        return None, f"Unexpected response: {response.text[:300]}"


def fetch_image_from_url(url):
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            return response.content, None
        return None, f"Could not fetch image (status {response.status_code})"
    except Exception as e:
        return None, f"Fetch failed: {e}"


def wikipedia_summary(topic):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(topic)}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("type") != "disambiguation":
                return data.get("extract", "")
    except Exception:
        pass
    return ""


def run_python_judge0(code):
    url = "https://ce.judge0.com/submissions?wait=true"
    payload = {
        "source_code": code,
        "language_id": 109  # Python 3 — verify against GET /languages if this ever errors
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
    except Exception as e:
        return None, None, f"Request failed: {e}"

    if response.status_code not in (200, 201):
        return None, None, f"Status {response.status_code}: {response.text[:300]}"

    data = response.json()
    stdout = data.get("stdout") or ""
    stderr = data.get("stderr") or data.get("compile_output") or ""
    status = data.get("status", {}).get("description", "Unknown")
    return stdout, stderr, status


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def process_uploaded_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = file.read().decode("utf-8", errors="ignore")
    chunks = chunk_text(text)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    st.session_state.doc_chunks = chunks
    st.session_state.doc_vectorizer = vectorizer
    st.session_state.doc_matrix = matrix
    st.session_state.doc_name = file.name


def get_relevant_context(query, top_k=3):
    if st.session_state.doc_chunks is None:
        return ""
    query_vec = st.session_state.doc_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, st.session_state.doc_matrix)[0]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    relevant = [st.session_state.doc_chunks[i] for i in top_indices if sims[i] > 0]
    if not relevant:
        return ""
    return "\n\nRelevant excerpts from the uploaded document:\n" + "\n---\n".join(relevant)


def suggest_chat_title(messages):
    if not messages:
        return "Untitled chat"
    convo_snippet = "\n".join(
        f"{m['role']}: {m['content'] if isinstance(m['content'], str) else '[message with image]'}"
        for m in messages[:6]
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Suggest a short, plain 3-6 word title summarizing this conversation. Reply with ONLY the title, no quotes, no punctuation at the end."},
            {"role": "user", "content": convo_snippet}
        ],
        reasoning_format="hidden",
        reasoning_effort="none"
    )
    return response.choices[0].message.content.strip()


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    persona_choice = st.selectbox("Choose a persona", list(PERSONAS.keys()))
    if persona_choice == "Custom":
        system_prompt = st.text_area("Write your own system prompt", height=150)
    else:
        system_prompt = st.text_area("System prompt (editable)", value=PERSONAS[persona_choice], height=150)

    st.divider()
    st.subheader("Response mode")
    reasoning_on = st.toggle("Deep reasoning (slower, better for math/code)", value=False)

    st.divider()
    st.subheader("Upload a document")
    uploaded_file = st.file_uploader("PDF or text file", type=["pdf", "txt"])
    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("Reading document..."):
            process_uploaded_file(uploaded_file)
        st.success(f"Loaded: {uploaded_file.name} ({len(st.session_state.doc_chunks)} chunks)")
    elif st.session_state.doc_name:
        st.caption(f"Active document: {st.session_state.doc_name}")
        if st.button("Remove document"):
            st.session_state.doc_chunks = None
            st.session_state.doc_vectorizer = None
            st.session_state.doc_matrix = None
            st.session_state.doc_name = None
            st.rerun()

    st.divider()
    st.subheader("Save current chat")

    title_col, suggest_col = st.columns([3, 1])
    with title_col:
        save_title = st.text_input("Chat title", value=st.session_state.suggested_title, key="save_title_input")
    with suggest_col:
        st.write("")
        if st.button("✨", help="Suggest a title") and st.session_state.messages:
            with st.spinner("Thinking..."):
                st.session_state.suggested_title = suggest_chat_title(st.session_state.messages)
            st.rerun()

    if st.button("Save chat") and save_title and st.session_state.messages:
        artifact_to_save = None
        if st.session_state.current_artifact:
            # strip out raw image bytes / run output before saving to Supabase (jsonb can't hold binary, and we can regenerate anyway)
            artifact_to_save = {
                k: v for k, v in st.session_state.current_artifact.items()
                if k not in ("image_bytes", "error", "run_output", "run_error", "run_status")
            }
        db.table("chats").insert({
            "title": save_title,
            "persona": system_prompt,
            "messages": st.session_state.messages,
            "artifact": artifact_to_save
        }).execute()
        st.success("Saved!")
        st.session_state.suggested_title = ""

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
            st.session_state.current_artifact = full.data.get("artifact")
            st.session_state.artifact_visible = bool(full.data.get("artifact"))
            st.rerun()

    if st.button("Clear current chat"):
        st.session_state.messages = []
        st.session_state.current_artifact = None
        st.session_state.suggested_title = ""

# --- Full-width toggle: 2 columns only if an artifact is showing ---
show_artifact_panel = st.session_state.current_artifact and st.session_state.artifact_visible

if show_artifact_panel:
    chat_col, artifact_col = st.columns([1, 1])
else:
    chat_col = st.container()
    artifact_col = None

with chat_col:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part["type"] == "text":
                        st.write(part["text"])
                    elif part["type"] == "image_url":
                        st.image(part["image_url"]["url"])
            else:
                st.write(msg["content"])

    with st.expander("📎 Attach an image (optional)"):
        tab_upload, tab_url = st.tabs(["Upload", "From URL"])
        with tab_upload:
            attached_image = st.file_uploader("Image file", type=["png", "jpg", "jpeg"], key="img_attach")
            if attached_image:
                st.image(attached_image, width=150)
                st.session_state.pending_image = attached_image
        with tab_url:
            image_url = st.text_input("Paste an image URL", key="img_url_input")
            if image_url and st.button("Fetch image"):
                img_bytes, error = fetch_image_from_url(image_url)
                if img_bytes:
                    st.session_state.last_attached_image_bytes = img_bytes
                    st.image(img_bytes, width=150)
                    st.success("Image loaded — you can now ask to edit it")
                else:
                    st.error(error)

    user_input = st.chat_input("Say something...")

    if user_input:
        if st.session_state.pending_image:
            img_bytes = st.session_state.pending_image.getvalue()
            st.session_state.last_attached_image_bytes = img_bytes
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            mime = st.session_state.pending_image.type
            user_content = [
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]
            st.session_state.pending_image = None
        else:
            user_content = user_input

        st.session_state.messages.append({"role": "user", "content": user_content})
        with st.chat_message("user"):
            if isinstance(user_content, list):
                for part in user_content:
                    if part["type"] == "text":
                        st.write(part["text"])
                    elif part["type"] == "image_url":
                        st.image(part["image_url"]["url"])
            else:
                st.write(user_content)

        doc_context = get_relevant_context(user_input)
        full_system_prompt = system_prompt + "\n\n" + ARTIFACT_INSTRUCTIONS + doc_context
        api_messages = [{"role": "system", "content": full_system_prompt}] + st.session_state.messages

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                stream=True,
                reasoning_format="hidden",
                reasoning_effort="default" if reasoning_on else "none"
            )

            def text_generator():
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            full_reply = st.write_stream(text_generator)

        clean_reply, artifact = extract_artifact(full_reply)
        st.session_state.messages.append({"role": "assistant", "content": clean_reply})

        if artifact:
            st.session_state.current_artifact = artifact
            st.session_state.artifact_visible = True

        st.rerun()

if show_artifact_panel:
    with artifact_col:
        art = st.session_state.current_artifact
        header_col, close_col = st.columns([5, 1])
        with header_col:
            st.subheader("Artifact Preview")
            st.caption(art["title"])
        with close_col:
            st.write("")
            if st.button("✕", help="Close preview"):
                st.session_state.artifact_visible = False
                st.rerun()

        if art["type"] == "image":
            if "image_bytes" not in art:
                wiki_facts = wikipedia_summary(art["title"])
                final_prompt = art["code"]
                if wiki_facts:
                    final_prompt += f"\n\nFactual reference for accuracy: {wiki_facts[:400]}"

                with st.spinner("Generating image..."):
                    image_bytes, error = generate_image_cloudflare(final_prompt)
                art["image_bytes"] = image_bytes
                art["error"] = error
                art["used_wiki_facts"] = bool(wiki_facts)
                st.session_state.current_artifact = art

            if art.get("image_bytes"):
                st.image(art["image_bytes"], caption=art["title"], use_container_width=True)
                st.download_button("Download image", art["image_bytes"], file_name=f"{art['title']}.png", mime="image/png")
                if art.get("used_wiki_facts"):
                    st.caption("ℹ️ Grounded with a Wikipedia summary for accuracy")
            else:
                st.error(f"Image generation failed: {art.get('error')}")
            st.caption(f"Prompt used: {art['code']}")

        elif art["type"] == "edit_image":
            if "image_bytes" not in art:
                if st.session_state.last_attached_image_bytes:
                    with st.spinner("Editing image..."):
                        image_bytes, error = edit_image_cloudflare(
                            st.session_state.last_attached_image_bytes, art["code"]
                        )
                    art["image_bytes"] = image_bytes
                    art["error"] = error
                else:
                    art["image_bytes"] = None
                    art["error"] = "No attached image found to edit."
                st.session_state.current_artifact = art

            if art.get("image_bytes"):
                st.image(art["image_bytes"], caption=art["title"], use_container_width=True)
                st.download_button("Download image", art["image_bytes"], file_name=f"{art['title']}.png", mime="image/png")
            else:
                st.error(art.get("error"))
            st.caption(f"Edit applied: {art['code']}")

        elif art["type"] == "python":
            if "run_output" not in art:
                with st.spinner("Running code..."):
                    stdout, stderr, status = run_python_judge0(art["code"])
                art["run_output"] = stdout
                art["run_error"] = stderr
                art["run_status"] = status
                st.session_state.current_artifact = art

            st.caption(f"Status: {art['run_status']}")
            tab1, tab2 = st.tabs(["Output", "Code"])
            with tab1:
                if art["run_output"]:
                    st.code(art["run_output"], language=None)
                if art["run_error"]:
                    st.error(art["run_error"])
                if not art["run_output"] and not art["run_error"]:
                    st.info("No output produced.")
            with tab2:
                st.code(art["code"], language="python")
            st.download_button("Download code", art["code"], file_name=f"{art['title']}.py", mime="text/plain")

        elif art["type"] == "html":
            tab1, tab2 = st.tabs(["Preview", "Code"])
            with tab1:
                st.components.v1.html(art["code"], height=500, scrolling=True)
            with tab2:
                edited_code = st.text_area("Edit the code", value=art["code"], height=400, key=f"editor_{id(art)}")
                if st.button("Update preview"):
                    art["code"] = edited_code
                    st.session_state.current_artifact = art
                    st.rerun()
            st.download_button("Download code", art["code"], file_name=f"{art['title']}.html", mime="text/html")

elif st.session_state.current_artifact and not st.session_state.artifact_visible:
    st.info(f"Preview closed: **{st.session_state.current_artifact['title']}**")
    if st.button("Reopen preview"):
        st.session_state.artifact_visible = True
        st.rerun()
