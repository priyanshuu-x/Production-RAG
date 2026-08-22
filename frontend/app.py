import streamlit as st
import requests
import markdown as md

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="PaperMind", page_icon="📄", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

.stApp {
    background-color: #12181F;
    color: #F1E9D8;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

.pm-hero {
    font-family: 'Source Serif 4', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #F1E9D8;
    margin-bottom: 0;
    letter-spacing: -0.02em;
}

.pm-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #E3A542;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
}

.stTextInput input {
    background-color: #1B232D;
    color: #F1E9D8;
    border: 1px solid #3A4552;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    padding: 0.7rem;
}

.stTextInput input:focus {
    border-color: #E3A542;
    box-shadow: 0 0 0 1px #E3A542;
}

.stButton button {
    background-color: #E3A542;
    color: #12181F;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
}

.stButton button:hover {
    background-color: #f0b862;
    color: #12181F;
}

.pm-answer-card {
    background-color: #F1E9D8;
    color: #12181F;
    font-family: 'Source Serif 4', serif;
    font-size: 1.15rem;
    line-height: 1.7;
    padding: 2rem;
    border-radius: 8px;
    margin-top: 1.5rem;
}

.pm-source-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7C9885;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
}

.pm-source-tag {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background-color: #1B232D;
    border-left: 3px solid #E3A542;
    padding: 0.6rem 1rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #F1E9D8;
}

.pm-source-num {
    background-color: #E3A542;
    color: #12181F;
    font-weight: 700;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="pm-hero">PaperMind</div>', unsafe_allow_html=True)
st.markdown('<div class="pm-tagline">ask questions, get cited answers.</div>', unsafe_allow_html=True)

# ---------- Input ----------
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input("Question", placeholder="What is retrieval augmented generation?", label_visibility="collapsed")
with col2:
    ask_clicked = st.button("Ask →", use_container_width=True)



# ---------- Query + Response ----------
if ask_clicked and question:
    with st.spinner("Reading the papers..."):
        response = requests.post(
            API_URL,
            json={"question": question, "top_k": 5},
        )

    if response.status_code == 200:
        data = response.json()

        answer_html = md.markdown(data["answer"])
        st.markdown(f'<div class="pm-answer-card">{answer_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="pm-source-label">Sources</div>', unsafe_allow_html=True)
        for source in data["sources"]:
            st.markdown(
                f'''<div class="pm-source-tag">
                    <span class="pm-source-num">{source['citation_number']}</span>
                    <span>{source['paper_id']} · chunk {source['chunk_id']}</span>
                </div>''',
                unsafe_allow_html=True,
            )
    else:
        st.error(f"Something went wrong: {response.status_code}")