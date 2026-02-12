import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# Твои модули
from src.load_data import load_profiles
from src.profile_text import profile_to_text
from src.embeddings import embed_texts, normalize_embeddings
from src.retrieval import VectorIndex

# --- CONFIG ---
st.set_page_config(page_title="Neural Matchmaker", page_icon="💖", layout="centered")
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Custom CSS для красоты (стеклянный эффект и шрифты)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #ff4b4b; color: white; }
    .match-card { padding: 20px; border-radius: 15px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)


# Кэшируем загрузку данных, чтобы не ждать при каждом клике
@st.cache_resource
def get_engine():
    profiles = load_profiles("data/raw_profiles.json")
    texts = [profile_to_text(p) for p in profiles]
    embs = normalize_embeddings(embed_texts(texts))
    index = VectorIndex(dim=embs.shape[1])
    index.add(embs, profiles)
    return index, profiles


index, profiles = get_engine()

# --- SIDEBAR: LOGIN ---
st.sidebar.title("👤 Account")
user_ids = [p.id for p in profiles]
my_id = st.sidebar.selectbox("Log in as:", user_ids, index=user_ids.index("u101") if "u101" in user_ids else 0)
me = next(p for p in profiles if p.id == my_id)

st.sidebar.success(f"Logged in as {me.id}")
st.sidebar.info(f"📍 Location: {me.location}\n\n🎭 Seeking: {me.preferred_gender}")

# --- MAIN UI ---
st.title("💖 Neural Matchmaker")
st.write(f"Welcome back, **{me.id}**! Let's find some cosmic connections in **Cluj**.")

tab1, tab2 = st.tabs(["🚀 Discovery", "👩‍❤️‍👨 Couple Test"])

with tab1:
    st.subheader("Find your perfect match")
    alpha = st.slider("Balance: Vibe vs Keywords", 0.0, 1.0, 0.7)

    if st.button("Generate Matches"):
        me_emb = embed_texts([profile_to_text(me)])
        matches = index.search_hybrid(me, me_emb, k=3, alpha=alpha)

        for m in matches:
            cand = m['profile']
            with st.container():
                st.markdown(f"""
                <div class="match-card">
                    <h3>{cand.id} | {cand.location}</h3>
                    <p><i>"{cand.bio}"</i></p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Match Score", f"{int(m['score'] * 100)}%")
                col2.metric("Vibe", f"{int(m['reasons']['vector'] * 100)}%")
                col3.metric("Distance", f"{m['distance']} km")

                with st.expander("🤖 Read AI Verdict"):
                    # Здесь вызываем твой explain_match_llm
                    prompt = f"Analyze synergy between {me.id} and {cand.id}. Bio A: {me.bio}. Bio B: {cand.bio}. 2 sentences."
                    res = client.chat.completions.create(model="gpt-4o-mini",
                                                         messages=[{"role": "user", "content": prompt}])
                    st.write(res.choices[0].message.content)

with tab2:
    st.subheader("Compatibility Analysis")
    partner_id = st.selectbox("Select Partner:", [p.id for p in profiles if p.id != me.id],
                              index=user_ids.index("u102") - 1 if "u102" in user_ids else 0)

    if st.button("Analyze Our Connection"):
        partner = next(p for p in profiles if p.id == partner_id)

        # Математика совместимости
        emb_me = normalize_embeddings(embed_texts([profile_to_text(me)]))
        emb_pa = normalize_embeddings(embed_texts([profile_to_text(partner)]))
        vibe_score = float(np.dot(emb_me, emb_pa.T))

        st.divider()
        st.balloons()

        st.markdown(f"### Connection Strength: **{int(vibe_score * 100)}%**")
        st.progress(vibe_score)

        # Глубокий вердикт для пары
        prompt = f"Deep analysis for u101 and u102 in Cluj. He loves history/literature. She loves languages/travel. Write 3 romantic sentences about their synergy."
        res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        st.write(f"✨ **AI Counselor:** {res.choices[0].message.content}")