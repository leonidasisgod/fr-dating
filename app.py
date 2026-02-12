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

# Инициализация клиента OpenAI (ключ берется из Secrets в облаке или .env локально)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Custom CSS для стиля
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #ff4b4b; color: white; border: none; }
    .match-card { padding: 20px; border-radius: 15px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px; }
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)


# --- ENGINE LOADING (WITH CACHE & PERSISTENCE) ---
@st.cache_resource
def get_engine():
    """Загружает движок: либо с диска, либо создает новый."""
    engine = VectorIndex(dim=1536)

    # Пытаемся загрузить готовую базу с диска
    if engine.load():
        st.sidebar.caption("🟢 База загружена с диска")
        return engine, engine.profiles

    # Если базы нет на диске, создаем её (через OpenAI)
    with st.status("Создание новой векторной базы...") as status:
        profiles = load_profiles("data/raw_profiles.json")
        texts = [profile_to_text(p) for p in profiles]

        # Получаем эмбеддинги
        raw_embs = embed_texts(texts)
        embs = np.array(raw_embs).astype('float32')

        # Добавляем в движок
        engine.add(embs, profiles)
        # Сохраняем на диск для следующего раза
        engine.save()

        status.update(label="✅ База создана и сохранена!", state="complete")

    return engine, profiles


index, profiles = get_engine()

# --- SIDEBAR: SETTINGS ---
st.sidebar.title("👤 Личный кабинет")
user_ids = [p.id for p in profiles]
my_id = st.sidebar.selectbox("Войти как:", user_ids, index=user_ids.index("u101") if "u101" in user_ids else 0)
me = next(p for p in profiles if p.id == my_id)

st.sidebar.success(f"Вы вошли как: {me.id}")
st.sidebar.info(f"📍 Город: {me.location}\n\n🎭 Поиск: {me.preferred_gender}")

# Кнопка сброса базы (если обновил JSON)
if st.sidebar.button("♻️ Пересобрать базу"):
    st.cache_resource.clear()
    # Удаляем файлы индекса, если они есть
    if os.path.exists("data/vector_db/index.faiss"):
        os.remove("data/vector_db/index.faiss")
    st.rerun()

# --- MAIN UI ---
st.title("💖 Neural Matchmaker")
st.write(f"Привет, **{me.id}**! Давай найдем идеальный мэтч в **Клуже**.")

tab1, tab2 = st.tabs(["🚀 Поиск", "👩‍❤️‍👨 Проверка пары"])

with tab1:
    st.subheader("Твои идеальные кандидаты")
    alpha = st.slider("Баланс: Вайб (AI) vs Ключевые слова", 0.0, 1.0, 0.7)

    if st.button("Найти мэтчи"):
        # Получаем вектор текущего пользователя
        me_emb = embed_texts([profile_to_text(me)])
        # Используем наш "железный" поиск
        matches = index.search_hybrid(me, me_emb, k=3, alpha=alpha)

        if not matches:
            st.warning("Никого не найдено. Попробуй изменить настройки или город.")

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
                col3.metric("Distance", f"{m['distance']} км")

                with st.expander("🤖 Вердикт ИИ"):
                    prompt = f"Analyze synergy between {me.id} and {cand.id}. Bio A: {me.bio}. Bio B: {cand.bio}. 2 witty sentences."
                    res = client.chat.completions.create(model="gpt-4o-mini",
                                                         messages=[{"role": "user", "content": prompt}])
                    st.write(res.choices[0].message.content)

with tab2:
    st.subheader("Проверка нашей химии")
    partner_id = st.selectbox("Выбери партнера:", [p.id for p in profiles if p.id != me.id],
                              index=user_ids.index("u102") - 1 if "u102" in user_ids and my_id != "u102" else 0)

    if st.button("Проанализировать нас"):
        partner = next(p for p in profiles if p.id == partner_id)

        # Безопасная математика совместимости (fix TypeError)
        emb_me = np.array(embed_texts([profile_to_text(me)])).flatten().astype('float32')
        emb_pa = np.array(embed_texts([profile_to_text(partner)])).flatten().astype('float32')

        # Нормализуем векторы перед умножением
        emb_me /= np.linalg.norm(emb_me)
        emb_pa /= np.linalg.norm(emb_pa)

        vibe_score = float(np.dot(emb_me, emb_pa))

        st.divider()
        st.balloons()

        st.markdown(f"### Сила вашей связи: **{int(vibe_score * 100)}%**")
        st.progress(vibe_score)

        # Глубокий вердикт для пары
        with st.spinner("ИИ изучает ваши профили..."):
            prompt = (f"Deep compatibility analysis for {me.id} and {partner.id}. "
                      f"Context: They are in Cluj. {me.id}: {me.bio}. {partner.id}: {partner.bio}. "
                      f"Write 3 romantic and deep sentences about why they fit together.")
            res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
            st.write(f"✨ **AI Counselor:** {res.choices[0].message.content}")