import os
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

from src.load_data import load_profiles
from src.profile_text import profile_to_text
from src.embeddings import embed_texts
from src.retrieval import VectorIndex

# -------------------------------
# Настройка OpenAI
# -------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# Загрузка профилей
# -------------------------------
profiles = load_profiles("data/raw_profiles.json")
texts = [profile_to_text(p) for p in profiles]
embeddings = embed_texts(texts)

# -------------------------------
# Создаем FAISS index
# -------------------------------
dim = embeddings.shape[1]
index = VectorIndex(dim)
index.add(embeddings)

# -------------------------------
# Выбираем пользователя
# -------------------------------
me = profiles[0]
me_text = profile_to_text(me)
me_emb = embed_texts([me_text])

# -------------------------------
# Поиск кандидатов
# -------------------------------
k = len(profiles)
ids, scores = index.search(me_emb, k=k)

# Сортировка
sorted_pairs = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)

top_k = 5
bottom_k = 3

top_candidates = [profiles[i] for i, s in sorted_pairs if profiles[i].id != me.id][:top_k]
bottom_candidates = [profiles[i] for i, s in sorted_pairs if profiles[i].id != me.id][-bottom_k:]

# -------------------------------
# Функция вызова LLM для объяснения
# -------------------------------
def explain_match(user, candidate):
    prompt = f"""
Ты — эксперт по знакомствам. Оцени совместимость между двумя людьми.
User:
{user.bio}
Values: {user.values}
Lifestyle: {user.lifestyle}
Deal_breakers: {user.deal_breakers}

Candidate:
{candidate.bio}
Values: {candidate.values}
Lifestyle: {candidate.lifestyle}
Deal_breakers: {candidate.deal_breakers}

Правила:
1. Считай, что deal_breakers критичны — если есть конфликт, score падает.
2. Учитывай lifestyle и ценности.
3. Дай score от 0 до 100, где 100 = идеально совместимы.
4. Дай 2–3 причины, почему подходят/не подходят.
5. Дай verdict: strong_match | weak_match | no_match.

Выведи в формате JSON:
{{
  "score": <число>,
  "reasons": [<строки>],
  "verdict": "<строка>"
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content
    return content

# -------------------------------
# Генерация объяснений
# -------------------------------
print(f"\nTop {top_k} matches with explanations:\n")
for c in top_candidates:
    print(f"Candidate {c.id}:\n", explain_match(me, c), "\n")

print(f"\nBottom {bottom_k} matches with explanations:\n")
for c in bottom_candidates:
    print(f"Candidate {c.id}:\n", explain_match(me, c), "\n")
