from src.load_data import load_profiles
from src.profile_text import profile_to_text
from src.embeddings import embed_texts
from src.retrieval import VectorIndex
import numpy as np

# Загружаем профили
profiles = load_profiles("data/raw_profiles.json")
texts = [profile_to_text(p) for p in profiles]
embeddings = embed_texts(texts)

# Создаем FAISS index
dim = embeddings.shape[1]
index = VectorIndex(dim)
index.add(embeddings)

# Выбираем пользователя для проверки
me = profiles[0]
me_text = profile_to_text(me)
me_emb = embed_texts([me_text])

# Делаем поиск всех кандидатов (k = total)
ids, scores = index.search(me_emb, k=len(profiles))

# Сортируем по возрастанию score (наименее похожие в начале)
sorted_pairs = sorted(zip(ids, scores), key=lambda x: x[1])

print(f"\nWorst matches for {me.id}:\n")

for i, score in sorted_pairs[:5]:  # 5 самых неподходящих
    candidate = profiles[i]
    if candidate.id == me.id:
        continue
    print(f"- {candidate.id} | score={score:.3f}")
    print(f"  bio: {candidate.bio}\n")
