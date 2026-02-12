import faiss
import numpy as np
import math
import pickle
import os
from rank_bm25 import BM25Okapi


def calculate_distance(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние по формуле гаверсинуса."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def check_deal_breakers(user, candidate):
    """Проверяет критические несовместимости."""
    cand_content = (
            candidate.bio + " " +
            " ".join(candidate.lifestyle) + " " +
            " ".join(candidate.values)
    ).lower()

    for db in user.deal_breakers:
        if db.lower() in cand_content:
            return True
    return False


class VectorIndex:
    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.profiles = []
        self.embeddings = None
        self.bm25 = None
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray, profiles: list):
        """Загружает данные в индекс."""
        # Приводим к float32 — это 'родной' тип для FAISS и numpy-операций
        vectors = np.ascontiguousarray(vectors).astype('float32')
        faiss.normalize_L2(vectors)

        self.embeddings = vectors
        self.profiles = profiles
        self.index.add(vectors)

        # Инициализация BM25
        corpus = [
            (p.bio + " " + " ".join(p.values) + " " + " ".join(p.goals)).lower().split()
            for p in profiles
        ]
        self.bm25 = BM25Okapi(corpus)

    def save(self, folder="data/vector_db"):
        """Сохраняет базу на диск, чтобы не платить за эмбеддинги снова."""
        if not os.path.exists(folder):
            os.makedirs(folder)

        faiss.write_index(self.index, f"{folder}/index.faiss")
        with open(f"{folder}/meta.pkl", "wb") as f:
            pickle.dump({
                "profiles": self.profiles,
                "embeddings": self.embeddings,
                "bm25": self.bm25
            }, f)
        print(f"✅ База сохранена в {folder}")

    def load(self, folder="data/vector_db"):
        """Загружает базу с диска."""
        if os.path.exists(f"{folder}/index.faiss"):
            self.index = faiss.read_index(f"{folder}/index.faiss")
            with open(f"{folder}/meta.pkl", "rb") as f:
                data = pickle.load(f)
                self.profiles = data["profiles"]
                self.embeddings = data["embeddings"]
                self.bm25 = data["bm25"]
            return True
        return False

    def search_hybrid(self, me, me_emb, max_km=2000, k=5, alpha=0.7):
        """Гибридный поиск с защитой от ошибок типов."""
        if self.embeddings is None:
            return []

        # ПОДГОТОВКА ЗАПРОСА
        # Превращаем me_emb в чистый float32 вектор (1D)
        query_vec = np.array(me_emb).flatten().astype('float32')
        # Нормализация для косинусного сходства
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Считаем BM25
        tokenized_query = (me.bio + " " + " ".join(me.values)).lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        max_bm25 = np.max(bm25_scores)
        if max_bm25 > 0:
            bm25_scores = bm25_scores / max_bm25

        results = []

        for idx, candidate in enumerate(self.profiles):
            # --- ФИЛЬТРЫ ---
            if candidate.id == me.id: continue

            # Гендерный фильтр
            me_compatible = (me.preferred_gender == "all" or me.preferred_gender == candidate.gender)
            cand_compatible = (candidate.preferred_gender == "all" or candidate.preferred_gender == me.gender)
            if not (me_compatible and cand_compatible): continue

            # Гео фильтр
            dist = calculate_distance(me.lat, me.lon, candidate.lat, candidate.lon)
            if dist > max_km: continue

            # Deal-breakers
            if check_deal_breakers(me, candidate): continue

            # --- СКОРИНГ ---
            # Векторное сходство (теперь без ошибок размерности)
            cand_vec = self.embeddings[idx].astype('float32')
            v_score = float(np.dot(query_vec, cand_vec))

            # Текстовое сходство
            kw_score = float(bm25_scores[idx])

            # Итоговый гибрид
            final_score = (alpha * v_score) + ((1 - alpha) * kw_score)

            results.append({
                "profile": candidate,
                "score": final_score,
                "distance": round(dist, 1),
                "reasons": {
                    "vector": round(v_score, 2),
                    "keyword": round(kw_score, 2)
                }
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]