import faiss
import numpy as np
import math
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
    """Проверяет, нет ли в профиле кандидата вещей, которые юзер не приемлет."""
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
    def __init__(self, dim: int):
        self.dim = dim
        self.profiles = []
        self.embeddings = None  # Храним для ручного скоринга на малых выборках
        self.bm25 = None

    def add(self, vectors: np.ndarray, profiles: list):
        """Добавляет профили и настраивает текстовый поиск."""
        faiss.normalize_L2(vectors)
        self.embeddings = vectors
        self.profiles = profiles

        # Подготовка BM25 (по био, ценностям и целям)
        corpus = [
            (p.bio + " " + " ".join(p.values) + " " + " ".join(p.goals)).lower().split()
            for p in profiles
        ]
        self.bm25 = BM25Okapi(corpus)

    def search_hybrid(self, me, me_emb, max_km=2000, k=5, alpha=0.7):
        """
        Умный поиск с учетом гендерных предпочтений, дистанции и дилбрейкеров.
        """
        # Нормализуем вектор запроса
        faiss.normalize_L2(me_emb)

        # Считаем BM25 для всех заранее
        tokenized_query = (me.bio + " " + " ".join(me.values)).lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)

        results = []

        for idx, candidate in enumerate(self.profiles):
            # --- HARD FILTERS ---

            # 1. Исключаем себя
            if candidate.id == me.id:
                continue

            # 2. МУЖСКОЙ/ЖЕНСКИЙ ФИЛЬТР (Взаимная совместимость)
            # Подходит ли кандидат мне?
            me_compatible = (me.preferred_gender == "all" or me.preferred_gender == candidate.gender)
            # Подхожу ли я кандидату?
            cand_compatible = (candidate.preferred_gender == "all" or candidate.preferred_gender == me.gender)

            if not (me_compatible and cand_compatible):
                continue

            # 3. ГЕО ФИЛЬТР
            dist = calculate_distance(me.lat, me.lon, candidate.lat, candidate.lon)
            if dist > max_km:
                continue

            # 4. DEAL-BREAKERS
            if check_deal_breakers(me, candidate):
                continue

            # --- SOFT SCORING ---

            # Векторное сходство (Cosine Similarity)
            # .flatten() гарантирует, что оба вектора — просто списки чисел одинаковой длины
            v_score = float(np.dot(me_emb.flatten(), self.embeddings[idx].flatten()))

            # Текстовое сходство (BM25)
            kw_score = float(bm25_scores[idx])

            # Гибридный результат
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

        # Сортировка по убыванию и возврат топ-K
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]