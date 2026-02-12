import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

from src.load_data import load_profiles
from src.profile_text import profile_to_text
from src.embeddings import embed_texts, normalize_embeddings
from src.retrieval import VectorIndex

# Initialize Environment and Client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("CRITICAL ERROR: OpenAI API Key not found. Check your .env file.")
    sys.exit(1)

client = OpenAI(api_key=api_key)


def explain_match_llm(user, candidate, match_data):
    """
    Generates a witty match explanation using GPT-4o-mini.
    Prevents repetitive phrasing by enforcing creative synthesis.
    """
    prompt = f"""
    You are an elite, sophisticated AI Matchmaker. 
    Analyze the synergy between these two individuals:

    USER PROFILE:
    - Bio: {user.bio}
    - Values: {', '.join(user.values)}
    - Seeking: {user.looking_for}

    CANDIDATE PROFILE:
    - Bio: {candidate.bio}
    - Values: {', '.join(candidate.values)}
    - Lifestyle: {', '.join(candidate.lifestyle)}

    CONSTRAINTS:
    1. Write exactly 2 sentences explaining the 'vibe' connection.
    2. ABSOLUTELY FORBIDDEN: Do not use direct phrases or adjectives from the bios (e.g., if bio says 'dusty hands', do not use 'dusty' or 'hands').
    3. Use metaphors and high-level personality synthesis.
    4. Be witty, elegant, and positive.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"A cosmic connection is detected, but words fail the AI right now. (Error: {e})"


def main():
    # --- 1. Data Ingestion ---
    print("--- 1. Loading Profiles ---")
    profiles = load_profiles("data/raw_profiles.json")
    if not profiles:
        print("No profiles found. Termination.")
        return

    texts = [profile_to_text(p) for p in profiles]
    print(f"Successfully loaded {len(profiles)} profiles.")

    # --- 2. Vector Generation ---
    print("\n--- 2. Generating Semantic Embeddings ---")
    raw_embs = embed_texts(texts)
    embs = normalize_embeddings(raw_embs)
    print("Embeddings generated and normalized.")

    # --- 3. Hybrid Indexing ---
    print("\n--- 3. Initializing Hybrid Vector Index ---")
    index = VectorIndex(dim=embs.shape[1])
    index.add(embs, profiles)
    print("Index is live.")

    # --- 4. User Targeting ---
    # Targeting u3: The Berlin Sound Engineer (Non-binary, seeking all)
    target_id = "u3"
    me = next((p for p in profiles if p.id == target_id), profiles[0])

    print(f"\n🎧 SEARCHING MATCHES FOR: {me.id} | {me.location} ({me.gender})")
    print(f"   Current Bio: {me.bio}")

    # Generate query embedding for u3
    me_text = profile_to_text(me)
    me_emb = embed_texts([me_text])

    # --- 5. Execution ---
    # alpha=0.7 prioritizes 'Vibe' (Semantic) over 'Keys' (Keywords)
    matches = index.search_hybrid(me, me_emb, k=3, alpha=0.7)

    print(f"\n🔥 TOP {len(matches)} MATCHES FOR THE TECHNO SOUL:\n")

    for m in matches:
        cand = m['profile']
        v_score = m['reasons']['vector']
        k_score = m['reasons']['keyword']

        # Interpret the scores for the user
        score_note = ""
        if v_score > 0.65:
            score_note = "→ [Deep Vibe Match: Highly aligned worldview and energy]"
        elif k_score > 0.15:
            score_note = "→ [Keyword Match: Significant overlap in hobbies and interests]"
        else:
            score_note = "→ [Balanced Match: A solid mix of intuition and shared facts]"

        print(f"MATCH: {cand.id} ({cand.location}) | {cand.gender}")
        print(f"SCORE: {m['score']:.3f} {score_note}")
        print(f"   - Vibe Index: {v_score} (Semantic soul-matching)")
        print(f"   - Keys Index: {k_score} (Literal interest overlap)")
        print(f"   - Distance: {m['distance']} km")

        # Explain the logic via LLM
        print("   Analyzing chemistry...")
        verdict = explain_match_llm(me, cand, m)
        print(f"   VERDICT: {verdict}\n")
        print("-" * 65)


if __name__ == "__main__":
    main()