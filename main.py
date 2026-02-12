import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

# Local imports (assuming your project structure is correct)
from src.load_data import load_profiles
from src.profile_text import profile_to_text
from src.embeddings import embed_texts, normalize_embeddings
from src.retrieval import VectorIndex

# 1. Init Configuration
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

client = OpenAI(api_key=api_key)


def explain_match_llm(user, candidate, match_data):
    """
    Generates a creative match explanation using GPT-4o-mini.
    Includes error handling to prevent crash on API failure.
    """
    prompt = f"""
    You are an elite Dating Coach AI. Analyze the chemistry between these two:

    User ({user.location}): {user.bio}
    - Values: {', '.join(user.values)}
    - Seeking: {user.looking_for}

    Match ({candidate.location}): {candidate.bio}
    - Values: {', '.join(candidate.values)}
    - Lifestyle: {', '.join(candidate.lifestyle)}

    Metrics:
    - Compatibility Score: {match_data['score']:.2f}
    - Distance: {match_data['distance']} km

    Task:
    Write a witty, insightful 2-sentence verdict on why they might click.
    Focus on the intersection of their vibes (e.g., "His chaos meets her order").
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI Explanation Failed: {str(e)}]"


def main():
    data_path = "data/raw_profiles.json"

    # --- 1. Loading Data ---
    print(f"--- 1. Loading Profiles from {data_path} ---")
    try:
        profiles = load_profiles(data_path)
    except FileNotFoundError:
        print(f"Error: File {data_path} not found. Please create it first.")
        return

    if not profiles:
        print("Error: No profiles loaded.")
        return

    # Prepare text representation for embedding
    texts = [profile_to_text(p) for p in profiles]
    print(f"✅ Loaded {len(profiles)} profiles.")

    # --- 2. Generating Vectors ---
    print("\n--- 2. Generating Embeddings (OpenAI) ---")
    try:
        # Embed all profiles at once (batch processing)
        raw_embs = embed_texts(texts)
        embs = normalize_embeddings(raw_embs)
        print(f"✅ Generated {len(embs)} vectors.")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return

    # --- 3. Creating Index ---
    print("\n--- 3. Building Hybrid Index ---")
    index = VectorIndex(dim=embs.shape[1])
    index.add(embs, profiles)
    print("✅ Index ready.")

    # --- 4. Select a User ---
    # Let's pick 'u1' - The Piano Restorer from Bucharest
    target_id = "u1"
    me = next((p for p in profiles if p.id == target_id), profiles[0])

    print(f"\n🔎 SEARCHING MATCHES FOR: {me.id} | {me.location}")
    print(f"📝 Bio: \"{me.bio}\"")
    print(f"❤️ Looking for: {me.looking_for}")

    # Generate embedding for the user specifically for the query
    # (Note: In a real app, 'me' might be a new user not yet in the index)
    me_text = profile_to_text(me)
    me_emb_raw = embed_texts([me_text])
    me_emb = normalize_embeddings(me_emb_raw)

    # --- 5. Hybrid Search ---
    print("\n🚀 Running Hybrid Search...")
    matches = index.search_hybrid(
        me=me,
        me_emb=me_emb,
        max_km=2500,  # Wide radius to find matches across Europe
        k=3,  # Top 3 results
        alpha=0.7  # 70% Semantic (Vibe), 30% Keywords
    )

    # --- 6. Display Results ---
    print(f"\n🏆 Found {len(matches)} candidates:\n")

    for i, m in enumerate(matches):
        cand = m['profile']
        print(f"#{i + 1} MATCH: {cand.id} ({cand.location}) - {cand.bio[:60]}...")
        print(f"   📊 Score: {m['score']:.3f} (Vibe: {m['reasons']['vector']:.2f}, Keys: {m['reasons']['keyword']:.2f})")
        print(f"   📍 Distance: {m['distance']} km")

        # Generate AI verdict
        print("   🤖 AI Verdict: Generating...")
        explanation = explain_match_llm(me, cand, m)
        print(f"   👉 {explanation}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()

# dusty hands
# score Score: 0.515 (Vibe: 0.68, Keys: 0.13)

