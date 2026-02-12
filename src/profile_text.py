from src.models import UserProfile

def profile_to_text(p: UserProfile) -> str:
    """Превращает объект профиля в богатый текстом формат для эмбеддинга."""
    return f"""
About me: {p.bio}
Values: {", ".join(p.values)}
Lifestyle: {", ".join(p.lifestyle)}
Relationship goals: {", ".join(p.goals)}
Deal breakers: {", ".join(p.deal_breakers)}
Looking for: {p.looking_for}
Location: {p.location}
""".strip()