from pydantic import BaseModel
from typing import List, Optional

class UserProfile(BaseModel):
    id: str
    age: int
    gender: str  # "male" / "female" / "non-binary"
    preferred_gender: str  # "male" / "female" / "all"
    location: str
    bio: str
    values: List[str]
    goals: List[str]
    lifestyle: List[str]
    deal_breakers: List[str]
    looking_for: str
    lat: float
    lon: float