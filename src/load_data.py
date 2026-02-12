import json
from typing import List
from src.models import UserProfile

# Database of coordinates for supported cities
CITY_COORDS = {
    "Bucharest": (44.4268, 26.1025),
    "Cluj": (46.7712, 23.6236),
    "Berlin": (52.5200, 13.4050),
    "Vienna": (48.2082, 16.3738),
    "Timisoara": (45.7489, 21.2087),
    "Prague": (50.0755, 14.4378),
    "Iasi": (47.1585, 27.6014),
    "Warsaw": (52.2297, 21.0122),
    "Sofia": (42.6977, 23.3219),
    "Budapest": (47.4979, 19.0402),
    "Lisbon": (38.7223, -9.1393),
    "Rome": (41.9028, 12.4964),
    "Madrid": (40.4168, -3.7038),
    "Paris": (48.8566, 2.3522),
    "London": (51.5074, -0.1278),
    "Amsterdam": (52.3676, 4.9041),
    "Zurich": (47.3769, 8.5417),
    "Oslo": (59.9139, 10.7522),
    "Stockholm": (59.3293, 18.0686),
    "Helsinki": (60.1699, 24.9384),
    "Bali / Remote": (-8.6500, 115.2167),
    "Copenhagen": (55.6761, 12.5683),
    "Milan": (45.4642, 9.1900),
    "Tokyo / Remote": (35.6895, 139.6917),
    "Barcelona": (41.3851, 2.1734),
    "Dublin": (53.3498, -6.2603),
    "Athens": (37.9838, 23.7275),
    "Reykjavik": (64.1265, -21.8271),
    "Istanbul": (41.0082, 28.9784),
    "Moscow": (55.7558, 37.6173),
    "New York": (40.7128, -74.0060),
    "Tel Aviv": (32.0853, 34.7818)
}

def load_profiles(filepath: str) -> List[UserProfile]:
    """
    Loads profiles from a JSON file, enriches them with GPS coordinates,
    and returns a list of UserProfile Pydantic models.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from '{filepath}': {e}")
        return []

    profiles = []
    for item in data:
        # Resolve coordinates based on city name
        location_raw = item.get("location", "Unknown")
        lat, lon = CITY_COORDS.get(location_raw, (0.0, 0.0))

        if (lat, lon) == (0.0, 0.0) and location_raw != "Unknown":
            print(f"Warning: Coordinates not found for '{location_raw}'. Using default (0,0).")

        try:
            # Explicitly mapping all fields including new gender fields
            profile = UserProfile(
                id=item["id"],
                age=item["age"],
                gender=item["gender"],
                preferred_gender=item["preferred_gender"],
                location=location_raw,
                bio=item.get("bio", ""),
                values=item.get("values", []),
                goals=item.get("goals", []),
                lifestyle=item.get("lifestyle", []),
                deal_breakers=item.get("deal_breakers", []),
                looking_for=item.get("looking_for", ""),
                lat=lat,
                lon=lon
            )
            profiles.append(profile)
        except KeyError as e:
            print(f"Error: Profile {item.get('id', 'unknown')} is missing a required field: {e}")
            continue
        except Exception as e:
            print(f"Error validating profile {item.get('id', 'unknown')}: {e}")
            continue

    return profiles