"""Validation and prompt rendering for global user profiles."""

MAX_USER_PROFILES = 20
PROFILE_ID_PREFIX = "profile-"
EDITABLE_PROFILE_FIELDS = ("name", "style", "format", "constraints")
MAX_NAME_LENGTH = 80
MAX_PREFERENCE_LENGTH = 500
DEFAULT_PROFILE_FIELDS = {
    "name": "Пользователь",
    "style": "ясный и дружелюбный",
    "format": "краткие абзацы",
    "constraints": "без Markdown",
}


def normalize_profile(fields):
    """Return validated, trimmed editable profile fields."""
    if not isinstance(fields, dict) or set(fields) != set(EDITABLE_PROFILE_FIELDS):
        raise ValueError("Профиль имеет неверный формат.")
    normalized = {}
    limits = {"name": MAX_NAME_LENGTH, "style": MAX_PREFERENCE_LENGTH,
              "format": MAX_PREFERENCE_LENGTH, "constraints": MAX_PREFERENCE_LENGTH}
    for key in EDITABLE_PROFILE_FIELDS:
        value = fields[key]
        if not isinstance(value, str):
            raise ValueError("Все поля профиля должны быть строками.")
        value = value.strip()
        if not value or len(value) > limits[key]:
            raise ValueError("Поле профиля имеет неверную длину.")
        normalized[key] = value
    return normalized


def _profile_number(profile_id):
    if not isinstance(profile_id, str) or not profile_id.startswith(PROFILE_ID_PREFIX):
        return None
    value = profile_id[len(PROFILE_ID_PREFIX):]
    if not value.isascii() or not value.isdigit() or value.startswith("0"):
        return None
    number = int(value)
    return number if number > 0 else None


def profile_with_id(profile_id, fields):
    if _profile_number(profile_id) is None:
        raise ValueError("Идентификатор профиля имеет неверный формат.")
    return {"id": profile_id, **normalize_profile(fields)}


def default_profile():
    return profile_with_id(f"{PROFILE_ID_PREFIX}1", DEFAULT_PROFILE_FIELDS)


def valid_profile(profile):
    if not isinstance(profile, dict) or set(profile) != {"id", *EDITABLE_PROFILE_FIELDS}:
        return False
    try:
        normalized = normalize_profile({key: profile[key] for key in EDITABLE_PROFILE_FIELDS})
    except ValueError:
        return False
    return _profile_number(profile["id"]) is not None and normalized == {
        key: profile[key] for key in EDITABLE_PROFILE_FIELDS
    }


def valid_profile_collection(profiles, active_profile_id, next_profile_id):
    if (
        not isinstance(profiles, list)
        or not 1 <= len(profiles) <= MAX_USER_PROFILES
        or not isinstance(next_profile_id, int)
        or isinstance(next_profile_id, bool)
        or next_profile_id <= 0
        or not all(valid_profile(profile) for profile in profiles)
    ):
        return False
    identifiers = [profile["id"] for profile in profiles]
    names = [profile["name"].casefold() for profile in profiles]
    numbers = [_profile_number(profile_id) for profile_id in identifiers]
    return (
        len(set(identifiers)) == len(identifiers)
        and len(set(names)) == len(names)
        and active_profile_id in identifiers
        and next_profile_id > max(numbers)
    )


def profile_prompt_block(profile):
    if not valid_profile(profile):
        raise ValueError("Профиль имеет неверный формат.")
    return "\n\n".join((
        "[Профиль пользователя]",
        "Применяй эти предпочтения, если они не противоречат предыдущим инструкциям.",
        f"Имя: {profile['name']}",
        f"Стиль: {profile['style']}",
        f"Формат: {profile['format']}",
        f"Ограничения: {profile['constraints']}",
    ))
