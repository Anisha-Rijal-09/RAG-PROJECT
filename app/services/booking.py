from sqlalchemy.orm import Session
from openai import OpenAI
import json

from app.db.models import Booking


def extract_booking_details(text: str) -> dict:
    """Extract booking information from structured text."""

    details = {}

    parts = [part.strip() for part in text.split(",")]

    for part in parts:
        if ":" not in part:
            continue

        key, value = part.split(":", 1)
        details[key.strip().lower()] = value.strip()

    required_fields = ["name", "email", "date", "time"]

    missing = [
        field
        for field in required_fields
        if field not in details
    ]

    if missing:
        raise ValueError(
            f"Missing booking fields: {', '.join(missing)}"
        )

    return {
        "name": details["name"],
        "email": details["email"],
        "date": details["date"],
        "time": details["time"],
    }


def extract_booking_details_llm(
    client: OpenAI,
    text: str
) -> dict:
    """Use Groq LLM to extract booking details from natural language."""

    prompt = f"""
Extract booking information from the following user message.

Return ONLY valid JSON with these four fields:

name
email
date
time

If a field is not provided, use null.

User message:
{text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def save_booking(
    db: Session,
    booking_details: dict,
) -> Booking:
    """Save booking details to SQLite."""

    booking = Booking(
        name=booking_details["name"],
        email=booking_details["email"],
        date=booking_details["date"],
        time=booking_details["time"],
    )

    db.add(booking)
    db.commit()
    db.refresh(booking)

    return booking
