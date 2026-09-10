from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import OpenAI
import os

from app.db.session import get_db
from app.services.memory import get_history, save_history
from app.services.embeddings import embed_chunks
from app.services.vectorstore import search_qdrant
from app.services.booking import extract_booking_details_llm, save_booking


router = APIRouter(prefix="/chat", tags=["chat"])

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not loaded. Check your .env file.")

client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")


class ChatRequest(BaseModel):
    session_id: str
    message: str


@router.post("")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    history = get_history(request.session_id)

    query_embedding = embed_chunks([request.message])[0]
    results = search_qdrant(query_embedding)
    context = "\n\n".join(results)

    booking_keywords = ["book", "booking", "reserve", "reservation", "appointment", "schedule"]
    is_booking_request = any(k in request.message.lower() for k in booking_keywords)

    booking_status = None
    if is_booking_request:
        details = extract_booking_details_llm(client, request.message)
        if details and all(details.get(f) for f in ["name", "email", "date", "time"]):
            save_booking(db, details)
            booking_status = f"Booking confirmed: {details}"
        elif details:
            missing = [f for f in ["name", "email", "date", "time"] if not details.get(f)]
            booking_status = f"Missing info needed to complete booking: {', '.join(missing)}"

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    prompt = f"""You are a helpful RAG assistant.

Use the provided document context to answer the user's question.
If the answer is not present in the context, say that you do not
have enough information from the uploaded documents.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

BOOKING STATUS:
{booking_status or "N/A"}

USER MESSAGE:
{request.message}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
        )
        reply = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GROQ request failed: {str(e)}")

    save_history(request.session_id, "user", request.message)
    save_history(request.session_id, "assistant", reply)

    return {"session_id": request.session_id, "reply": reply, "booking_status": booking_status}




   
