import asyncio
import base64
import logging
import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("netra")

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_LANGUAGES = {"Hindi", "English", "Bengali"}

app = FastAPI(
    title="NETRA API",
    description="Image-description endpoint for the NETRA accessibility prototype.",
    version="0.1.0",
)

# Exact origin only; set FRONTEND_ORIGIN to the final HTTPS frontend URL.
origins = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

client = genai.Client(api_key=API_KEY) if API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NETRA API",
        "vision_configured": bool(client),
        "openai_chat_configured": bool(openai_client),
    }


def make_prompt(question: str, language: str) -> str:
    return f"""
You are NETRA, an accessibility-focused visual description assistant.
Respond in {language}. The user's question is: {question}

Instructions:
- Answer in concise, natural spoken language. Use 2-6 short sentences when appropriate.
- Prioritize the user's question, not a long generic image caption.
- Describe visible objects and their relative positions only when reasonably clear.
- Distinguish observation from uncertainty. Say when a detail cannot be determined.
- Do not invent distances, object identity, door direction, road conditions, traffic-light states, or whether a path is safe.
- A single still image cannot establish that an area is currently clear or safe to walk through.
- Do not identify people by face, infer sensitive traits, or guess a person's identity.
- Do not claim the image provides live navigation.
- If the user asks about a safety-critical detail, explain the limitation and recommend checking with a trusted human or established mobility aid.
- Return only the answer the user should hear.
""".strip()


def call_gemini(image_bytes: bytes, mime_type: str, question: str, language: str) -> str:
    if client is None:
        raise RuntimeError("AI provider is not configured")

    response = client.interactions.create(
        model=MODEL,
        input=[
            {"type": "text", "text": make_prompt(question, language)},
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
                "mime_type": mime_type,
            },
        ],
    )
    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError("AI returned an empty answer")
    return text[:5000]


@app.get("/")
def root():
    return {"service": "NETRA API", "health": "/health", "docs": "/docs"}


@app.post("/api/describe")
async def describe(
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP image up to 8 MB")],
    question: Annotated[str, Form()] = "Describe my surroundings.",
    language: Annotated[str, Form()] = "Hindi",
):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="AI is not configured yet. Check the backend environment variables.",
        )

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Use a JPEG, PNG, or WebP image.")

    language = language.strip()
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Choose Hindi, English, or Bengali.")

    question = question.strip()[:500]
    if not question:
        question = "Describe my surroundings."

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller.")

    # The image is held in memory for this request only; this app does not write it to disk.
    try:
        description = await asyncio.to_thread(
            call_gemini, image_bytes, image.content_type, question, language
        )
        return {"description": description, "language": language, "mode": "image_description"}
    except Exception as exc:
        # Avoid logging image bytes, prompts, API keys, or provider error details.
        logger.warning("Vision request failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="The visual AI service is temporarily unavailable. Check the API key, model access, quota, and try again.",
        ) from None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=12)
    detected_objects: list[str] = Field(default_factory=list, max_length=12)


def call_openai_chat(message: str, history: list[dict], detected_objects: list[str]) -> str:
    if openai_client is None:
        raise RuntimeError("OpenAI is not configured")
    safe_history = []
    for item in history[-8:]:
        role = item.get("role") if isinstance(item, dict) else None
        content = item.get("content") if isinstance(item, dict) else None
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            safe_history.append({"role": role, "content": content[:2000]})
    objects = [str(item)[:80] for item in detected_objects[:12] if str(item).strip()]
    context = "Currently detected common-object labels (may be wrong): " + (", ".join(objects) if objects else "none available")
    safe_history.append({"role": "user", "content": f"{message.strip()}\n\nNETRA context: {context}"})
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are NETRA, a friendly accessibility assistant. Answer in the language the user uses, "
            "including Hindi or Hinglish when appropriate. Be concise, clear, and supportive. You may use "
            "the provided detected-object labels as tentative context, never as certainty. Do not claim "
            "that the prototype can guarantee safe navigation, identify a person, or detect every hazard. "
            "For road-crossing, medical, or other safety-critical questions, clearly explain limitations "
            "and encourage checking with a trusted person or established assistive aid. Never invent live "
            "location or map details; ask the user to use the map feature for location searches."
        ),
        input=safe_history,
        max_output_tokens=450,
    )
    answer = (response.output_text or "").strip()
    if not answer:
        raise RuntimeError("OpenAI returned an empty answer")
    return answer[:5000]


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if openai_client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI chat is not configured. Add OPENAI_API_KEY to the backend environment variables.",
        )
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        answer = await asyncio.to_thread(
            call_openai_chat, message, request.history, request.detected_objects
        )
        return {"answer": answer, "model": OPENAI_MODEL}
    except Exception as exc:
        logger.warning("OpenAI chat request failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="NETRA AI chat is temporarily unavailable. Check the OpenAI API key, model access, quota, and try again.",
        ) from None
