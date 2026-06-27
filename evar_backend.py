"""
E-VAR Companion — FastAPI backend
==================================
Bridges the static frontend (localhost:3000) to IBM Granite.

Primary path:  IBM watsonx.ai (Granite running on IBM Cloud)
Fallback path: Ollama, running Granite locally (if watsonx isn't configured
               or a call fails — e.g. offline / rate-limited / no credits)

Contract expected by the frontend (index.html):
    GET /var-event/{event_id}?mode=fan|pro&language=EN|ES|FR|...
        -> {"ai_response": {"explanation": "..."}}
    GET /chat/{event_id}?q=...&mode=fan|pro&language=EN
        -> {"ai_response": {"explanation": "..."}}

Run:
    uvicorn evar_backend:app --port 8000 --reload

Author: Jessica · IBM SkillsBuild AI Builders Challenge 2026
Data: StatsBomb Open Data · FIFA World Cup 2022
"""

import os
from typing import Optional
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
def _get_secret(key, default=""):
    """Read from env vars first, then Streamlit secrets (for cloud deployment)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default

# ── watsonx.ai (primary) ──
WATSONX_API_KEY    = _get_secret("WATSONX_API_KEY")
WATSONX_PROJECT_ID = _get_secret("WATSONX_PROJECT_ID")
WATSONX_SPACE_ID   = _get_secret("WATSONX_SPACE_ID")
WATSONX_URL        = _get_secret("WATSONX_URL") or "https://us-south.ml.cloud.ibm.com"
WATSONX_MODEL_ID   = _get_secret("WATSONX_MODEL_ID") or "ibm/granite-4-h-small"

# ── Ollama (fallback, local) ──
OLLAMA_URL    = os.getenv("OLLAMA_URL", "")
GRANITE_MODEL = os.getenv("GRANITE_MODEL", "granite3.2:2b")

# ── LangFlow (optional) ──
LANGFLOW_URL   = os.getenv("LANGFLOW_URL", "")
LANGFLOW_TOKEN = os.getenv("LANGFLOW_TOKEN", "")

# Lazily-created watsonx model client (only built if a key is present).
_watsonx_model = None
_watsonx_init_error = None


def _get_watsonx_model():
    """Build (once) and cache the watsonx ModelInference client.
    Supports project_id OR space_id — whichever is set."""
    global _watsonx_model, _watsonx_init_error
    if _watsonx_model is not None or _watsonx_init_error is not None:
        return _watsonx_model
    if not WATSONX_API_KEY:
        _watsonx_init_error = "WATSONX_API_KEY not set"
        return None
    if not (WATSONX_PROJECT_ID or WATSONX_SPACE_ID):
        _watsonx_init_error = "Neither WATSONX_PROJECT_ID nor WATSONX_SPACE_ID is set"
        return None
    try:
        from ibm_watsonx_ai import Credentials, APIClient
        from ibm_watsonx_ai.foundation_models import ModelInference
        creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        # Use space_id if project_id isn't available (works identically)
        if WATSONX_PROJECT_ID:
            client = APIClient(creds, project_id=WATSONX_PROJECT_ID)
            print(f"[watsonx] using project_id: {WATSONX_PROJECT_ID[:8]}...")
        else:
            client = APIClient(creds, space_id=WATSONX_SPACE_ID)
            print(f"[watsonx] using space_id: {WATSONX_SPACE_ID[:8]}...")
        _watsonx_model = ModelInference(api_client=client, model_id=WATSONX_MODEL_ID)
        return _watsonx_model
    except Exception as e:
        _watsonx_init_error = str(e)
        print(f"[watsonx] init error: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────
# Real ML / StatsBomb facts for each VAR event (mirrors the ML object in the
# frontend so Granite is grounded in the same numbers the UI shows).
# ──────────────────────────────────────────────────────────────────────────
EVENTS = {
    "ARG_KSA_2022_E1": {
        "match": "Argentina vs Saudi Arabia (Group C, 22 Nov 2022)",
        "player": "Lionel Messi",
        "clock": "22'",
        "body_part": "Tracked offside point",
        "margin_cm": 3.0,
        "dist_from_goal_m": 5.1,
        "pass_length_m": 57.9,
        "zone": "Danger Zone (within 10m of goal)",
        "ml_danger_prob": 0.582,
        "danger_percentile": 94,
        "law_reference": "FIFA Law 11 — Offside",
        "outcome": "Goal disallowed for offside",
        "top_factors": "distance from goal (5.1m) and a long 57.9m through-ball",
        "source_note": "Public match reports confirm the goal was ruled out for offside; the exact body point and centimetre margin are demo model fields unless verified against official semi-automated offside output.",
    },
    "ARG_KSA_2022_E2": {
        "match": "Argentina vs Saudi Arabia (Group C, 22 Nov 2022)",
        "player": "Lautaro Martínez",
        "clock": "28'",
        "body_part": "Tracked offside point",
        "margin_cm": 3.2,
        "dist_from_goal_m": 20.1,
        "pass_length_m": 34.2,
        "zone": "Shooting Range (10–25m from goal)",
        "ml_danger_prob": 0.766,
        "danger_percentile": 66,
        "law_reference": "FIFA Law 11 — Offside",
        "outcome": "Goal disallowed for offside",
        "top_factors": "distance from goal (20.1m) and a central pitch position",
        "source_note": "Public match reports confirm the goal was ruled out for offside; the exact body point and centimetre margin are demo model fields unless verified against official semi-automated offside output.",
    },
}

# Language name map so we can instruct Granite to answer in the fan's language.
LANG_NAMES = {
    "EN": "English", "ES": "Spanish", "FR": "French", "PT": "Portuguese",
    "DE": "German", "AR": "Arabic", "IT": "Italian", "JA": "Japanese",
    "ZH": "Chinese", "RU": "Russian", "NL": "Dutch", "TR": "Turkish",
    "PL": "Polish", "KO": "Korean", "HI": "Hindi", "SV": "Swedish",
    "DA": "Danish", "NO": "Norwegian", "FI": "Finnish", "HU": "Hungarian",
    "RO": "Romanian", "HR": "Croatian", "CS": "Czech",
}


# ──────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────
def build_verdict_prompt(ev: dict, mode: str, language: str) -> str:
    lang = LANG_NAMES.get(language.upper(), "English")
    if mode == "pro":
        style = (
            "You are a professional football analyst. Answer in 3-4 flowing sentences. "
            "No bullet points. No bold text. No markdown. Respond directly to the question asked "
            "using the event data provided."
        )
    else:
        style = (
            "You are a friendly football commentator. Answer in 2-3 short clear sentences. "
            "Never start with Hi, Hey, Totally or Similar greetings. Respond directly to the question asked "
            "using the event data provided. Sound like a knowledgeable friend watching the match with the fan. "
            "Do not add sympathy, emotional reactions, or comfort phrases. "
            "Use simple words like 'just ahead', 'the goal had to come back', and 'that is why VAR stepped in'. "
            "Avoid stiff phrases such as 'violating Law 11', 'determination was made', 'in the context of', "
            "'aligning with standards', or 'StatsBomb' unless the fan specifically asks about data sources."
        )
    return (
        f"{style}\n\n"
        f"Answer ONLY in {lang}.\n\n"
        f"Facts about this VAR decision:\n"
        f"- Match: {ev['match']}\n"
        f"- Player ruled offside: {ev['player']} at {ev['clock']}\n"
        f"- Tracked body point: {ev['body_part']}\n"
        f"- Offside margin: {ev['margin_cm']}cm beyond the last defender\n"
        f"- Distance from goal: {ev['dist_from_goal_m']}m\n"
        f"- Pitch zone: {ev['zone']}\n"
        f"- Pass length: {ev['pass_length_m']}m\n"
        f"- Danger score: {round(ev['ml_danger_prob']*100)}%\n"
        f"- Confidence percentile: {ev['danger_percentile']}th percentile of all WC2022 offsides\n"
        f"- Law reference: {ev.get('law_reference', 'FIFA Law 11 — Offside')}\n"
        f"- Most important factors: {ev['top_factors']}\n"
        f"- Source note: {ev.get('source_note', 'Use only the provided verified facts')}\n"
        f"- Outcome: {ev['outcome']}\n\n"
        f"Explain why this was offside and how dangerous the chance was."
    )


def build_chat_prompt(ev: dict, question: str, mode: str, language: str) -> str:
    lang = LANG_NAMES.get(language.upper(), "English")
    if mode == "pro":
        tone = (
            "You are a professional football analyst. Answer in 3-4 flowing sentences. "
            "No bullet points. No bold text. No markdown. Respond directly to the question asked "
            "using the event data provided."
        )
    else:
        tone = (
            "You are a friendly football commentator. Answer in 2-3 short clear sentences. "
            "Never start with Hi, Hey, Totally or Similar greetings. Respond directly to the question asked "
            "using the event data provided. Sound like a knowledgeable friend watching the match with the fan. "
            "Do not add sympathy, emotional reactions, or comfort phrases. "
            "Use simple words like 'just ahead', 'the goal had to come back', and 'that is why VAR stepped in'. "
            "Avoid stiff phrases such as 'violating Law 11', 'determination was made', 'in the context of', "
            "'aligning with standards', or 'StatsBomb' unless the fan specifically asks about data sources."
        )
    return (
        f"{tone}\n\n"
        f"Answer ONLY in {lang}.\n\n"
        f"Context — the VAR decision being discussed:\n"
        f"- {ev['player']} ruled offside at {ev['clock']} in {ev['match']}\n"
        f"- Tracked body point: {ev['body_part']}\n"
        f"- Offside margin: {ev['margin_cm']}cm\n"
        f"- Distance from goal: {ev['dist_from_goal_m']}m\n"
        f"- Pitch zone: {ev['zone']}\n"
        f"- Pass length: {ev['pass_length_m']}m\n"
        f"- Danger score: {round(ev['ml_danger_prob']*100)}%\n"
        f"- Confidence percentile: {ev['danger_percentile']}th percentile\n"
        f"- Law reference: {ev.get('law_reference', 'FIFA Law 11 — Offside')}\n\n"
        f"Fan question: {question}\n\n"
        f"Answer the question using the context above."
    )


def build_selected_event(match: Optional[str] = None, player: Optional[str] = None,
                         minute: Optional[str] = None, decision: Optional[str] = None,
                         outcome: Optional[str] = None, zone: Optional[str] = None,
                         confidence: Optional[str] = None, law: Optional[str] = None,
                         source: Optional[str] = None) -> dict:
    return {
        "match": match or "Selected FIFA World Cup 2022 match",
        "player": player or "Selected player/event",
        "clock": minute or "selected minute",
        "zone": zone or "selected pitch zone",
        "decision": decision or "VAR decision",
        "outcome": outcome or "Selected VAR outcome",
        "confidence": confidence or "not shown",
        "law": law or "Relevant FIFA Law",
        "source": source or "selected History row",
    }


def build_selected_chat_prompt(ev: dict, question: str, mode: str, language: str) -> str:
    lang = LANG_NAMES.get(language.upper(), "English")
    if mode == "pro":
        tone = (
            "You are a professional football analyst. Answer in 3-4 flowing sentences. "
            "No bullet points. No bold text. No markdown. Respond directly to the question asked "
            "using the event data provided."
        )
    else:
        tone = (
            "You are a friendly football commentator. Answer in 2-3 short clear sentences. "
            "Never start with Hi, Hey, Totally or Similar greetings. Respond directly to the question asked "
            "using the event data provided. Sound like a knowledgeable friend watching the match with the fan. "
            "Do not add sympathy, emotional reactions, or comfort phrases. "
            "Use simple words like 'just ahead', 'the goal had to come back', and 'that is why VAR stepped in'. "
            "Avoid stiff phrases such as 'violating Law 11', 'determination was made', 'in the context of', "
            "'aligning with standards', or 'StatsBomb' unless the fan specifically asks about data sources."
        )
    return (
        f"{tone}\n\n"
        f"Answer ONLY in {lang}.\n\n"
        f"Selected context:\n"
        f"- Match: {ev['match']}\n"
        f"- Minute: {ev['clock']}\n"
        f"- Player/subject: {ev['player']}\n"
        f"- Decision: {ev['decision']}\n"
        f"- Outcome: {ev['outcome']}\n"
        f"- Zone: {ev['zone']}\n"
        f"- Confidence/danger shown: {ev['confidence']}\n"
        f"- Law context: {ev['law']}\n"
        f"- Source: {ev['source']}\n\n"
        f"Fan question: {question}\n\n"
        f"Answer only from the selected context above."
    )


# ──────────────────────────────────────────────────────────────────────────
# Model callers
# ──────────────────────────────────────────────────────────────────────────
async def call_watsonx(prompt: str) -> str | None:
    """Call IBM Granite on watsonx.ai. Returns text or None on failure."""
    model = _get_watsonx_model()
    if model is None:
        if _watsonx_init_error:
            print(f"[watsonx] skipped: {_watsonx_init_error}")
        return None
    try:
        messages = [{"role": "user", "content": prompt}]
        # generate_text is sync in the SDK; the prompts here are short so this
        # is fine for a demo backend without adding a thread pool.
        resp = model.chat(messages=messages)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[watsonx] call error: {e}")
        return None


async def call_langflow(prompt: str) -> str | None:
    """Call a deployed LangFlow flow. Returns text or None on failure."""
    if not LANGFLOW_URL:
        return None
    headers = {"Content-Type": "application/json"}
    if LANGFLOW_TOKEN:
        headers["Authorization"] = f"Bearer {LANGFLOW_TOKEN}"
    payload = {"input_value": prompt, "output_type": "chat", "input_type": "chat"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(LANGFLOW_URL, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            # LangFlow nests the text fairly deep; dig it out defensively.
            return _extract_langflow_text(data)
    except Exception as e:
        print(f"[LangFlow] error: {e}")
        return None


def _extract_langflow_text(data: dict) -> str | None:
    try:
        outputs = data["outputs"][0]["outputs"][0]
        # Common shapes across LangFlow versions:
        if "results" in outputs and "message" in outputs["results"]:
            return outputs["results"]["message"]["text"]
        if "artifacts" in outputs and "message" in outputs["artifacts"]:
            return outputs["artifacts"]["message"]
        if "messages" in outputs and outputs["messages"]:
            return outputs["messages"][0]["message"]
    except Exception:
        pass
    return None


async def call_ollama(prompt: str) -> str | None:
    """Call IBM Granite directly through Ollama. Returns text or None."""
    if not OLLAMA_URL:
        return None
    payload = {"model": GRANITE_MODEL, "prompt": prompt, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
            return r.json().get("response", "").strip()
    except Exception as e:
        print(f"[Ollama] error: {e}")
        return None


async def generate(prompt: str) -> tuple[str, str]:
    """
    Try watsonx.ai first (primary, IBM Cloud), then LangFlow (if configured),
    then Ollama (local fallback), then a graceful offline message.
    Returns (text, source) so the API can report which backend answered.
    """
    text = await call_watsonx(prompt)
    if text:
        return text.strip(), "watsonx.ai"
    text = await call_langflow(prompt)
    if text:
        return text.strip(), "langflow"
    text = await call_ollama(prompt)
    if text:
        return text.strip(), "ollama"
    return (
        "The AI explanation service is offline. Configure WATSONX_API_KEY / "
        "WATSONX_PROJECT_ID to see live Granite explanations — the "
        "on-screen stats are still real.",
        "offline",
    )


def polish_fan_answer(answer: str, question: str) -> str:
    """Keep fan mode short, friendly, and free of stiff or emotional phrasing."""
    if not answer:
        return answer
    text = answer.strip()
    lowered = text.lower()
    for greeting in ("hi, ", "hey, ", "totally, ", "totally — ", "totally - "):
        if lowered.startswith(greeting):
            text = text[len(greeting):].lstrip()
            lowered = text.lower()
            break
    replacements = {
        "violating Law 11": "being just ahead of the defensive line",
        "violating law 11": "being just ahead of the defensive line",
        "deemed offside": "called offside",
        "determination was made": "VAR made the call",
        "in the context of": "in",
        "against the law": "not allowed by the offside rule",
        "using StatsBomb data, ": "",
        "using StatsBomb data": "",
        "with StatsBomb data, ": "",
        "with StatsBomb data": "",
        "StatsBomb data": "the match data",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    empathy_phrases = (
        "I get why that hurts—it would have been a huge goal—but ",
        "I get why that hurts - it would have been a huge goal - but ",
        "I get why that hurts. ",
        "It is a brutal one, but ",
        "It's a brutal one, but ",
        "That would have been a huge goal, but ",
    )
    for phrase in empathy_phrases:
        text = text.replace(phrase, "")
    return text


# ──────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(title="E-VAR Companion Backend", version="1.0")

# Allow the static frontend (any localhost port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "E-VAR Companion backend",
        "status": "ok",
        "primary_model": WATSONX_MODEL_ID,
        "watsonx_configured": bool(WATSONX_API_KEY and (WATSONX_PROJECT_ID or WATSONX_SPACE_ID)),
        "fallback_model": GRANITE_MODEL,
        "langflow": bool(LANGFLOW_URL),
        "events": list(EVENTS.keys()),
    }


@app.get("/health")
async def health():
    """Check model layers in priority order: watsonx → LangFlow → Ollama."""
    configured = bool(WATSONX_API_KEY and (WATSONX_PROJECT_ID or WATSONX_SPACE_ID))
    wx = await call_watsonx("Reply with the single word: ok")
    result = {
        "watsonx_configured": configured,
        "watsonx_reachable": wx is not None,
        "watsonx_model": WATSONX_MODEL_ID,
    }
    if _watsonx_init_error:
        result["watsonx_note"] = _watsonx_init_error
    # Only check lower layers if watsonx didn't work
    if not wx:
        if LANGFLOW_URL:
            lf = await call_langflow("Reply with the single word: ok")
            result["langflow_configured"] = True
            result["langflow_reachable"] = lf is not None
            if lf:
                return result  # LangFlow working — no need to check Ollama
        ol = await call_ollama("Reply with the single word: ok")
        result["ollama_reachable"] = ol is not None
        result["ollama_model"] = GRANITE_MODEL
    return result


@app.get("/var-event/{event_id}")
async def var_event(event_id: str, mode: str = "fan", language: str = "EN"):
    ev = EVENTS.get(event_id)
    if not ev:
        return {"ai_response": {"explanation": f"Unknown event '{event_id}'."}}
    prompt = build_verdict_prompt(ev, mode, language)
    explanation, source = await generate(prompt)
    return {
        "ai_response": {"explanation": explanation},
        "event_id": event_id,
        "mode": mode,
        "language": language,
        "source": source,
    }


@app.get("/chat/{event_id}")
async def chat(event_id: str, q: str = "", mode: str = "fan", language: str = "EN",
               match: Optional[str] = None, player: Optional[str] = None,
               minute: Optional[str] = None, decision: Optional[str] = None,
               outcome: Optional[str] = None, zone: Optional[str] = None,
               confidence: Optional[str] = None, law: Optional[str] = None,
               source: Optional[str] = None):
    ev = EVENTS.get(event_id)
    if not q.strip():
        return {"ai_response": {"explanation": "Ask a question about the VAR decision."}}
    if ev:
        prompt = build_chat_prompt(ev, q, mode, language)
    else:
        ev = build_selected_event(match, player, minute, decision, outcome, zone, confidence, law, source)
        prompt = build_selected_chat_prompt(ev, q, mode, language)
    explanation, source = await generate(prompt)
    if mode == "fan" and source != "offline":
        explanation = polish_fan_answer(explanation, q)
    return {
        "ai_response": {"explanation": explanation},
        "event_id": event_id,
        "mode": mode,
        "language": language,
        "source": source,
    }
