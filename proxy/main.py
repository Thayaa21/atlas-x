"""
FIG Proxy Server — wraps OpenAI API with team key auth + daily token budget.
Deploy this to Render. Teammates never see your real OpenAI key.
"""

import os
import httpx
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FIG OpenAI Proxy")

# ── Config (set these as Render environment variables) ──────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]        # your real key (Render secret)
TEAM_KEY       = os.environ["TEAM_KEY"]              # e.g. "fig-team-2025"
DAILY_TOKEN_BUDGET = int(os.getenv("DAILY_TOKEN_BUDGET", "100000"))  # default 100k/day

OPENAI_BASE = "https://api.openai.com"

# ── In-memory budget tracker (resets on server restart / new day) ────────────
budget = {
    "date": datetime.now(timezone.utc).date().isoformat(),
    "tokens_used": 0,
}

def get_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def check_and_reserve(estimated_tokens: int = 0) -> dict:
    """Check budget and reset daily if needed."""
    today = get_today()
    if budget["date"] != today:
        logger.info(f"New day {today} — resetting budget. Yesterday used: {budget['tokens_used']}")
        budget["date"] = today
        budget["tokens_used"] = 0

    remaining = DAILY_TOKEN_BUDGET - budget["tokens_used"]
    return {"remaining": remaining, "used": budget["tokens_used"], "limit": DAILY_TOKEN_BUDGET}

def record_usage(tokens: int):
    budget["tokens_used"] += tokens
    logger.info(f"Tokens used today: {budget['tokens_used']} / {DAILY_TOKEN_BUDGET}")

# ── Auth helper ──────────────────────────────────────────────────────────────
def verify_team_key(authorization: str | None):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    # Accept "Bearer fig-team-2025" or just "fig-team-2025"
    token = authorization.replace("Bearer ", "").strip()
    if token != TEAM_KEY:
        raise HTTPException(status_code=403, detail="Invalid team key")

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
async def health():
    b = check_and_reserve()
    return {
        "status": "ok",
        "service": "FIG OpenAI Proxy",
        "budget": b,
    }

@app.get("/budget")
async def budget_status(authorization: str = Header(default=None)):
    verify_team_key(authorization)
    b = check_and_reserve()
    return {
        "date": budget["date"],
        "tokens_used": b["used"],
        "daily_limit": b["limit"],
        "tokens_remaining": b["remaining"],
        "percent_used": round(b["used"] / b["limit"] * 100, 1),
    }

# ── Main proxy — forwards any /v1/* request to OpenAI ────────────────────────
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy(path: str, request: Request, authorization: str = Header(default=None)):
    verify_team_key(authorization)

    # Check budget before forwarding
    b = check_and_reserve()
    if b["remaining"] <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Daily token budget exhausted ({DAILY_TOKEN_BUDGET} tokens). Resets tomorrow UTC."
        )

    # Read body
    body = await request.body()
    body_json = {}
    if body:
        try:
            body_json = json.loads(body)
        except Exception:
            pass

    # Forward to OpenAI with real key
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if "OpenAI-Organization" in request.headers:
        headers["OpenAI-Organization"] = request.headers["OpenAI-Organization"]

    target_url = f"{OPENAI_BASE}/v1/{path}"
    
    # Handle streaming
    stream = body_json.get("stream", False)

    async with httpx.AsyncClient(timeout=120.0) as client:
        if stream:
            async def stream_response():
                total_tokens = 0
                async with client.stream(
                    request.method, target_url,
                    headers=headers, content=body
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                        # rough token accounting for streamed responses
                        total_tokens += len(chunk) // 4
                record_usage(total_tokens)

            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            resp = await client.request(
                request.method, target_url,
                headers=headers, content=body
            )
            # Record actual token usage from response
            try:
                resp_json = resp.json()
                usage = resp_json.get("usage", {})
                tokens = usage.get("total_tokens", 0)
                if tokens:
                    record_usage(tokens)
                    logger.info(f"Request to /v1/{path} used {tokens} tokens")
            except Exception:
                pass

            return JSONResponse(content=resp.json(), status_code=resp.status_code)