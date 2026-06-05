"""
api.py — FastAPI endpoint for Agent Debugger.
Accepts a trace, returns full diagnostic JSON.
Used by nano-vm-mcp integration and any external caller.
"""

import json
import re
import os
import secrets

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from openai import OpenAI

from core import (
    detect_failures,
    detect_context_drops,
)

app = FastAPI(title="Agent Debugger API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
API_KEY = os.environ.get("AGENT_DEBUGGER_API_KEY", "")
security = HTTPBearer()


# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────

class TraceRequest(BaseModel):
    trace: dict


class DiagnosticResponse(BaseModel):
    score: int
    pattern: str | None
    failures: list
    confidence: float
    debugging_signals: list
    trace_id: str | None = None


# ─────────────────────────────────────────
# AUTH CHECK
# ─────────────────────────────────────────

def verify_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if not secrets.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials


# ─────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────

def compute_score(failures):
    score = 100
    SEVERITY_PENALTIES = {"critical": 30, "high": 20, "medium": 15, "low": 5}
    seen = {}
    for f in failures:
        ftype = f.get("failure_type", "unknown")
        severity = f.get("severity", "medium")
        if ftype not in seen:
            score -= SEVERITY_PENALTIES.get(severity, 15)
            seen[ftype] = True
        else:
            score -= 5
    return max(score, 10)


# ─────────────────────────────────────────
# PATTERN DETECTION
# ─────────────────────────────────────────

def detect_pattern(failures):
    types = [f["failure_type"] for f in failures]
    steps = sorted(set(f["step"] for f in failures))

    if types.count("hallucination") >= 1:
        return {
            "pattern": "hallucination",
            "label": "HALLUCINATION PATTERN",
            "affected_steps": steps,
        }
    if types.count("retry_loop") >= 2:
        return {
            "pattern": "retry_loop",
            "label": "INFINITE RETRY LOOP PATTERN",
            "affected_steps": steps,
        }
    if types.count("action_skipped") >= 1:
        return {
            "pattern": "missing_tool_call",
            "label": "MISSING MANDATORY TOOL CALL PATTERN",
            "affected_steps": steps,
        }
    return None


# ─────────────────────────────────────────
# GPT ANALYSIS
# ─────────────────────────────────────────

def run_gpt_analysis(steps, failures, score):
    MAX_CHARS = 8000
    failed_steps = set(f["step"] for f in failures)
    relevant = [
        s for s in steps
        if s["step"] in failed_steps
        or s["step"] in {n - 1 for n in failed_steps}
        or s["step"] in {n + 1 for n in failed_steps}
    ]

    clean_steps = [
        {"step": s["step"], "actor": s["actor"], "content": s["content"]}
        for s in relevant
    ]

    prompt = f"""
You are an AI agent debugging engine.
Return valid JSON only. No markdown. No commentary.

Steps: {json.dumps(clean_steps)[:MAX_CHARS]}
Detected Failures: {json.dumps(failures)[:MAX_CHARS]}

For every failure provide:
- TRIGGER: exact step and what it returned
- PROPAGATION: how that caused the bad outcome
- PREVENTION: specific code-level fix

Schema:
{{
  "failures": [{{
    "failure_type": "string",
    "severity": "critical | high | medium | low",
    "failure_point": {{"step": 0, "evidence": "string"}},
    "likely_cause": {{
      "confirmed": "TRIGGER + PROPAGATION explanation",
      "hypothesis": "string or unknown"
    }},
    "suggested_fix": {{
      "quick": "specific code-level action",
      "robust": "architectural solution"
    }}
  }}],
  "debugging_signals": ["string"],
  "overall_confidence": 0.0,
  "reliability_score": {score}
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()
    raw = re.sub(r':\s*unknown\b', ': "unknown"', raw)
    return json.loads(raw)


# ─────────────────────────────────────────
# NANO-VM TRACE PARSER
# ─────────────────────────────────────────

def parse_nano_vm_trace(trace: dict):
    trace_id = trace.get("trace_id", "unknown")
    status = trace.get("status", "UNKNOWN")
    steps_raw = trace.get("steps", [])

    steps = []
    for i, step_data in enumerate(steps_raw):
        step_type = step_data.get("type", "unknown")
        output = step_data.get("output", "")
        error = step_data.get("error", None)

        if step_type == "llm":
            actor = "agent"
            content = str(output)
        elif step_type == "tool":
            actor = "tool"
            content = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
            if error:
                content = f"error: {error}\n\n{content}"
        elif step_type == "condition":
            actor = "system"
            content = f"Condition: {step_data.get('condition_expr')} → {step_data.get('result')}"
        else:
            actor = "system"
            content = str(output)

        steps.append({
            "step": i + 1,
            "actor": actor,
            "content": content,
            "duration_ms": step_data.get("duration_ms", 0),
            "step_type": step_type,
            "step_hash": None,
            "status": step_data.get("status", "SUCCESS"),
            "step_id": step_data.get("step_id", f"step_{i}"),
            "error": error,
        })

    return steps, trace_id, status


# ─────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────

@app.post("/analyze", response_model=DiagnosticResponse)
async def analyze_trace(
    request: TraceRequest,
    credentials: HTTPAuthorizationCredentials = Security(verify_key),
):
    try:
        steps, trace_id, status = parse_nano_vm_trace(request.trace)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse trace: {str(e)}")

    if not steps:
        raise HTTPException(status_code=400, detail="No steps found in trace")

    failures = detect_failures(steps)
    context_failures = detect_context_drops(steps)
    all_failures = failures + context_failures

    score = compute_score(all_failures)
    pattern = detect_pattern(all_failures)

    try:
        gpt_result = run_gpt_analysis(steps, all_failures, score)
        confidence = gpt_result.get("overall_confidence", 0.0)
        signals = gpt_result.get("debugging_signals", [])
        gpt_failures = gpt_result.get("failures", [])
    except Exception:
        confidence = 0.0
        signals = []
        gpt_failures = []

    return DiagnosticResponse(
        score=score,
        pattern=pattern["label"] if pattern else None,
        failures=gpt_failures if gpt_failures else all_failures,
        confidence=confidence,
        debugging_signals=signals,
        trace_id=trace_id,
    )


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-debugger-api"}
