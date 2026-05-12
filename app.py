import streamlit as st
import json
import re
from openai import OpenAI

st.set_page_config(page_title="Agent Debugger", page_icon="🔍", layout="wide")

st.title("🔍 Agent Debugger")
st.write("Paste any agent execution trace — raw JSON or line format — to get an instant debugging report.")

api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
trace_input = st.text_area("Paste agent trace here", height=250)


# ─────────────────────────────────────────
# RAW JSON TRACE PARSER
# ─────────────────────────────────────────
def parse_raw_json_trace(raw_input):
    if isinstance(raw_input, str):
        raw_input = raw_input.strip()
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            match = re.search(r'(\{.*\}|\[.*\])', raw_input, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
            else:
                raise ValueError("Could not parse input as JSON")
    else:
        parsed = raw_input

    if isinstance(parsed, dict):
        if "trace" in parsed:
            messages = parsed["trace"]
        elif "steps" in parsed:
        # Structured observability format
            messages = []
            for s in parsed["steps"]:
                step_type = s.get("type", "")
                status = s.get("status", "")
                if step_type == "reasoning":
                    messages.append({
                    "role": "assistant",
                    "content": s.get("thought", "") or str(s.get("output", ""))
                    })
                elif step_type == "tool_call":
                    tool_name = s.get("tool", {}).get("name", "tool")
                    tool_input = s.get("input", {})
                    tool_output = s.get("output", {})
                    messages.append({
                        "role": "assistant",
                        "tool_call": f"{tool_name}({json.dumps(tool_input)})"
                    })
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_output) if status == "success" else f"error: {json.dumps(tool_output)}"
                    })
            elif step_type == "memory_lookup":
                messages.append({
                    "role": "tool",
                    "content": json.dumps(s.get("output", {}))
                })
        # Add final output as agent message
        final = parsed.get("final_output", {})
        if final:
            messages.append({
                "role": "assistant",
                "content": final.get("response_summary", str(final))
            })
    else:
        messages = [parsed]
    elif isinstance(parsed, list):
        messages = parsed
    else:
        raise ValueError("Unexpected JSON structure")

    ROLE_MAP = {
        "user": "user", "human": "user",
        "assistant": "agent", "ai": "agent",
        "tool": "tool", "function": "tool",
        "system": None
    }

    steps = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "").lower()
        actor = ROLE_MAP.get(role)
        if actor is None:
            continue

        content_raw = msg.get("content", "")
        tool_call = msg.get("tool_call", "")

        if tool_call and actor == "agent":
            content = f"[TOOL CALL] {tool_call}"
            if content_raw:
                content += " | " + str(content_raw)
        elif isinstance(content_raw, list):
            content = " ".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in content_raw
            )
        elif isinstance(content_raw, dict):
            content = json.dumps(content_raw)
        else:
            content = str(content_raw)

        steps.append({"step": i + 1, "actor": actor, "content": content.strip()})

    return steps


# ─────────────────────────────────────────
# SMART ENTRY POINT — tries JSON first
# ─────────────────────────────────────────
def parse_trace(trace_input):
    trace_input = trace_input.strip()
    try:
        return parse_raw_json_trace(trace_input)
    except Exception:
        pass

    # Line-by-line fallback
    steps = []
    for i, line in enumerate(trace_input.split("\n")):
        line = line.strip()
        if not line or ":" not in line:
            continue
        actor, content = line.split(":", 1)
        actor = actor.strip().lower()
        content = content.strip()
        if actor not in ["user", "agent", "tool"]:
            continue
        steps.append({"step": i + 1, "actor": actor, "content": content})
    return steps


# ─────────────────────────────────────────
# LAYER 1 — DETERMINISTIC FAILURE DETECTION
# ─────────────────────────────────────────
def detect_failures(steps):
    failures = []
    last_tool_error = None
    retry_count = 0
    last_scheduled_date = None
    last_scheduled_step = None

    TOOL_ERROR_WORDS = [
        "error", "failed", "invalid", "not found", "missing",
        "unavailable", "could not", "cannot", "unable",
        "no flight selected", "no event details", "no valid",
        "no previous", "no document", "no session", "no scheduled",
        "dropped", "none found", "exceeded",
        "cancelled", "canceled", "rejected", "denied", "expired",
        "returned", "refunded", "closed", "terminated", "suspended"
    ]
    SUCCESS_CLAIMS = [
        "successfully", "confirmed", "completed", "done",
        "booked", "processed", "sent", "added",
        "taken care of", "you will receive", "should receive",
        "check your inbox", "i found", "i've added",
        "no duplicates", "cleanup complete", "last session",
        "as we discussed", "as requested", "market trends",
        "revenue growth", "here is your", "your report",
        "scheduled", "set", "all set", "i've scheduled",
        "on its way", "will be delivered", "out for delivery",
        "in transit", "is active", "is valid", "is available",
        "is confirmed", "is complete", "is ready"
    ]
    RETRY_WORDS = ["retry", "retrying", "attempting again", "trying again"]
    PERMISSION_WORDS = [
        "unauthorized", "access denied", "insufficient permissions", "permission denied"
    ]
    BOOKING_CLAIMS = ["booked", "reserved", "confirmed", "purchased", "ordered"]

    for step in steps:
        actor = step["actor"]
        content = step["content"]
        content_lower = content.lower()

        if actor == "tool":
            is_tool_error = any(word in content_lower for word in TOOL_ERROR_WORDS)
            if is_tool_error:
                last_tool_error = step
                if any(word in content_lower for word in PERMISSION_WORDS):
                    failures.append({
                        "root_cause": "permission_failure",
                        "failure_type": "tool_misuse",
                        "step": step["step"],
                        "severity": "high",
                        "description": "Tool returned a permission or authorization failure",
                        "evidence": content
                    })
            else:
                last_tool_error = None

            date_match = re.search(r'scheduled_for[^0-9]*(\d{4}-\d{2}-\d{2})', content)
            if date_match:
                last_scheduled_date = date_match.group(1)
                last_scheduled_step = step

        elif actor == "agent":
            claims_success = any(word in content_lower for word in SUCCESS_CLAIMS)

            if last_tool_error and claims_success:
                failures.append({
                    "root_cause": "contradiction",
                    "failure_type": "hallucination",
                    "step": step["step"],
                    "severity": "critical",
                    "description": "Agent claimed success after a tool failure",
                    "evidence": content,
                    "contradicted_by": last_tool_error["content"]
                })
                failures.append({
                    "root_cause": "contradiction",
                    "failure_type": "tool_misuse",
                    "step": step["step"],
                    "severity": "high",
                    "description": "Agent ignored tool failure and proceeded anyway",
                    "evidence": content,
                    "contradicted_by": last_tool_error["content"]
                })

            if any(word in content_lower for word in BOOKING_CLAIMS):
                booking_tool_found = any(
                    s["actor"] == "tool" and s["step"] < step["step"]
                    for s in steps
                    if any(w in s["content"].lower() for w in ["book", "reserv", "confirm", "purchas"])
                )
                if not booking_tool_found:
                    failures.append({
                        "root_cause": "missing_tool_call",
                        "failure_type": "action_skipped",
                        "step": step["step"],
                        "severity": "critical",
                        "description": "Agent claimed to complete a booking/action without calling the required tool",
                        "evidence": content
                    })

            if last_scheduled_date:
                mentioned_dates = re.findall(r'\b(\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b', content)
                if mentioned_dates:
                    failures.append({
                        "root_cause": "logic_failure",
                        "failure_type": "date_misinterpretation",
                        "step": step["step"],
                        "severity": "high",
                        "description": "Tool scheduled a different date than agent confirmed to user",
                        "evidence": content,
                        "contradicted_by": f"Tool scheduled: {last_scheduled_date}"
                    })
                    last_scheduled_date = None

            is_retry = any(word in content_lower for word in RETRY_WORDS)
            if is_retry:
                retry_count += 1
                if retry_count >= 2:
                    failures.append({
                        "root_cause": "logic_failure",
                        "failure_type": "retry_loop",
                        "step": step["step"],
                        "severity": "medium",
                        "description": "Agent appears to be retrying repeatedly",
                        "evidence": content
                    })
            else:
                retry_count = 0

    return failures


# ─────────────────────────────────────────
# PATTERN DETECTION
# ─────────────────────────────────────────
def detect_pattern(failures):
    failure_types = [f["failure_type"] for f in failures]
    steps_affected = sorted(set([f["step"] for f in failures]))
    hallucination_count = failure_types.count("hallucination")
    tool_misuse_count = failure_types.count("tool_misuse")
    retry_count = failure_types.count("retry_loop")
    action_skipped_count = failure_types.count("action_skipped")
    date_misinterp_count = failure_types.count("date_misinterpretation")

    if hallucination_count >= 3 and tool_misuse_count >= 3:
        return {
            "pattern": "cascading_failure",
            "label": "CASCADING FAILURE PATTERN",
            "description": "Agent has no error checking after tool calls. Same failure repeating across multiple steps.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Add a hard rule: if any tool returns an error, the agent must stop, explain the failure, and either retry with corrected input or ask the user for missing information."
        }
    if retry_count >= 2:
        return {
            "pattern": "retry_loop",
            "label": "INFINITE RETRY LOOP PATTERN",
            "description": "Agent is retrying the same failed action repeatedly without a stopping condition.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Implement a maximum retry limit with exponential backoff. After N retries, agent must stop and inform the user."
        }
    if hallucination_count >= 2:
        return {
            "pattern": "repeated_hallucination",
            "label": "REPEATED HALLUCINATION PATTERN",
            "description": "Agent is repeatedly fabricating successful outcomes without tool verification.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Enforce tool output verification before any user-facing confirmation. Agent must never confirm success without explicit tool success signal."
        }
    if action_skipped_count >= 1:
        return {
            "pattern": "missing_tool_call",
            "label": "MISSING MANDATORY TOOL CALL PATTERN",
            "description": "Agent skipped a required tool call and fabricated the outcome directly.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Enforce a tool-call gate: agent must receive explicit tool confirmation before reporting any action as complete to the user."
        }
    if date_misinterp_count >= 1:
        return {
            "pattern": "date_misinterpretation",
            "label": "DATE MISINTERPRETATION PATTERN",
            "description": "Tool scheduled a different date than what agent confirmed to user.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Agent must verify tool's scheduled_for date matches requested date before confirming to user."
        }
    return None


# ─────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────
def compute_score(failures, pattern):
    score = 100
    SEVERITY_PENALTIES = {"critical": 30, "high": 20, "medium": 15, "low": 5}
    REPEAT_PENALTY = 5
    breakdown = {
        "hallucination_penalty": 0,
        "tool_misuse_penalty": 0,
        "logic_error_penalty": 0,
        "missing_context_penalty": 0,
        "repeat_penalty": 0
    }
    seen_failure_types = {}
    for failure in failures:
        failure_type = failure.get("failure_type", "unknown")
        severity = failure.get("severity", "medium")
        if failure_type not in seen_failure_types:
            penalty = SEVERITY_PENALTIES.get(severity, 15)
            seen_failure_types[failure_type] = True
            key = f"{failure_type}_penalty"
            if key in breakdown:
                breakdown[key] += penalty
            else:
                breakdown["repeat_penalty"] += penalty
        else:
            penalty = REPEAT_PENALTY
            breakdown["repeat_penalty"] += penalty
        score -= penalty
    return max(score, 10), breakdown


# ─────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────
def build_prompt(steps, failures, trace, score, breakdown):
    return f"""
You are an AI agent debugging engine.
Return valid JSON only. No markdown. No commentary.

Parsed Steps: {steps}
Detected Failures: {failures}
Original Trace: {trace}

CRITICAL RULES:
- reliability_score is FIXED at {score}. Do not change it.
- score_breakdown is FIXED at {breakdown}. Do not change it.
- If tool output directly contradicts agent response, confirmed cause must state the contradiction explicitly.
- Hallucination severity is always critical when agent produces false user-facing output.
- Tool misuse severity is always critical when agent ignores a tool failure causing false output.
- Use exact quotes from trace as evidence.
- overall_confidence above 0.8 if contradiction is obvious.
- Quick fix must be implementable in under 1 hour.
- Robust fix must be a systemic architectural solution.
- Response must be valid JSON.

FAILURE DETECTION GUIDANCE:

1. MISSING TOOL CALL (action_skipped):
   Example: Agent says "I've booked Capsule Inn" but no book_hotel() tool was called.
   Diagnosis: root_cause=missing_tool_call, failure_type=action_skipped, severity=critical

2. CALCULATION ERROR:
   Example: Tool returns price=150, tax=0.1. Agent says total is $155 (correct is $165).
   Diagnosis: root_cause=logic_failure, failure_type=calculation_error, severity=high

3. DATE MISINTERPRETATION:
   Example: Tool returns scheduled_for=2024-12-05 but agent confirms May 12th to user.
   Diagnosis: root_cause=logic_failure, failure_type=date_misinterpretation, severity=high

Schema:
{{
  "timeline": [{{"step": 1, "actor": "User | Agent | Tool", "event": "string", "evidence": "exact quote"}}],
  "failures": [{{
    "root_cause": "contradiction | permission_failure | logic_failure | missing_tool_call | missing_context | unknown",
    "failure_type": "hallucination | tool_misuse | retry_loop | action_skipped | calculation_error | date_misinterpretation | tool_schema_error | context_drop | unknown",
    "failure_point": {{"step": "number", "description": "string", "evidence": "exact quote"}},
    "impact": "string",
    "likely_cause": {{"confirmed": "string", "hypothesis": "string or unknown"}},
    "suggested_fix": {{"quick": "string", "robust": "string"}},
    "severity": "critical | high | medium | low"
  }}],
  "reliability_score": {score},
  "score_breakdown": {breakdown},
  "debugging_signals": ["string"],
  "overall_confidence": 0.0
}}
"""


# ─────────────────────────────────────────
# MAIN ANALYZE BUTTON
# ─────────────────────────────────────────
if st.button("Analyze Trace", type="primary"):
    if not api_key or not trace_input:
        st.error("Please add your API key and paste a trace.")
    else:
        with st.spinner("Analyzing trace..."):
            steps = parse_trace(trace_input)

            if not steps:
                st.error("Could not parse trace. Make sure it is valid JSON or line format (actor: content).")
            else:
                failures = detect_failures(steps)
                pattern = detect_pattern(failures)
                score, breakdown = compute_score(failures, pattern)
                prompt = build_prompt(steps, failures, trace_input, score, breakdown)

                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )

                raw = response.choices[0].message.content.strip()
                if raw.startswith("```json"):
                    raw = raw.replace("```json", "").replace("```", "").strip()
                elif raw.startswith("```"):
                    raw = raw.replace("```", "").strip()
                raw = re.sub(r':\s*unknown\b', ': "unknown"', raw)

                try:
                    parsed = json.loads(raw)

                    # Adjust score for failures GPT-4o caught that Layer 1 missed
                    gpt_failures = parsed.get("failures", [])
                    layer1_types = [f.get("failure_type") for f in failures]
                    gpt_only_failures = [f for f in gpt_failures if f.get("failure_type") not in layer1_types]
                    for gf in gpt_only_failures:
                        severity = gf.get("severity", "medium")
                        penalty = {"critical": 30, "high": 20, "medium": 15, "low": 5}.get(severity, 15)
                        score = max(score - penalty, 10)

                    # ── SCORE ──
                    st.subheader("📊 Reliability Score")
                    if score < 40:
                        st.error(f"Score: {score}/100 🔴 CRITICAL")
                    elif score < 80:
                        st.warning(f"Score: {score}/100 🟡 WARNING")
                    else:
                        st.success(f"Score: {score}/100 🟢 OK")

                    # ── PATTERN ──
                    if pattern:
                        st.subheader("🚨 Core Failure Pattern")
                        st.error(f"**{pattern['label']}**")
                        st.markdown(f"**Description:** {pattern['description']}")
                        st.markdown(f"**Affected steps:** {pattern['affected_steps']}")
                        st.markdown(f"**Root fix:** {pattern['root_fix']}")

                    # ── FAILURES ──
                    st.subheader("📍 Failures Detected")
                    if not parsed.get("failures"):
                        st.success("No failures detected.")
                    for f in parsed.get("failures", []):
                        fp = f.get("failure_point", {})
                        severity = f.get("severity", "").upper()
                        ftype = f.get("failure_type", "").upper()
                        color = "🔴" if severity == "CRITICAL" else "🟡" if severity == "HIGH" else "🔵"
                        st.markdown(f"{color} **{ftype}** — Step {fp.get('step','?')} — {severity}")
                        st.markdown(f"*Evidence:* {fp.get('evidence','')}")
                        st.markdown(f"*Confirmed cause:* {f.get('likely_cause',{}).get('confirmed','')}")
                        fix = f.get("suggested_fix", {})
                        st.markdown(f"⚡ **Quick fix:** {fix.get('quick','')}")
                        st.markdown(f"🏗️ **Robust fix:** {fix.get('robust','')}")
                        st.divider()

                    # ── DEBUGGING SIGNALS ──
                    signals = parsed.get("debugging_signals", [])
                    if signals:
                        st.subheader("🔍 Debugging Signals")
                        for signal in signals:
                            st.markdown(f"- {signal}")

                    # ── CONFIDENCE ──
                    confidence = parsed.get("overall_confidence", 0.0)
                    st.subheader("📈 Overall Confidence")
                    st.progress(confidence)
                    st.markdown(f"{confidence:.2f} / 1.0")

                except json.JSONDecodeError:
                    st.error("Failed to parse GPT-4o response. Raw output:")
                    st.markdown(raw)

st.divider()
st.caption("Agent Debugger | AI Agent Observability")
