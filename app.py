import streamlit as st
import json
import re
from openai import OpenAI

st.set_page_config(page_title="Agent Debugger", page_icon="🔍", layout="wide")

st.title("🔍 Agent Debugger")
st.write("Paste any agent execution trace — raw JSON or line format — to get an instant debugging report.")

# ─────────────────────────────────────────
# API KEY FROM BACKEND
# ─────────────────────────────────────────
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("API key not configured. Please add OPENAI_API_KEY to Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)
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
            messages = []
            raw_steps = parsed["steps"]
            for s in raw_steps:
                step_type = s.get("type", "")
                status = s.get("status", "")
                duration_ms = s.get("duration_ms", None)

                if step_type == "reasoning":
                    messages.append({
                        "role": "assistant",
                        "content": s.get("thought", "") or str(s.get("output", "")),
                        "duration_ms": duration_ms,
                        "step_type": step_type
                    })
                elif step_type == "tool_call":
                    tool_name = s.get("tool", {}).get("name", "tool")
                    tool_input = s.get("input", {})
                    tool_output = s.get("output", {})
                    messages.append({
                        "role": "assistant",
                        "tool_call": f"{tool_name}({json.dumps(tool_input)})",
                        "duration_ms": duration_ms,
                        "step_type": step_type
                    })
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_output) if status == "success" else f"error: {json.dumps(tool_output)}",
                        "duration_ms": duration_ms,
                        "step_type": step_type
                    })
                elif step_type == "memory_lookup":
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(s.get("output", {})),
                        "duration_ms": duration_ms,
                        "step_type": step_type
                    })

            final = parsed.get("final_output", {})
            if final:
                messages.append({
                    "role": "assistant",
                    "content": final.get("response_summary", str(final)),
                    "duration_ms": None,
                    "step_type": "final"
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

        steps.append({
            "step": i + 1,
            "actor": actor,
            "content": content.strip(),
            "duration_ms": msg.get("duration_ms", None),
            "step_type": msg.get("step_type", None)
        })

    return steps


# ─────────────────────────────────────────
# SMART ENTRY POINT
# ─────────────────────────────────────────
def parse_trace(trace_input):
    trace_input = trace_input.strip()
    try:
        return parse_raw_json_trace(trace_input)
    except Exception:
        pass

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
        steps.append({
            "step": i + 1,
            "actor": actor,
            "content": content,
            "duration_ms": None,
            "step_type": None
        })
    return steps


# ─────────────────────────────────────────
# SEMANTIC CHECKER — only fires when
# keyword matching is uncertain
# ─────────────────────────────────────────
def semantic_check_tool_failure(content):
    """
    Called only when keyword matching is uncertain.
    Asks GPT-4o-mini a simple yes/no: did this tool call fail?
    Returns True if failed, False if success, None if can't determine.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Analyze this tool output and answer with ONLY 'FAILED' or 'SUCCESS'.
A tool has FAILED if: it returned an error, exception, empty result when data was expected,
wrong status, or any indication the operation did not complete correctly.
A tool has SUCCEEDED if: it returned valid data, a success status, or completed the operation.

Tool output: {content[:500]}

Answer (FAILED or SUCCESS only):"""
            }],
            max_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().upper()
        if "FAILED" in answer:
            return True
        elif "SUCCESS" in answer:
            return False
        return None
    except Exception:
        return None


# ─────────────────────────────────────────
# LATENCY DETECTION
# ─────────────────────────────────────────
def detect_latency_issues(steps):
    """
    Detects steps with abnormally high duration compared to
    the average for that step type. Flags bottlenecks.
    """
    latency_failures = []

    # Expected max durations by step type in ms
    EXPECTED_MAX_MS = {
        "tool_call": 3000,
        "reasoning": 2000,
        "memory_lookup": 1000,
        "final": 5000
    }

    # Collect durations per step type
    durations_by_type = {}
    for step in steps:
        duration = step.get("duration_ms")
        step_type = step.get("step_type")
        if duration and step_type:
            if step_type not in durations_by_type:
                durations_by_type[step_type] = []
            durations_by_type[step_type].append((step["step"], duration))

    # Flag steps that exceed 2x the average for their type
    # OR exceed the absolute expected max
    for step_type, entries in durations_by_type.items():
        if not entries:
            continue

        durations = [d for _, d in entries]
        avg_duration = sum(durations) / len(durations)
        expected_max = EXPECTED_MAX_MS.get(step_type, 3000)

        for step_num, duration in entries:
            is_outlier = len(durations) > 1 and duration > avg_duration * 2
            exceeds_expected = duration > expected_max

            if is_outlier or exceeds_expected:
                step_content = next(
                    (s["content"] for s in steps if s["step"] == step_num), ""
                )
                latency_failures.append({
                    "root_cause": "latency_bottleneck",
                    "failure_type": "performance_degradation",
                    "step": step_num,
                    "severity": "high" if duration > expected_max * 2 else "medium",
                    "description": f"Step took {duration}ms — significantly above expected {expected_max}ms for {step_type}",
                    "evidence": step_content[:200],
                    "duration_ms": duration,
                    "avg_duration_ms": round(avg_duration),
                    "expected_max_ms": expected_max
                })

    return latency_failures


# ─────────────────────────────────────────
# LAYER 1 — DETERMINISTIC + SEMANTIC
# ─────────────────────────────────────────
def detect_failures(steps):
    failures = []
    last_tool_error = None
    retry_count = 0
    last_scheduled_date = None
    last_scheduled_step = None

    # Clear error/success signals — high confidence, no LLM needed
    CLEAR_ERROR_WORDS = [
        "error", "failed", "invalid", "not found", "missing",
        "unavailable", "could not", "cannot", "unable",
        "no flight selected", "no event details", "no valid",
        "no previous", "no document", "no session", "no scheduled",
        "dropped", "none found", "exceeded",
        "cancelled", "canceled", "rejected", "denied", "expired",
        "returned", "refunded", "closed", "terminated", "suspended"
    ]

    # Ambiguous signals — uncertain, trigger semantic check
    AMBIGUOUS_SIGNALS = [
        "null", "none", "0", "empty", "[]", "{}", "n/a",
        "no results", "not available", "not set", "undefined",
        "false", "status: 0", "count: 0", "results_count\": 0"
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
            # Check for clear error signals first
            is_clear_error = any(word in content_lower for word in CLEAR_ERROR_WORDS)
            is_ambiguous = any(signal in content_lower for signal in AMBIGUOUS_SIGNALS)

            if is_clear_error:
                # High confidence — no LLM needed
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
            elif is_ambiguous:
                # Low confidence — ask LLM to verify
                semantic_result = semantic_check_tool_failure(content)
                if semantic_result is True:
                    last_tool_error = step
                else:
                    last_tool_error = None
            else:
                last_tool_error = None

            # Date detection
            date_match = re.search(r'scheduled_for[^0-9]*(\d{4}-\d{2}-\d{2})', content)
            if date_match:
                last_scheduled_date = date_match.group(1)
                last_scheduled_step = step

        elif actor == "agent":
            claims_success = any(word in content_lower for word in SUCCESS_CLAIMS)

            # Contradiction: success after tool error
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

            # Missing tool call
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

            # Date mismatch
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

            # Retry loop
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
    latency_count = failure_types.count("performance_degradation")

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
            "root_fix": "Enforce tool output verification before any user-facing confirmation."
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
            "root_fix": "Agent must verify tool scheduled_for date matches requested date before confirming to user."
        }
    if latency_count >= 1:
        return {
            "pattern": "latency_bottleneck",
            "label": "LATENCY BOTTLENECK PATTERN",
            "description": "One or more steps are taking significantly longer than expected, causing performance degradation.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Profile the slow step, check for API timeouts, oversized payloads, or missing caching. Add timeout limits and fallback logic."
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
def build_prompt(steps, failures, score, breakdown):
    MAX_CHARS = 12000
    failed_step_numbers = set(f["step"] for f in failures)

    if failed_step_numbers:
        relevant_steps = [s for s in steps if s["step"] in failed_step_numbers
                         or s["step"] in {n-1 for n in failed_step_numbers}
                         or s["step"] in {n+1 for n in failed_step_numbers}]
    else:
        relevant_steps = steps

    steps_str = json.dumps(relevant_steps)[:MAX_CHARS]
    failures_str = json.dumps(failures)[:MAX_CHARS]

    return f"""
You are an AI agent debugging engine.
Return valid JSON only. No markdown. No commentary.

Relevant Steps: {steps_str}
Detected Failures: {failures_str}

CRITICAL RULES:
- reliability_score is FIXED at {score}. Do not change it.
- score_breakdown is FIXED at {breakdown}. Do not change it.
- If tool output directly contradicts agent response, confirmed cause must state the contradiction explicitly.
- Hallucination severity is always critical when agent produces false user-facing output.
- Use exact quotes from steps as evidence.
- overall_confidence above 0.8 if contradiction is obvious.
- Quick fix must be implementable in under 1 hour.
- Robust fix must be a systemic architectural solution.
- Response must be valid JSON.

FAILURE DETECTION GUIDANCE:

1. MISSING TOOL CALL: Agent claims completion without calling required tool. severity=critical
2. CALCULATION ERROR: Agent math doesn't match tool output. severity=high
3. DATE MISINTERPRETATION: Tool scheduled different date than agent confirmed. severity=high
4. LATENCY BOTTLENECK: Step duration significantly exceeds expected. severity=high/medium
5. SEMANTIC TOOL FAILURE: Tool returned ambiguous output that indicates failure. severity=high

Schema:
{{
  "timeline": [{{"step": 1, "actor": "User | Agent | Tool", "event": "string", "evidence": "exact quote"}}],
  "failures": [{{
    "root_cause": "contradiction | permission_failure | logic_failure | missing_tool_call | latency_bottleneck | missing_context | unknown",
    "failure_type": "hallucination | tool_misuse | retry_loop | action_skipped | calculation_error | date_misinterpretation | performance_degradation | tool_schema_error | context_drop | unknown",
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
    if not trace_input:
        st.error("Please paste a trace.")
    else:
        with st.spinner("Analyzing trace..."):
            steps = parse_trace(trace_input)

            if not steps:
                st.error("Could not parse trace. Make sure it is valid JSON or line format (actor: content).")
            else:
                # Layer 1 — deterministic + semantic
                failures = detect_failures(steps)

                # Latency detection — pure Python, no LLM
                latency_failures = detect_latency_issues(steps)
                all_failures = failures + latency_failures

                pattern = detect_pattern(all_failures)
                score, breakdown = compute_score(all_failures, pattern)
                prompt = build_prompt(steps, all_failures, score, breakdown)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
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

                    # Adjust score for GPT-4o-mini findings Layer 1 missed
                    gpt_failures = parsed.get("failures", [])
                    layer1_types = [f.get("failure_type") for f in all_failures]
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

                    # ── LATENCY SUMMARY ──
                    if latency_failures:
                        st.subheader("⚡ Latency Bottlenecks Detected")
                        for lf in latency_failures:
                            st.warning(
                                f"Step {lf['step']} — {lf['duration_ms']}ms "
                                f"(expected max {lf['expected_max_ms']}ms) — {lf['description']}"
                            )

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
                    st.progress(float(confidence))
                    st.markdown(f"{confidence:.2f} / 1.0")

                except json.JSONDecodeError:
                    st.error("Failed to parse response. Raw output:")
                    st.markdown(raw)

st.divider()
st.caption("Agent Debugger | AI Agent Observability")
