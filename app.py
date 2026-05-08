import streamlit as st
import json
import re
from openai import OpenAI

st.set_page_config(page_title="Agent Debugger", page_icon="🔍")
st.title("🔍 Agent Debugger")
st.write("Paste any agent execution trace to get an instant debugging report.")

api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
trace_input = st.text_area("Paste agent trace here", height=250)

def parse_trace(trace):
    steps = []
    for i, line in enumerate(trace.strip().split("\n")):
        line = line.strip()
        if not line or ":" not in line:
            continue
        actor, content = line.split(":", 1)
        actor = actor.strip().lower()
        content = content.strip()
        if actor not in ["user", "agent", "tool"]:
            continue
        steps.append({"step": i+1, "actor": actor, "content": content})
    return steps

def detect_failures(steps):
    failures = []
    last_tool_error = None
    retry_count = 0
    TOOL_ERROR_WORDS = [
        "error", "failed", "invalid", "not found", "missing",
        "unavailable", "could not", "cannot", "unable",
        "no flight selected", "no event details", "no valid",
        "no previous", "no document", "no session", "no scheduled",
        "dropped", "none found", "exceeded"
    ]
    SUCCESS_CLAIMS = [
        "successfully", "confirmed", "completed", "done",
        "booked", "processed", "sent", "added",
        "taken care of", "you will receive", "should receive",
        "check your inbox", "i found", "i've added",
        "no duplicates", "cleanup complete", "last session",
        "as we discussed", "as requested", "market trends",
        "revenue growth", "here is your", "your report"
    ]
    RETRY_WORDS = ["retry", "retrying", "attempting again", "trying again"]
    PERMISSION_WORDS = [
        "unauthorized", "access denied", "insufficient permissions",
        "permission denied"
    ]
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

def detect_pattern(failures):
    failure_types = [f["failure_type"] for f in failures]
    steps_affected = sorted(set([f["step"] for f in failures]))
    hallucination_count = failure_types.count("hallucination")
    tool_misuse_count = failure_types.count("tool_misuse")
    retry_count = failure_types.count("retry_loop")
    if hallucination_count >= 3 and tool_misuse_count >= 3:
        return {
            "pattern": "cascading_failure",
            "label": "CASCADING FAILURE PATTERN",
            "description": "Agent has no error checking after tool calls.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Add a hard rule: if any tool returns an error, the agent must stop, explain the failure, and either retry with corrected input or ask the user for missing information."
        }
    if retry_count >= 2:
        return {
            "pattern": "retry_loop",
            "label": "INFINITE RETRY LOOP PATTERN",
            "description": "Agent is retrying the same failed action repeatedly.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Implement a maximum retry limit with exponential backoff."
        }
    if hallucination_count >= 2:
        return {
            "pattern": "repeated_hallucination",
            "label": "REPEATED HALLUCINATION PATTERN",
            "description": "Agent is repeatedly fabricating successful outcomes.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Enforce tool output verification before any user-facing confirmation."
        }
    return None

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
- If tool output directly contradicts agent response confirmed cause must state contradiction explicitly.
- Hallucination severity is always critical.
- Tool misuse severity is always critical when agent ignores tool failure causing false output.
- Use exact quotes from trace as evidence.
- overall_confidence above 0.8 if contradiction is obvious.
- Quick fix must be implementable in under 1 hour.
- Robust fix must be a systemic architectural solution.
- Response must be valid JSON.
Schema:
{{
  "timeline": [{{"step": 1, "actor": "User | Agent | Tool", "event": "string", "evidence": "exact quote"}}],
  "failures": [{{
    "root_cause": "contradiction | permission_failure | logic_failure | missing_context | unknown",
    "failure_type": "hallucination | tool_misuse | retry_loop | tool_schema_error | context_drop | unknown",
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

if st.button("Analyze Trace", type="primary"):
    if not api_key or not trace_input:
        st.error("Please add your API key and paste a trace.")
    else:
        with st.spinner("Analyzing trace..."):
            steps = parse_trace(trace_input)
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

                st.subheader("📊 Reliability Score")
                score_val = parsed.get("reliability_score", score)
                if score_val < 30:
                    st.error(f"Score: {score_val}/100 🔴 CRITICAL")
                elif score_val < 60:
                    st.warning(f"Score: {score_val}/100 🟡 WARNING")
                else:
                    st.success(f"Score: {score_val}/100 🟢 OK")

                if pattern:
                    st.subheader("🚨 Core Failure Pattern")
                    st.error(pattern["label"])
                    st.markdown(f"**Affected steps:** {pattern['affected_steps']}")
                    st.markdown(f"**Root fix:** {pattern['root_fix']}")

                st.subheader("📍 Failures Detected")
                for f in parsed.get("failures", []):
                    fp = f.get("failure_point", {})
                    st.markdown(f"**{f.get('failure_type','').upper()}** — Step {fp.get('step','?')} — {f.get('severity','').upper()}")
                    st.markdown(f"Evidence: _{fp.get('evidence','')}_")
                    st.markdown(f"Confirmed cause: {f.get('likely_cause',{}).get('confirmed','')}")
                    fix = f.get("suggested_fix", {})
                    st.markdown(f"Quick fix: {fix.get('quick','')}")
                    st.markdown(f"Robust fix: {fix.get('robust','')}")
                    st.divider()

                st.subheader("🔍 Debugging Signals")
                for signal in parsed.get("debugging_signals", []):
                    st.markdown(f"- {signal}")

                st.subheader("📈 Confidence")
                st.markdown(str(parsed.get("overall_confidence", 0.0)))

            except json.JSONDecodeError:
                st.markdown(raw)

st.divider()
st.caption("Built by Dhruva | AI Agent Observability")
