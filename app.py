import streamlit as st
import json
import re
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple

st.set_page_config(page_title="Agent Debugger", page_icon="🔍", layout="wide")

st.title("🔍 Agent Debugger")
st.write("Paste any agent execution trace — nano-vm, raw JSON, or line format — to get an instant debugging report.")

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
# NANO-VM TRACE PARSER (v0.7.5)
# ─────────────────────────────────────────
def parse_nano_vm_trace(raw_input: str):
    if isinstance(raw_input, str):
        raw_input = raw_input.strip()
        try:
            parsed_json = json.loads(raw_input)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON")

        if not isinstance(parsed_json, dict):
            raise ValueError("Root must be a JSON object")
        if "trace_id" not in parsed_json or "steps" not in parsed_json:
            raise ValueError("Missing 'trace_id' or 'steps': not a nano-vm trace")

        trace_obj = parsed_json
    else:
        raise ValueError("Input must be a JSON string")

    trace_id = trace_obj.get("trace_id", "unknown")
    status = trace_obj.get("status", "UNKNOWN")
    final_output = trace_obj.get("final_output", None)
    steps_raw = trace_obj.get("steps", [])
    snapshots_raw = trace_obj.get("state_snapshots", [])

    # Build snapshot map for O(1) lookup
    snapshot_map = {}
    for item in snapshots_raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            idx, hsh = item
            snapshot_map[int(idx)] = str(hsh)
        elif isinstance(item, dict):
            if "step" in item and "hash" in item:
                snapshot_map[int(item["step"])] = str(item["hash"])

    steps = []
    for i, step_data in enumerate(steps_raw):
        step_type = step_data.get("type", "unknown")
        duration_ms = step_data.get("duration_ms", 0)
        output = step_data.get("output", "")
        error = step_data.get("error", None)
        status_step = step_data.get("status", "SUCCESS")

        if step_type == "llm":
            actor = "agent"
            content = str(output)
        elif step_type == "tool":
            actor = "tool"
            if isinstance(output, (dict, list)):
                content = json.dumps(output, indent=2)
            else:
                content = str(output)
            if error:
                content = f"error: {error}\n\n{content}"
        elif step_type == "condition":
            actor = "system"
            cond_expr = step_data.get("condition_expr", "N/A")
            result = step_data.get("result", "N/A")
            content = f"Condition: {cond_expr} → Result: {result}"
        elif step_type == "parallel":
            actor = "system"
            content = f"Parallel Block ({len(step_data.get('parallel_steps', []))} sub-steps)"
        else:
            actor = "system"
            content = str(output)

        step_hash = snapshot_map.get(i, None)

        steps.append({
            "step": i + 1,
            "actor": actor,
            "content": content,
            "duration_ms": duration_ms,
            "step_type": step_type,
            "step_hash": step_hash,
            "error": error,
            "status": status_step,
            "step_id": step_data.get("step_id", f"step_{i}")
        })

    metrics = {
        "total_tokens": trace_obj.get("total_tokens", 0),
        "total_cost_usd": trace_obj.get("total_cost_usd", 0.0),
        "vm_version": "0.7.5",
        "trace_id": trace_id,
        "status": status,
        "final_output": final_output,
        "_nano_vm_snapshots": snapshot_map,
        "_is_nano_vm": True
    }

    return steps, metrics


# ─────────────────────────────────────────
# RAW JSON TRACE PARSER (unchanged)
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

    # Extract metrics if present at top level
    metrics = None
    top_level_errors = []
    if isinstance(parsed, dict):
        metrics = parsed.get("metrics", None)
        top_level_errors = parsed.get("errors", [])

    if isinstance(parsed, dict):
        if "trace" in parsed:
            messages = parsed["trace"]
        elif "steps" in parsed:
            messages = []
            for s in parsed["steps"]:
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
                        "duration_ms": None,
                        "step_type": step_type
                    })
                elif step_type == "memory_lookup":
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(s.get("output", {})),
                        "duration_ms": duration_ms,
                        "step_type": step_type
                    })
                else:
                    output = s.get("output", {})
                    if status == "failure":
                        failure_reason = output.get("failure_reason", str(output))
                        messages.append({
                            "role": "tool",
                            "content": f"error: {failure_reason}",
                            "duration_ms": duration_ms,
                            "step_type": step_type
                        })
                    elif status == "warning":
                        messages.append({
                            "role": "tool",
                            "content": f"warning: {json.dumps(output)}",
                            "duration_ms": duration_ms,
                            "step_type": step_type
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "content": json.dumps(output),
                            "duration_ms": duration_ms,
                            "step_type": step_type
                        })

            for err in top_level_errors:
                messages.append({
                    "role": "tool",
                    "content": f"error: [{err.get('code', 'ERROR')}] {err.get('message', '')}",
                    "duration_ms": None,
                    "step_type": "system_error",
                    "severity": err.get("severity", "high")
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
            "step_type": msg.get("step_type", None),
            "step_hash": None,
            "status": "success"
        })

    return steps, metrics


# ─────────────────────────────────────────
# SMART ENTRY POINT — nano-vm first
# ─────────────────────────────────────────
def parse_trace(trace_input):
    trace_input = trace_input.strip()

    # 1. Try nano-vm parser first
    try:
        if '"trace_id"' in trace_input and '"steps"' in trace_input and '"vm_version"' not in trace_input:
            # Additional check: nano-vm traces have simple step types (llm, tool, condition, parallel)
            if any(t in trace_input for t in ['"type": "llm"', '"type": "tool"', '"type": "condition"', '"type": "parallel"']):
                steps, metrics = parse_nano_vm_trace(trace_input)
                return steps, metrics
    except Exception:
        pass

    # 2. Try legacy JSON parser
    try:
        return parse_raw_json_trace(trace_input)
    except Exception:
        pass

    # 3. Line-by-line fallback with multiline JSON support (unchanged)
    steps = []
    lines = trace_input.split("\n")
    i = 0
    step_num = 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        first_colon = line.index(":")
        actor = line[:first_colon].strip().lower()
        content = line[first_colon+1:].strip()
        if actor not in ["user", "agent", "tool"]:
            i += 1
            continue
        if content.startswith("{"):
            json_lines = [content]
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                json_lines.append(next_line)
                if next_line == "}":
                    break
                i += 1
            content = " ".join(json_lines)
        steps.append({
            "step": step_num,
            "actor": actor,
            "content": content,
            "duration_ms": None,
            "step_type": None,
            "step_hash": None,
            "status": "success"
        })
        step_num += 1
        i += 1
    return steps, None


# ─────────────────────────────────────────
# SEMANTIC CHECKER (unchanged)
# ─────────────────────────────────────────
def semantic_check_tool_failure(content):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Analyze this tool output and answer with ONLY 'FAILED' or 'SUCCESS'.
FAILED: error, exception, empty result when data expected, wrong status, operation incomplete.
SUCCESS: valid data returned, success status, operation completed correctly.

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
# NUMERICAL MISMATCH DETECTION (unchanged)
# ─────────────────────────────────────────
def extract_numbers(text):
    return re.findall(r'-?\d+\.?\d*', text)


def detect_numerical_mismatch(tool_content, agent_content, step_num):
    if tool_content.strip().startswith("error:"):
        return None

    tool_numbers = extract_numbers(tool_content)
    agent_numbers = extract_numbers(agent_content)

    if not tool_numbers or not agent_numbers:
        return None

    for tool_num in tool_numbers:
        tool_val = float(tool_num)
        if abs(tool_val) < 10:
            continue
        agent_vals = [float(n) for n in agent_numbers if n]
        agent_mentions_similar = any(
            abs(av - tool_val) < tool_val * 0.5
            for av in agent_vals
            if abs(av) > 10
        )
        if not agent_mentions_similar:
            continue
        exact_match = any(abs(av - tool_val) < 0.01 for av in agent_vals)
        if not exact_match:
            mismatched_val = next(
                (av for av in agent_vals if abs(av - tool_val) < tool_val * 0.5),
                None
            )
            if mismatched_val:
                return {
                    "root_cause": "data_distortion",
                    "failure_type": "numerical_mismatch",
                    "step": step_num,
                    "severity": "critical",
                    "description": f"Agent reported {mismatched_val} but tool returned {tool_num}.",
                    "evidence": agent_content[:300],
                    "contradicted_by": f"Tool returned: {tool_num}"
                }
    return None


# ─────────────────────────────────────────
# CONTEXT DROP DETECTION (unchanged)
# ─────────────────────────────────────────
def detect_context_drops(steps):
    context_failures = []

    user_entities = set()
    for step in steps:
        if step["actor"] == "user":
            words = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', step["content"])
            user_entities.update(words)

    agent_claims = []
    for step in steps:
        if step["actor"] == "agent":
            for prev_step, prev_claim in agent_claims:
                prev_numbers = extract_numbers(prev_claim)
                curr_numbers = extract_numbers(step["content"])

                for pn in prev_numbers:
                    pval = float(pn)
                    if abs(pval) < 2:
                        continue
                    for cn in curr_numbers:
                        cval = float(cn)
                        if abs(cval) > 2 and abs(cval - pval) > 0.01:
                            prev_lower = prev_claim.lower()
                            curr_lower = step["content"].lower()
                            shared_words = set(prev_lower.split()) & set(curr_lower.split())
                            common_words = {"the", "a", "an", "is", "are", "was", "and", "or", "for", "to", "in", "of", "your", "i", "it"}
                            meaningful_shared = shared_words - common_words
                            if len(meaningful_shared) >= 2:
                                context_failures.append({
                                    "root_cause": "context_drop",
                                    "failure_type": "self_contradiction",
                                    "step": step["step"],
                                    "severity": "critical",
                                    "description": f"Agent contradicted its own earlier statement. Previously stated {pn}, now states {cn}.",
                                    "evidence": step["content"][:300],
                                    "contradicted_by": f"Step {prev_step}: {prev_claim[:200]}"
                                })

            agent_claims.append((step["step"], step["content"]))

    return context_failures


# ─────────────────────────────────────────
# LATENCY DETECTION — nano-vm aware
# ─────────────────────────────────────────
def detect_latency_issues(steps):
    latency_failures = []

    # Detect if this is a nano-vm trace
    is_nano_vm_trace = any(s.get("step_hash") is not None for s in steps)

    EXPECTED_MAX_MS = {
        "tool": 2000,
        "llm": 1500,
        "condition": 50,
        "parallel": 5000,
        "tool_call": 2000,
        "reasoning": 1500,
        "memory_lookup": 500,
        "final": 3000
    }

    durations_by_type = {}
    for step in steps:
        duration = step.get("duration_ms")
        step_type = step.get("step_type")
        if not duration or duration == 0:
            continue

        lookup_type = step_type
        if step_type in ["llm", "reasoning"]:
            lookup_type = "llm"
        elif step_type in ["tool", "tool_call"]:
            lookup_type = "tool"

        if lookup_type not in durations_by_type:
            durations_by_type[lookup_type] = []
        durations_by_type[lookup_type].append((step["step"], duration))

    for step_type, entries in durations_by_type.items():
        if not entries:
            continue

        durations = [d for _, d in entries]
        avg_duration = sum(durations) / len(durations)
        expected_max = EXPECTED_MAX_MS.get(step_type, 10000)

        for step_num, duration in entries:
            # Tighter threshold for nano-vm (1.3x) since VM overhead is near zero
            multiplier = 1.3 if is_nano_vm_trace else 1.5
            is_outlier = len(durations) > 1 and duration > avg_duration * multiplier
            exceeds_expected = duration > expected_max

            if is_outlier or exceeds_expected:
                step_content = next(
                    (s["content"] for s in steps if s["step"] == step_num), ""
                )
                latency_failures.append({
                    "root_cause": "latency_bottleneck",
                    "failure_type": "performance_degradation",
                    "step": step_num,
                    "severity": "high" if duration > expected_max * 1.5 else "medium",
                    "description": f"Step took {duration}ms — above expected {expected_max}ms for {step_type}",
                    "evidence": step_content[:200],
                    "duration_ms": duration,
                    "avg_duration_ms": round(avg_duration),
                    "expected_max_ms": expected_max
                })

    return latency_failures


# ─────────────────────────────────────────
# METRICS EXTRACTION — nano-vm aware
# ─────────────────────────────────────────
def extract_metrics_insights(metrics):
    if not metrics:
        return None

    insights = []

    # nano-vm metrics format
    if metrics.get("_is_nano_vm"):
        cost = metrics.get("total_cost_usd", 0)
        tokens = metrics.get("total_tokens", 0)
        if cost:
            cost_1k = round(cost * 1000, 2)
            cost_10k = round(cost * 10000, 2)
            insights.append(f"💰 Cost per query: USD {cost:.4f} → USD {cost_1k} per 1K queries → USD {cost_10k} per 10K queries")
            if cost > 0.05:
                insights.append("⚠️ High cost per query — consider prompt compression or caching")
        if tokens:
            insights.append(f"📊 Total tokens: {tokens}")
        return insights

    # Standard metrics format (unchanged)
    cost = metrics.get("estimated_cost_usd", None)
    tokens_in = metrics.get("total_tokens_input", None)
    tokens_out = metrics.get("total_tokens_output", None)
    tool_calls = metrics.get("tool_calls", None)

    if cost:
        cost_1k = round(cost * 1000, 2)
        cost_10k = round(cost * 10000, 2)
        insights.append(f"💰 Cost per query: USD {cost:.4f} → USD {cost_1k} per 1K queries → USD {cost_10k} per 10K queries")
        if cost > 0.05:
            insights.append(f"⚠️ High cost per query — consider prompt compression or caching repeated tool calls")

    if tokens_in and tokens_out:
        ratio = round(tokens_in / tokens_out, 1) if tokens_out > 0 else 0
        insights.append(f"📊 Token usage: {tokens_in} input / {tokens_out} output (ratio {ratio}:1)")
        if tokens_in > 2000:
            insights.append(f"⚠️ Large input context ({tokens_in} tokens) — consider summarizing earlier steps to reduce cost")

    if tool_calls is not None:
        insights.append(f"🔧 Tool calls made: {tool_calls}")

    return insights


# ─────────────────────────────────────────
# LAYER 1 — DETERMINISTIC + SEMANTIC (unchanged)
# ─────────────────────────────────────────
def detect_failures(steps):
    failures = []
    last_tool_error = None
    last_tool_content = None
    last_tool_error_step = 0
    retry_count = 0
    last_scheduled_date = None

    CLEAR_ERROR_WORDS = [
        "error", "failed", "invalid", "not found", "missing",
        "unavailable", "could not", "cannot", "unable",
        "no flight selected", "no event details", "no valid",
        "no previous", "no document", "no session", "no scheduled",
        "dropped", "None found", "exceeded",
        "cancelled", "canceled", "rejected", "denied", "expired",
        "returned", "refunded", "closed", "terminated", "suspended"
    ]

    AMBIGUOUS_SIGNALS = [
        "null", "None", "0", "empty", "[]", "{}", "n/a",
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
            is_clear_error = any(word in content_lower for word in CLEAR_ERROR_WORDS)
            is_ambiguous = any(signal in content_lower for signal in AMBIGUOUS_SIGNALS)

            if is_clear_error:
                last_tool_error = step
                last_tool_content = content
                last_tool_error_step = step["step"]
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
                semantic_result = semantic_check_tool_failure(content)
                if semantic_result is True:
                    last_tool_error = step
                    last_tool_content = content
                else:
                    last_tool_error = None
                    last_tool_content = content
            else:
                last_tool_error = None
                last_tool_content = content

            date_match = re.search(r'scheduled_for[^0-9]*(\d{4}-\d{2}-\d{2})', content)
            if date_match:
                last_scheduled_date = date_match.group(1)
            date_match_alt = re.search(r'scheduled_for[^0-9]*(\d{2}[/-]\d{2}[/-]\d{4})', content)
            if date_match_alt and not last_scheduled_date:
                last_scheduled_date = date_match_alt.group(1)

        elif actor == "agent":
            claims_success = any(word in content_lower for word in SUCCESS_CLAIMS)
            RETRY_SUCCESS_CLAIMS = [
                "retry succeeded", "retried successfully",
                "retry was successful", "attempt succeeded",
                "succeeded after retry", "resolved after retry"
            ]
            is_hallucinated_retry = any(
                claim in content_lower for claim in RETRY_SUCCESS_CLAIMS
            )

            if last_tool_error and claims_success:
                if not is_hallucinated_retry:
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

            if last_tool_content and step.get("step_type") not in ["system_error", "final"]:
                num_mismatch = detect_numerical_mismatch(
                    last_tool_content, content, step["step"]
                )
                if num_mismatch:
                    failures.append(num_mismatch)

            if any(word in content_lower for word in BOOKING_CLAIMS) and step.get("step_type") in [None, "final"]:
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
                RETRY_SUCCESS_CLAIMS = [
                    "retry succeeded", "retried successfully",
                    "retry was successful", "attempt succeeded",
                    "succeeded after retry", "resolved after retry"
                ]
                if is_hallucinated_retry:
                    retry_tool_found = any(
                        s["actor"] == "tool"
                        and s["step"] > last_tool_error_step
                        and s["step"] < step["step"]
                        for s in steps
                    )
                    if not retry_tool_found:
                        failures.append({
                            "root_cause": "contradiction",
                            "failure_type": "hallucinated_retry",
                            "step": step["step"],
                            "severity": "critical",
                            "description": "Agent claimed retry succeeded but no retry tool call exists after the error",
                            "evidence": content,
                            "contradicted_by": last_tool_error["content"] if last_tool_error else "No retry tool call found"
                        })
            else:
                retry_count = 0

    for step in steps:
        if step.get("step_type") == "system_error":
            failures.append({
                "root_cause": "system_error",
                "failure_type": "critical_system_failure",
                "step": step["step"],
                "severity": "critical",
                "description": "System reported a critical error in the errors block",
                "evidence": step["content"]
            })

    for step in steps:
        content = step.get("content", "")
        if content.lower().startswith("warning:"):
            failures.append({
                "root_cause": "agent_warning",
                "failure_type": "risk_flag",
                "step": step["step"],
                "severity": "medium",
                "description": "Step reported warning status with risk flags",
                "evidence": content[:300]
            })

    return failures


# ─────────────────────────────────────────
# PATTERN DETECTION (unchanged)
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
    numerical_count = failure_types.count("numerical_mismatch")
    context_drop_count = failure_types.count("self_contradiction")

    if hallucination_count >= 2 and tool_misuse_count >= 2:
        return {
            "pattern": "cascading_failure",
            "label": "CASCADING FAILURE PATTERN",
            "description": "Agent has no error checking after tool calls. Failure repeating across multiple steps.",
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
    if hallucination_count >= 1 and numerical_count >= 1:
        return {
            "pattern": "data_distortion",
            "label": "DATA DISTORTION PATTERN",
            "description": "Agent is misreporting tool data to the user — wrong numbers, flipped values, or fabricated figures.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Enforce strict output binding — agent must pass tool values directly to user without transformation unless explicitly computing a derived value."
        }
    if context_drop_count >= 1:
        return {
            "pattern": "context_drop",
            "label": "CONTEXT DROP PATTERN",
            "description": "Agent contradicted its own earlier statement or referred to information not present in the conversation.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Implement conversation state tracking. Agent must verify claims against earlier steps before responding."
        }
    if hallucination_count >= 1:
        return {
            "pattern": "hallucination",
            "label": "HALLUCINATION PATTERN",
            "description": "Agent fabricated a successful outcome without tool verification.",
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
            "root_fix": "Enforce a tool-call gate: agent must receive explicit tool confirmation before reporting any action as complete."
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
            "description": "One or more steps are taking significantly longer than expected.",
            "affected_steps": steps_affected,
            "unique_failure_points": len(steps_affected),
            "total_instances": len(failures),
            "root_fix": "Profile the slow step, check for API timeouts, oversized payloads, or missing caching. Add timeout limits and fallback logic."
        }
    return None


# ─────────────────────────────────────────
# SCORING (unchanged)
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
# PROMPT BUILDER (unchanged)
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

    # Clean steps for prompt — remove nano-vm specific fields GPT doesn't need
    clean_steps = [{
        "step": s["step"],
        "actor": s["actor"],
        "content": s["content"],
        "duration_ms": s.get("duration_ms")
    } for s in relevant_steps]

    steps_str = json.dumps(clean_steps)[:MAX_CHARS]
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
- Hallucination severity is always critical.
- Numerical mismatch severity is always critical — even small differences matter at scale.
- Use exact quotes from steps as evidence.
- overall_confidence above 0.8 if contradiction is obvious.
- Quick fix implementable in under 1 hour.
- Robust fix is a systemic architectural solution.
- Response must be valid JSON.

FAILURE DETECTION GUIDANCE:
1. MISSING TOOL CALL: Agent claims completion without calling required tool. severity=critical
2. CALCULATION/NUMERICAL ERROR: Any number agent states differs from tool output. severity=critical
3. DATE MISINTERPRETATION: Tool scheduled different date than agent confirmed. severity=high
4. LATENCY BOTTLENECK: Step duration significantly exceeds expected. severity=high/medium
5. CONTEXT DROP: Agent contradicts itself or references information not in conversation. severity=critical
6. DATA DISTORTION: Agent reports wrong values from tool output. severity=critical

Schema:
{{
  "timeline": [{{"step": 1, "actor": "User | Agent | Tool", "event": "string", "evidence": "exact quote"}}],
  "failures": [{{
    "root_cause": "contradiction | permission_failure | logic_failure | missing_tool_call | latency_bottleneck | context_drop | data_distortion | missing_context | unknown",
    "failure_type": "hallucination | tool_misuse | retry_loop | action_skipped | calculation_error | numerical_mismatch | date_misinterpretation | performance_degradation | self_contradiction | context_drop | tool_schema_error | unknown",
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
# FULL TRACE VISUALIZATION — NEW
# Shows every step in clean expandable view
# ─────────────────────────────────────────
def render_trace_steps(steps, all_failures):
    st.subheader("🗂️ Full Execution Trace")

    failure_steps = set(f["step"] for f in all_failures)

    ACTOR_ICONS = {
        "user": "👤",
        "agent": "🤖",
        "tool": "🛠️",
        "system": "⚙️"
    }

    STATUS_ICONS = {
        "SUCCESS": "🟢",
        "success": "🟢",
        "FAILED": "🔴",
        "failed": "🔴",
        "WARNING": "🟡",
        "warning": "🟡",
        "SUSPENDED": "🟠",
        "PENDING": "🟠"
    }

    for step in steps:
        actor = step.get("actor", "unknown")
        step_num = step.get("step")
        step_type = step.get("step_type") or actor
        duration = step.get("duration_ms")
        step_hash = step.get("step_hash")
        status = step.get("status", "")
        step_id = step.get("step_id", f"step_{step_num}")
        has_failure = step_num in failure_steps

        icon = ACTOR_ICONS.get(actor, "📄")
        status_icon = STATUS_ICONS.get(status, "⚪")
        failure_badge = " 🚨" if has_failure else ""
        duration_str = f" · {duration}ms" if duration else ""
        label = f"{icon} Step {step_num}: {step_type.upper()}{duration_str}{failure_badge}"

        with st.expander(label, expanded=has_failure):
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                st.caption(f"**Actor:** {actor.upper()}")
                st.caption(f"**Type:** {step_type}")

            with col2:
                if status:
                    st.caption(f"**Status:** {status_icon} {status}")
                if duration:
                    st.caption(f"**Duration:** {duration}ms")

            with col3:
                if step_hash:
                    st.caption(f"**State Hash:** `{step_hash[:12]}...`")
                st.caption(f"**ID:** `{step_id}`")

            st.divider()

            content = step.get("content", "")
            if actor == "tool" or content.startswith("{") or content.startswith("["):
                try:
                    parsed_content = json.loads(content)
                    st.json(parsed_content)
                except Exception:
                    st.code(content, language="text")
            else:
                st.markdown(content)

            # Show failures inline for this step
            if has_failure:
                step_failures = [f for f in all_failures if f["step"] == step_num]
                for f in step_failures:
                    severity = f.get("severity", "").upper()
                    ftype = f.get("failure_type", "").upper()
                    color = "🔴" if severity == "CRITICAL" else "🟡" if severity == "HIGH" else "🔵"
                    st.warning(f"{color} **{ftype}** — {f.get('description', '')}")
                    if f.get("contradicted_by"):
                        st.caption(f"Contradicted by: {f.get('contradicted_by', '')[:200]}")


# ─────────────────────────────────────────
# DETERMINISM SECTION — NEW (nano-vm only)
# ─────────────────────────────────────────
def render_determinism_section(metrics, steps):
    if not metrics or not metrics.get("_is_nano_vm"):
        return

    st.markdown("---")
    st.subheader("🛡️ Determinism & Integrity")

    trace_id = metrics.get("trace_id", "N/A")
    status = metrics.get("status", "UNKNOWN")
    snapshots = metrics.get("_nano_vm_snapshots", {})

    STATUS_COLORS = {
        "SUCCESS": "green",
        "FAILED": "red",
        "SUSPENDED": "orange",
        "BUDGET_EXCEEDED": "red",
        "STALLED": "orange"
    }
    status_color = STATUS_COLORS.get(status, "gray")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**FSM Status:** :{status_color}[{status}]")
    with col2:
        st.markdown(f"**Trace ID:** `{trace_id}`")

    if snapshots:
        st.info("State hashes verify each step produced a deterministic state transition.")

        hash_data = []
        seen_hashes = set()
        duplicate_hashes = set()

        for s in steps:
            h = s.get("step_hash")
            if h:
                is_dup = h in seen_hashes
                if is_dup:
                    duplicate_hashes.add(h)
                seen_hashes.add(h)
                hash_data.append({
                    "Step": s["step"],
                    "Step ID": s.get("step_id", ""),
                    "Hash": h,
                    "Integrity": "⚠️ Duplicate" if is_dup else "✅ Unique"
                })

        if hash_data:
            st.dataframe(hash_data, hide_index=True, use_container_width=True)
            if duplicate_hashes:
                st.warning(f"⚠️ {len(duplicate_hashes)} duplicate state hashes detected — may indicate loops or stuck states.")
            else:
                st.success("✅ All state hashes unique — no loops or stuck states detected.")

        st.caption("To verify reproducibility: run the same Program with identical Context. Trace IDs will differ but step hashes must match.")

    final_output = metrics.get("final_output")
    if final_output:
        with st.expander("Final Output"):
            st.json(final_output)


# ─────────────────────────────────────────
# MAIN ANALYZE BUTTON
# ─────────────────────────────────────────
if st.button("Analyze Trace", type="primary"):
    if not trace_input:
        st.error("Please paste a trace.")
    else:
        with st.spinner("Analyzing trace..."):
            steps, metrics = parse_trace(trace_input)

            if not steps:
                st.error("Could not parse trace. Make sure it is valid JSON or line format (actor: content).")
            else:
                # Layer 1 — deterministic + semantic
                failures = detect_failures(steps)

                # Context drop detection
                context_failures = detect_context_drops(steps)

                # Latency detection
                latency_failures = detect_latency_issues(steps)

                # Merge all failures
                all_failures = failures + context_failures + latency_failures

                pattern = detect_pattern(all_failures)
                score, breakdown = compute_score(all_failures, pattern)
                prompt = build_prompt(steps, all_failures, score, breakdown)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
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

                    # ── METRICS ──
                    metrics_insights = extract_metrics_insights(metrics)
                    if metrics_insights:
                        st.subheader("💡 Cost & Efficiency Insights")
                        for insight in metrics_insights:
                            st.markdown(insight)

                    # ── LATENCY ──
                    if latency_failures:
                        st.subheader("⚡ Latency Bottlenecks")
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
                    if not all_failures and not parsed.get("failures"):
                        st.success("No failures detected.")

                    shown_types = set()
                    layer1_failure_types = [f.get("failure_type") for f in all_failures]
                    if "hallucinated_retry" in layer1_failure_types:
                        shown_types.add("hallucination")
                        shown_types.add("tool_misuse")
                    if "critical_system_failure" in layer1_failure_types:
                        shown_types.add("numerical_mismatch")
                        shown_types.add("hallucination")

                    for f in all_failures:
                        ftype = f.get("failure_type", "unknown").upper()
                        severity = f.get("severity", "").upper()
                        color = "🔴" if severity == "CRITICAL" else "🟡" if severity == "HIGH" else "🔵"
                        st.markdown(f"{color} **{ftype}** — Step {f.get('step','?')} — {severity}")
                        st.markdown(f"*Evidence:* {f.get('evidence','')[:200]}")
                        if f.get('contradicted_by'):
                            st.markdown(f"*Contradicted by:* {f.get('contradicted_by','')[:200]}")
                        shown_types.add(f.get("failure_type"))
                        st.divider()

                    for f in parsed.get("failures", []):
                        fp = f.get("failure_point", {})
                        ftype_raw = f.get("failure_type", "unknown")
                        if ftype_raw in shown_types:
                            continue
                        severity = f.get("severity", "").upper()
                        ftype = ftype_raw.upper()
                        color = "🔴" if severity == "CRITICAL" else "🟡" if severity == "HIGH" else "🔵"
                        st.markdown(f"{color} **{ftype}** — Step {fp.get('step','?')} — {severity}")
                        st.markdown(f"*Evidence:* {fp.get('evidence','')}")
                        st.markdown(f"*Confirmed cause:* {f.get('likely_cause',{}).get('confirmed','')}")
                        fix = f.get("suggested_fix", {})
                        st.markdown(f"⚡ **Quick fix:** {fix.get('quick','')}")
                        st.markdown(f"🏗️ **Robust fix:** {fix.get('robust','')}")
                        shown_types.add(ftype_raw)
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
                    logical_failure_types = ["hallucination", "tool_misuse", "action_skipped",
                          "date_misinterpretation", "numerical_mismatch",
                          "self_contradiction", "context_drop", "calculation_error"]
                    has_logical_failures = any(
                         f.get("failure_type") in logical_failure_types
                         for f in parsed.get("failures", [])
                    )
                    if confidence == 0.0 and not has_logical_failures:
                        st.info("N/A — performance issues only, no logical failures detected")
                    else:
                        st.progress(float(confidence))
                        st.markdown(f"{confidence:.2f} / 1.0")

                    # ── FULL TRACE VISUALIZATION — NEW ──
                    st.markdown("---")
                    render_trace_steps(steps, all_failures)

                    # ── DETERMINISM SECTION — NEW (nano-vm only) ──
                    render_determinism_section(metrics, steps)

                except json.JSONDecodeError:
                    st.error("Failed to parse response. Raw output:")
                    st.markdown(raw)

st.divider()
st.caption("Agent Debugger | AI Agent Observability")
