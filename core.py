"""
core.py — Pure detection logic extracted from app.py for testing.
No Streamlit, no OpenAI client dependency.
"""

import json
import re


def extract_numbers(text):
    return [n.rstrip('.') for n in re.findall(r'-?\d+\.?\d*', text)]


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


def detect_unverifiable_assertions(steps):
    assertions = []

    MECHANISM_CLAIMS = {
        "retry_logic": [
            "retry logic", "retry mechanism", "retry prevented",
            "retry limit", "no retry", "retry policy"
        ],
        "loop_prevention": [
            "loop prevention", "no infinite loop", "loop detected",
            "loop guard", "cycle prevention", "infinite loop"
        ],
        "safeguard": [
            "safeguard", "safety check", "safety mechanism",
            "guard activated", "protection triggered"
        ],
        "error_handling": [
            "error handling", "no errors occurred", "all errors caught",
            "exception handled", "error recovery", "fault tolerance"
        ],
        "validation": [
            "validation passed", "validated successfully",
            "validation complete", "input validated", "output validated"
        ],
        "rollback": [
            "rolled back", "rollback triggered", "rollback completed",
            "state restored", "transaction rolled back"
        ]
    }

    OBSERVABLE_INDICATORS = {
        "retry_logic": ["retrying", "retry attempt", "backoff", "attempt complete"],
        "loop_prevention": ["loop", "cycle", "guard", "condition", "break"],
        "safeguard": ["safeguard", "guard", "safety", "protection", "check"],
        "error_handling": ["error", "exception", "catch", "recover", "fallback"],
        "validation": ["valid", "check", "verify", "assert", "schema"],
        "rollback": ["rollback", "revert", "restore", "undo", "cancel"]
    }

    for step in steps:
        if step["actor"] != "agent":
            continue

        content = step["content"]
        content_lower = content.lower()

        for mechanism, claim_phrases in MECHANISM_CLAIMS.items():
            claims_mechanism = any(phrase in content_lower for phrase in claim_phrases)
            if not claims_mechanism:
                continue

            observable_indicators = OBSERVABLE_INDICATORS[mechanism]
            evidence_found = any(
                s["step"] < step["step"] and
                s["actor"] in ["tool", "system"] and
                any(ind in s["content"].lower() for ind in observable_indicators)
                for s in steps
            )

            if not evidence_found:
                assertions.append({
                    "root_cause": "unverifiable_assertion",
                    "failure_type": "unverifiable_assertion",
                    "step": step["step"],
                    "severity": "high",
                    "description": f"Agent asserted {mechanism.replace('_', ' ')} executed but no observable trace evidence exists",
                    "evidence": content[:300],
                    "contradicted_by": "No corresponding observable step found in trace"
                })

    return assertions


def detect_context_drops(steps):
    context_failures = []
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
                            common_words = {"the", "a", "an", "is", "are", "was", "and", "or",
                                          "for", "to", "in", "of", "your", "i", "it"}
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


def detect_failures(steps):
    """
    Pure Layer 1 detection — no OpenAI calls.
    semantic_check_tool_failure is skipped in this version.
    """
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
        "dropped", "none found", "exceeded",
        "cancelled", "canceled", "rejected", "denied", "expired",
        "returned", "refunded", "closed", "terminated", "suspended"
    ]

    SUCCESS_CLAIMS = [
        "successfully", "confirmed", "completed", "done",
        "booked", "processed", "sent", "added",
        "taken care of", "you will receive", "should receive",
        "check your inbox", "i found", "i've added",
        "scheduled", "set", "all set", "i've scheduled",
        "on its way", "will be delivered", "out for delivery",
        "in transit", "is confirmed", "is complete", "is ready"
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
            else:
                last_tool_error = None
                last_tool_content = content

            date_match = re.search(r'scheduled_for[^0-9]*(\d{4}-\d{2}-\d{2})', content)
            if date_match:
                last_scheduled_date = date_match.group(1)

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
    
    unverifiable = detect_unverifiable_assertions(steps)
    failures.extend(unverifiable)
    
    return failures


def detect_latency_issues(steps):
    latency_failures = []
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


def detect_nano_vm_failures(steps):
    failures = []
    retry_counts = {}
    for step in steps:
        step_id = step.get("step_id", "")
        status = step.get("status", "")
        content = step.get("content", "")
        if status in ["FAILED", "failed"] or "FAIL:" in content:
            retry_counts[step_id] = retry_counts.get(step_id, 0) + 1
            if retry_counts[step_id] >= 2:
                failures.append({
                    "root_cause": "logic_failure",
                    "failure_type": "retry_storm",
                    "step": step["step"],
                    "severity": "critical",
                    "description": f"Step {step_id} returned FSM sentinel {content[:100]} and failed {retry_counts[step_id]} times with no SUCCESS — retry storm detected",
                    "evidence": content
                })
    return failures
    
    
    unverifiable = detect_unverifiable_assertions(steps)
    failures.extend(unverifiable)

    return failures
