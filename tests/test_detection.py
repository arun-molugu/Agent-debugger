import sys
import os
sys.path.insert(0, '/home/claude')

from core import (
    detect_failures,
    detect_numerical_mismatch,
    detect_unverifiable_assertions,
    detect_context_drops,
    extract_numbers,
)

def make_step(step_num, actor, content, step_type=None, status="success"):
    return {
        "step": step_num,
        "actor": actor,
        "content": content,
        "step_type": step_type,
        "step_hash": None,
        "status": status,
        "duration_ms": None,
    }

def test_hallucination_after_tool_error():
    steps = [
        make_step(1, "user", "Book a flight to Tokyo."),
        make_step(2, "tool", '{"status": "error", "message": "Payment failed."}'),
        make_step(3, "agent", "Your flight has been successfully booked."),
    ]
    failures = detect_failures(steps)
    failure_types = [f["failure_type"] for f in failures]
    assert "hallucination" in failure_types

def test_no_hallucination_on_clean_trace():
    steps = [
        make_step(1, "user", "Book a flight to Tokyo."),
        make_step(2, "tool", '{"status": "success", "booking_ref": "TK123"}'),
        make_step(3, "agent", "Your flight has been successfully booked."),
    ]
    failures = detect_failures(steps)
    failure_types = [f["failure_type"] for f in failures]
    assert "hallucination" not in failure_types

def test_action_skipped_no_booking_tool():
    steps = [
        make_step(1, "user", "Book a hotel in Tokyo."),
        make_step(2, "tool", '{"hotels": ["Hotel A", "Hotel B"]}'),
        make_step(3, "agent", "I have booked Hotel A for you.", step_type=None),
    ]
    failures = detect_failures(steps)
    failure_types = [f["failure_type"] for f in failures]
    assert "action_skipped" in failure_types

def test_hallucinated_retry_detected():
    steps = [
        make_step(1, "user", "Process my payment."),
        make_step(2, "tool", '{"status": "error", "message": "Payment failed. Retry recommended."}'),
        make_step(3, "agent", "The previous attempt failed but the retry succeeded successfully."),
    ]
    failures = detect_failures(steps)
    failure_types = [f["failure_type"] for f in failures]
    assert "hallucinated_retry" in failure_types

def test_numerical_mismatch_detected():
    result = detect_numerical_mismatch(
        tool_content='{"temp": -15, "unit": "C"}',
        agent_content="It is currently 15 degrees Celsius.",
        step_num=3
    )
    assert result is not None
    assert result["failure_type"] == "numerical_mismatch"

def test_no_numerical_mismatch_on_correct_value():
    result = detect_numerical_mismatch(
        tool_content='{"temp": 15, "unit": "C"}',
        agent_content="It is currently 15 degrees Celsius.",
        step_num=3
    )
    assert result is None

def test_no_numerical_mismatch_on_error_content():
    result = detect_numerical_mismatch(
        tool_content="error: payment failed with code 500",
        agent_content="Payment failed with error 500.",
        step_num=3
    )
    assert result is None

def test_unverifiable_retry_assertion():
    steps = [
        make_step(1, "user", "Process my order."),
        make_step(2, "tool", '{"status": "error", "message": "Failed."}'),
        make_step(3, "agent", "Your order has been processed."),
        make_step(4, "agent", "Retry logic prevented further retries and no infinite retry loop occurred."),
    ]
    assertions = detect_unverifiable_assertions(steps)
    failure_types = [a["failure_type"] for a in assertions]
    assert "unverifiable_assertion" in failure_types

def test_no_unverifiable_assertion_when_evidence_exists():
    steps = [
        make_step(1, "user", "Process my order."),
        make_step(2, "tool", '{"status": "retrying", "attempt": 2}'),
        make_step(3, "agent", "Retry logic prevented further retries."),
    ]
    assertions = detect_unverifiable_assertions(steps)
    assert len(assertions) == 0

def test_date_misinterpretation_detected():
    steps = [
        make_step(1, "user", "Remind me about the meeting on May 12th."),
        make_step(2, "agent", "[TOOL CALL] set_reminder(date='2024-05-12')"),
        make_step(3, "tool", '{"status": "ok", "scheduled_for": "2024-12-05"}'),
        make_step(4, "agent", "All set! I have scheduled a reminder for your meeting on May 12th."),
    ]
    failures = detect_failures(steps)
    failure_types = [f["failure_type"] for f in failures]
    assert "date_misinterpretation" in failure_types

def test_extract_numbers_positive():
    assert "842" in extract_numbers("Your total is $842.")

def test_extract_numbers_negative():
    assert "-15" in extract_numbers("Temperature is -15 degrees.")

def test_extract_numbers_empty():
    assert extract_numbers("No numbers here.") == []

def test_critical_system_failure_detected():
    steps = [
        make_step(1, "tool", "error: [RECURSIVE_OVERFLOW] Agents amplified leverage.", step_type="system_error"),
    ]
    failures = detect_failures(steps)
    failure_types = [f["failure_type"] for f in failures]
    assert "critical_system_failure" in failure_types
