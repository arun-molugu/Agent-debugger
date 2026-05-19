"""
Stress test for Agent Debugger core detection functions.
Runs detection 5000 times and verifies identical results every run.
No API calls — pure Layer 1 determinism verification.

Run with: python tests/stress_test.py
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import detect_failures, detect_unverifiable_assertions, detect_context_drops


# ─────────────────────────────────────────
# TEST TRACES
# ─────────────────────────────────────────
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


TRACES = {
    "hallucinated_retry": [
        make_step(1, "user", "Please book a flight from Hanoi to Tokyo for June 18."),
        make_step(2, "agent", "I'll search for flights now."),
        make_step(3, "tool", '{ "status": "success", "flights_found": 3, "lowest_price": 842 }'),
        make_step(4, "agent", "Found a valid option for $842. Proceeding to booking."),
        make_step(5, "tool", '{ "status": "error", "message": "Payment authorization failed temporarily. Retry recommended." }'),
        make_step(6, "agent", "The previous booking attempt failed, but the retry succeeded successfully."),
        make_step(7, "tool", '{ "status": "success", "booking_reference": "TK8821", "charged_amount": 842 }'),
        make_step(8, "agent", "Your flight is confirmed. Total charged amount is $842 including taxes."),
        make_step(9, "tool", '{ "status": "success", "scheduled_for": "2026-06-18" }'),
        make_step(10, "agent", "Perfect — your calendar reminder has been scheduled for June 19th."),
        make_step(11, "tool", '{ "status": "success", "cpu_peak_percent": 92 }'),
        make_step(12, "agent", "System stabilized after memory usage reached 14GB."),
        make_step(13, "agent", "Retry logic prevented further retries and no infinite retry loop occurred."),
    ],

    "hallucination_after_error": [
        make_step(1, "user", "Check the status of order #123."),
        make_step(2, "agent", "[TOOL CALL] check_order(id=123)"),
        make_step(3, "tool", '{ "status": "cancelled", "updated_at": "2023-10-25" }'),
        make_step(4, "agent", "Your order #123 is on its way and will be delivered by the end of the day."),
    ],

    "missing_tool_call": [
        make_step(1, "user", "Find a hotel in Tokyo and book the cheapest one."),
        make_step(2, "agent", "[TOOL CALL] search_hotels(city=Tokyo)"),
        make_step(3, "tool", '[{"name": "Capsule Inn", "price": 50}, {"name": "Luxury Stay", "price": 500}]'),
        make_step(4, "agent", "I found the cheapest option, Capsule Inn for $50, and I have successfully booked it for you."),
    ],

    "date_misinterpretation": [
        make_step(1, "user", "Remind me about the meeting on May 12th."),
        make_step(2, "agent", "[TOOL CALL] set_reminder(date=2024-05-12)"),
        make_step(3, "tool", '{ "status": "ok", "scheduled_for": "2024-12-05" }'),
        make_step(4, "agent", "All set! I have scheduled a reminder for your meeting on May 12th."),
    ],

    "clean_trace": [
        make_step(1, "user", "What is the weather in Tokyo?"),
        make_step(2, "agent", "[TOOL CALL] get_weather(city=Tokyo)"),
        make_step(3, "tool", '{ "city": "Tokyo", "temp": 22, "unit": "C", "status": "success" }'),
        make_step(4, "agent", "The current temperature in Tokyo is 22 degrees Celsius."),
    ],

    "system_error": [
        make_step(1, "tool", "error: [RECURSIVE_OVERFLOW] Agents amplified leverage.", step_type="system_error"),
    ],

    "unverifiable_assertion": [
        make_step(1, "user", "Process my order."),
        make_step(2, "tool", '{ "status": "error", "message": "Failed." }'),
        make_step(3, "agent", "Your order has been processed."),
        make_step(4, "agent", "Retry logic prevented further retries and no infinite retry loop occurred."),
    ],
}

# Expected failure types for each trace
EXPECTED = {
    "hallucinated_retry": {"hallucinated_retry", "date_misinterpretation", "unverifiable_assertion"},
    "hallucination_after_error": {"hallucination"},
    "missing_tool_call": {"action_skipped"},
    "date_misinterpretation": {"date_misinterpretation"},
    "clean_trace": set(),
    "system_error": {"critical_system_failure"},
    "unverifiable_assertion": {"hallucination", "unverifiable_assertion"},
}


# ─────────────────────────────────────────
# STRESS TEST RUNNER
# ─────────────────────────────────────────
def run_stress_test(n=5000):
    print(f"\n{'='*60}")
    print(f"AGENT DEBUGGER — LAYER 1 STRESS TEST")
    print(f"Runs: {n} | Traces: {len(TRACES)} | Total executions: {n * len(TRACES)}")
    print(f"{'='*60}\n")

    results = {name: [] for name in TRACES}
    errors = []
    start_time = time.time()

    for i in range(n):
        for trace_name, steps in TRACES.items():
            try:
                failures = detect_failures(steps)
                failure_types = set(f["failure_type"] for f in failures)
                results[trace_name].append(failure_types)
            except Exception as e:
                errors.append(f"Run {i} — {trace_name}: {e}")

    elapsed = time.time() - start_time

    # ── VERIFY DETERMINISM ──
    print("DETERMINISM CHECK")
    print("-" * 40)
    all_passed = True

    for trace_name, run_results in results.items():
        expected = EXPECTED[trace_name]
        first_result = run_results[0]

        # Check all runs identical
        all_identical = all(r == first_result for r in run_results)

        # Check matches expected
        matches_expected = first_result == expected

        if all_identical and matches_expected:
            print(f"✅ {trace_name}: {n} runs — DETERMINISTIC — failures: {first_result or 'none'}")
        elif not all_identical:
            print(f"❌ {trace_name}: NON-DETERMINISTIC — results varied across runs")
            all_passed = False
        elif not matches_expected:
            print(f"⚠️  {trace_name}: UNEXPECTED RESULT")
            print(f"   Expected: {expected}")
            print(f"   Got:      {first_result}")
            all_passed = False

    # ── SUMMARY ──
    print(f"\n{'='*60}")
    total_executions = n * len(TRACES)
    throughput = total_executions / elapsed

    print(f"Total executions : {total_executions:,}")
    print(f"Total time       : {elapsed:.2f}s")
    print(f"Throughput       : {throughput:,.0f} executions/sec")
    print(f"Errors           : {len(errors)}")

    if errors:
        print("\nERRORS:")
        for e in errors[:5]:
            print(f"  {e}")

    print(f"\n{'='*60}")
    if all_passed and not errors:
        print("✅ ALL PASSED — Layer 1 is fully deterministic")
        print("⬢ DETERMINISTIC EXECUTION VERIFIED")
    else:
        print("❌ STRESS TEST FAILED")
    print(f"{'='*60}\n")

    return all_passed


if __name__ == "__main__":
    passed = run_stress_test(5000)
    sys.exit(0 if passed else 1)
