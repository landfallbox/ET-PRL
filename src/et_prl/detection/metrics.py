from __future__ import annotations


def compute_ppr(event_total_reward: float, baseline_total_reward: float) -> float:
    baseline = float(baseline_total_reward)
    if baseline == 0.0:
        return 0.0
    return float(float(event_total_reward) / baseline)


def compute_delta_violation_time_pct(event_violation_time_pct: float, baseline_violation_time_pct: float) -> float:
    return float(float(event_violation_time_pct) - float(baseline_violation_time_pct))


def check_trigger_rate_constraint(trigger_rate: float, min_rate: float | None = None, max_rate: float | None = None) -> bool:
    current = float(trigger_rate)
    if min_rate is not None and current < float(min_rate):
        return False
    if max_rate is not None and current > float(max_rate):
        return False
    return True


def summarize_event_gate_constraints(
    ppr: float,
    delta_violation_time_pct: float,
    trigger_rate: float,
    min_ppr: float,
    max_delta_violation_time_pct: float,
    min_trigger_rate: float | None = None,
    max_trigger_rate: float | None = None,
) -> dict:
    ppr_ok = float(ppr) >= float(min_ppr)
    violation_ok = float(delta_violation_time_pct) <= float(max_delta_violation_time_pct)
    trigger_rate_ok = check_trigger_rate_constraint(
        trigger_rate=trigger_rate,
        min_rate=min_trigger_rate,
        max_rate=max_trigger_rate,
    )

    return {
        "ppr": float(ppr),
        "delta_violation_time_pct": float(delta_violation_time_pct),
        "trigger_rate": float(trigger_rate),
        "constraints": {
            "min_ppr": float(min_ppr),
            "max_delta_violation_time_pct": float(max_delta_violation_time_pct),
            "min_trigger_rate": float(min_trigger_rate) if min_trigger_rate is not None else None,
            "max_trigger_rate": float(max_trigger_rate) if max_trigger_rate is not None else None,
        },
        "is_ppr_ok": bool(ppr_ok),
        "is_delta_violation_ok": bool(violation_ok),
        "is_trigger_rate_ok": bool(trigger_rate_ok),
        "all_constraints_passed": bool(ppr_ok and violation_ok and trigger_rate_ok),
    }
