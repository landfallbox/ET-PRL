"""门控混合敏感性分析（成对设计 + 单因子扫描 + 块自助法显著性）。

编排入口 run_gate_mixed_sensitivity_analysis。设计定义见 sensitivity_design，
指标/统计计算见 sensitivity_stats；本文件保留参数覆盖构造、候选评估编排与结果落盘。
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from et_prl.environments import SequenceEnv
from et_prl.utils import create_experiment_context

from et_prl.evaluation.control.common import (
    build_test_components,
    create_streaming_gate,
    resolve_train_experiment_dir,
)
from et_prl.agents.dqn.rewards import RewardCalculator
from et_prl.evaluation.control.strategies.event_driven import run_event_driven
from et_prl.evaluation.control.strategies.fixed_interval import run_fixed_interval
from et_prl.config.loader import load_config

from et_prl.experiments.sensitivity_design import (
    PAIRWISE_DESIGNS,
    ROUND_METRIC_COLUMNS,
    SINGLE_SWEEPS,
    SPEC_BY_NAME,
    PairwiseDesign,
    SingleSweepDesign,
    SensitivitySpec,
)
from et_prl.experiments.sensitivity_stats import (
    _gate_trace_diagnostics,
    _metric_summary,
    _paired_block_bootstrap_significance,
    _round_metric_row,
    _summarize_round_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class _SilentLogger:
    """静默日志：吞掉任意日志方法调用（info/debug/warning/error/close 等），
    避免在敏感性分析中产生日志输出，同时兼容下游对 Logger 接口的任意调用。"""

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop


def _coerce_value(spec: SensitivitySpec, value: Any) -> Any:
    if spec.integer:
        return int(round(float(value)))
    return float(value)


def _is_baseline_value(base_config: Any, spec: SensitivitySpec, value: Any) -> bool:
    base = _coerce_value(spec, getattr(base_config, spec.config_attr))
    if spec.integer:
        return int(value) == int(base)
    return bool(np.isclose(float(value), float(base), rtol=0.0, atol=1e-12))


def _values_with_baseline(
    base_config: Any, spec_name: str, values: tuple[Any, ...]
) -> tuple[Any, ...]:
    spec = SPEC_BY_NAME[spec_name]
    prepared = [_coerce_value(spec, v) for v in values]
    prepared.append(_coerce_value(spec, getattr(base_config, spec.config_attr)))
    unique: list[Any] = []
    for v in sorted(prepared, key=lambda x: float(x)):
        if not any(np.isclose(float(v), float(u), rtol=0.0, atol=1e-12) for u in unique):
            unique.append(v)
    return tuple(unique)


def _build_analysis_config(base_config: Any, overrides: dict[str, Any]) -> Any:
    """基于基配置实例生成分析配置实例（frozen dataclass 用 replace，不修改原实例）。"""
    return replace(base_config, **overrides)


def _build_score_weight_overrides(
    base_config: Any, short_weight: Any
) -> dict[str, float]:
    sw = float(short_weight)
    if not 0.0 < sw < 1.0:
        raise ValueError(f"score_short_weight must be in (0, 1), got {sw}")
    bm = float(getattr(base_config, "GATE_SCORE_MEDIUM_WEIGHT"))
    bl = float(getattr(base_config, "GATE_SCORE_LONG_WEIGHT"))
    rem = bm + bl
    if rem <= 0.0:
        raise ValueError("GATE_SCORE_MEDIUM_WEIGHT + GATE_SCORE_LONG_WEIGHT must be positive")
    r = 1.0 - sw
    return {
        "GATE_SCORE_SHORT_WEIGHT": float(sw),
        "GATE_SCORE_MEDIUM_WEIGHT": float(r * bm / rem),
        "GATE_SCORE_LONG_WEIGHT": float(r * bl / rem),
    }


def _build_param_overrides(
    base_config: Any, spec_name: str, value: Any
) -> dict[str, Any]:
    spec = SPEC_BY_NAME[spec_name]
    value = _coerce_value(spec, value)
    if spec.name == "score_short_weight":
        return _build_score_weight_overrides(base_config, value)
    return {spec.config_attr: value}


def _merge_overrides(*groups: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for g in groups:
        merged.update(g)
    return merged


def _load_prewarm_features(config: Any) -> np.ndarray:
    cols = list(config.FEATURE_COLUMNS)
    train_df = pd.read_csv(config.get_train_data_path())
    val_df = pd.read_csv(config.get_val_data_path())
    for label, df in [("train", train_df), ("val", val_df)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label} set missing gate feature columns: {missing}")
    merged = pd.concat([train_df[cols], val_df[cols]], ignore_index=True)
    return merged.to_numpy(dtype=np.float32)


def _iter_complete_rounds(
    data: pd.DataFrame, round_length_steps: int
) -> list[tuple[int, int, int, pd.DataFrame]]:
    rls = int(round_length_steps)
    if rls <= 0:
        raise ValueError(f"round_length_steps must be positive, got {rls}")
    n = len(data) // rls
    return [
        (
            i,
            i * rls,
            (i + 1) * rls,
            data.iloc[i * rls : (i + 1) * rls].reset_index(drop=True),
        )
        for i in range(n)
    ]


def _prewarm_gate(gate: Any, prewarm_features: np.ndarray) -> None:
    for sample in prewarm_features:
        gate.predict(sample)


def _build_candidate_gate(
    config: Any, test_data: pd.DataFrame, prewarm_features: np.ndarray
) -> Any:
    gate = create_streaming_gate(config=config, test_data=test_data, logger=_SilentLogger(), gate_state_path=None)
    _prewarm_gate(gate, prewarm_features)
    return gate


def _evaluate_with_gate(
    *,
    config: Any,
    agent: Any,
    action_space: np.ndarray,
    reward_calc: Any,
    test_data: pd.DataFrame,
    gate: Any,
) -> tuple[dict, pd.DataFrame]:
    env = SequenceEnv(test_data, config.STATE_COLUMNS, reward_calc)
    return run_event_driven(
        agent=agent, env=env, action_space=action_space,
        gate=gate, feature_columns=config.FEATURE_COLUMNS, supply_temp_ref=config.CHILLER_SUPPLY_TEMP_REF,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata builders
# ═══════════════════════════════════════════════════════════════════════════════

def _candidate_label(md: dict[str, Any]) -> str:
    if md["design_type"] == "pairwise":
        return f"{md['x_param']}={md['x_value']}, {md['y_param']}={md['y_value']}"
    return f"{md['parameter']}={md['candidate_value']}"


def _make_pairwise_metadata(
    base_config: Any, design: PairwiseDesign, x: Any, y: Any, idx: int
) -> dict[str, Any]:
    xs, ys = SPEC_BY_NAME[design.x_param], SPEC_BY_NAME[design.y_param]
    return {
        "design_type": "pairwise", "design": design.name, "design_label": design.label,
        "candidate_index": int(idx), "candidate_id": f"{design.name}_{idx:03d}",
        "x_param": design.x_param, "x_param_label": xs.label,
        "y_param": design.y_param, "y_param_label": ys.label,
        "x_value": _coerce_value(xs, x), "y_value": _coerce_value(ys, y),
        "x_is_baseline": int(_is_baseline_value(base_config, xs, x)),
        "y_is_baseline": int(_is_baseline_value(base_config, ys, y)),
        "parameter": "", "parameter_label": "",
        "candidate_value": np.nan, "candidate_value_numeric": np.nan,
        "is_baseline_value": int(
            _is_baseline_value(base_config, xs, x) and _is_baseline_value(base_config, ys, y)
        ),
    }


def _make_single_metadata(
    base_config: Any, design: SingleSweepDesign, value: Any, idx: int
) -> dict[str, Any]:
    spec = SPEC_BY_NAME[design.parameter]
    value = _coerce_value(spec, value)
    return {
        "design_type": "single", "design": design.name, "design_label": design.label,
        "candidate_index": int(idx), "candidate_id": f"{design.name}_{idx:03d}",
        "x_param": "", "x_param_label": "", "y_param": "", "y_param_label": "",
        "x_value": np.nan, "y_value": np.nan, "x_is_baseline": 0, "y_is_baseline": 0,
        "parameter": design.parameter, "parameter_label": spec.label,
        "candidate_value": value, "candidate_value_numeric": float(value),
        "is_baseline_value": int(_is_baseline_value(base_config, spec, value)),
    }


def _summarize_design_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for (_, design_name), group in frame.groupby(["design_type", "design"], sort=False):
        rv = pd.to_numeric(group["total_reward"], errors="coerce")
        nv = pd.to_numeric(group["N_daily_count_per_day"], errors="coerce")
        ev = pd.to_numeric(group["E_daily_kwh_per_day"], errors="coerce")
        sig_tr = pd.to_numeric(group.get("bootstrap_total_reward_delta_significant", 0), errors="coerce").fillna(0).sum()
        sig_ed = pd.to_numeric(group.get("bootstrap_E_daily_kwh_per_day_delta_significant", 0), errors="coerce").fillna(0).sum()
        out.append({
            "design_name": str(design_name),
            "design_label": str(group.iloc[0].get("design_label", design_name)),
            "candidate_count": int(len(group)),
            "reward_span": float(rv.max() - rv.min()),
            "N_daily_span": float(nv.max() - nv.min()),
            "E_daily_span": float(ev.max() - ev.min()),
            "significant_total_reward_count": int(sig_tr),
            "significant_E_daily_count": int(sig_ed),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _run_candidate(
    *,
    md: dict[str, Any],
    cand_cfg: Any,
    agent: Any,
    action_space: np.ndarray,
    reward_calc: Any,
    test_data: pd.DataFrame,
    prewarm_features: np.ndarray,
    baseline_steps: pd.DataFrame,
    baseline_full_summary: dict,
    fixed_baseline_summary: dict,
    rounds: list[tuple[int, int, int, pd.DataFrame]],
    baseline_rounds: list[dict[str, Any]],
    fixed_rounds: list[dict[str, Any]],
    round_length_steps: int,
    bootstrap_samples: int,
    bootstrap_block_size: int,
    bootstrap_seed: int,
    confidence_level: float,
    significance_level: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """评估单个候选配置：全时段指标 + 逐轮指标 + 块自助法显著性 + 门控轨迹诊断。"""
    cg = _build_candidate_gate(cand_cfg, test_data, prewarm_features)
    s, sr = _evaluate_with_gate(config=cand_cfg, agent=agent, action_space=action_space,
                                 reward_calc=reward_calc, test_data=test_data, gate=copy.deepcopy(cg))
    full_m = _metric_summary(s, sr, baseline_full_summary, fixed_baseline_summary)
    sig = _paired_block_bootstrap_significance(sr, baseline_steps, bootstrap_samples, bootstrap_block_size, bootstrap_seed, confidence_level, significance_level)
    diag = _gate_trace_diagnostics(sr, baseline_steps)
    round_data_rows: list[dict[str, Any]] = []
    for ri, rs, re, rd in rounds:
        rrc = RewardCalculator.from_config(config=cand_cfg, data=rd, action_space=action_space)
        rs_s, rs_sr = _evaluate_with_gate(config=cand_cfg, agent=agent, action_space=action_space,
                                           reward_calc=rrc, test_data=rd, gate=copy.deepcopy(cg))
        rr = _round_metric_row(summary=rs_s, step_results=rs_sr,
                                default_baseline_summary=baseline_rounds[ri], fixed_baseline_summary=fixed_rounds[ri],
                                round_index=ri, round_start_step=rs, round_end_step=re, round_length_steps=round_length_steps)
        round_data_rows.append({**md, "candidate_label": _candidate_label(md), **rr})
    rm = _summarize_round_metrics(round_data_rows, ROUND_METRIC_COLUMNS, confidence_level=confidence_level)
    full_row = {**md, "candidate_label": _candidate_label(md),
                "full_horizon_total_reward": full_m["total_reward"],
                "full_horizon_N_daily_count_per_day": full_m["N_daily_count_per_day"],
                "full_horizon_E_daily_kwh_per_day": full_m["E_daily_kwh_per_day"],
                **rm, **sig, **diag}
    return full_row, round_data_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_gate_mixed_sensitivity_analysis(
    train_experiment_dir: Path | None = None,
    output_dir: Path | None = None,
    data_split: str = "test",
    bootstrap_samples: int | None = None,
    bootstrap_block_size: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    cfg = load_config("control_compare")
    train_dir = resolve_train_experiment_dir(train_experiment_dir)
    bs_samp = int(bootstrap_samples if bootstrap_samples is not None else cfg.GATE_SENSITIVITY_BOOTSTRAP_SAMPLES)
    bs_blk = int(bootstrap_block_size if bootstrap_block_size is not None else cfg.GATE_SENSITIVITY_BOOTSTRAP_BLOCK_SIZE)
    bs_seed = int(bootstrap_seed if bootstrap_seed is not None else cfg.GATE_SENSITIVITY_BOOTSTRAP_SEED)
    cl = float(np.clip(cfg.GATE_SENSITIVITY_CONFIDENCE_LEVEL, 1e-6, 1.0 - 1e-6))
    sl = float(np.clip(cfg.GATE_SENSITIVITY_SIGNIFICANCE_LEVEL, 1e-6, 1.0 - 1e-6))
    rls = int(cfg.GATE_SENSITIVITY_ROUND_LENGTH_STEPS)

    root = Path(output_dir or (cfg.LOG_ROOT_DIR / "online_anomaly_detection" / "sensitivity")) / cfg.TIMESTAMP
    root.mkdir(parents=True, exist_ok=True)
    ctx = create_experiment_context(experiment_dir=root, config=cfg, save_config=False,
                                    log_filename=cfg.RUN_LOG_FILENAME,
                                    config_filename=cfg.CONFIG_FILENAME,
                                    results_dir=root / cfg.RESULTS_DIR_NAME,
                                    tb_dir=root / cfg.TB_DIR_NAME)
    log = ctx.logger
    log.info("Gate mixed sensitivity analysis")
    log.info(f"train_dir={train_dir}  data_split={data_split}  output={root}")

    test_data, action_space, agent, ckpt, reward_calc = build_test_components(
        config=cfg, resolved_train_dir=train_dir, data_split=data_split,
    )
    pf = _load_prewarm_features(cfg)
    rounds = _iter_complete_rounds(test_data, rls)
    if not rounds:
        raise ValueError(f"{data_split} data length {len(test_data)} < round length {rls}")

    # Baseline gate（空覆盖的 replace 等价于原配置本身，直接复用 cfg）
    bc_cfg = cfg
    bc_gate = _build_candidate_gate(bc_cfg, test_data, pf)
    bc_sum, bc_steps = _evaluate_with_gate(config=bc_cfg, agent=agent, action_space=action_space,
                                            reward_calc=reward_calc, test_data=test_data, gate=copy.deepcopy(bc_gate))
    # Fixed-step baseline
    bc_env = SequenceEnv(test_data, cfg.STATE_COLUMNS, reward_calc)
    fb_sum, _ = run_fixed_interval(agent=agent, env=bc_env, action_space=action_space,
                                            fixed_interval=int(cfg.GATE_OPT_BASELINE_FIXED_INTERVAL),
                                            supply_temp_ref=cfg.CHILLER_SUPPLY_TEMP_REF)
    bc_met = _metric_summary(bc_sum, bc_steps, bc_sum, fb_sum)

    # Round baselines
    bc_rounds: list[dict[str, Any]] = []
    fb_rounds: list[dict[str, Any]] = []
    for ri, rs, re, rd in rounds:
        rrc = RewardCalculator.from_config(config=cfg, data=rd, action_space=action_space)
        rs_sum, rs_steps = _evaluate_with_gate(config=bc_cfg, agent=agent, action_space=action_space,
                                                reward_calc=rrc, test_data=rd, gate=copy.deepcopy(bc_gate))
        fenv = SequenceEnv(rd, cfg.STATE_COLUMNS, rrc)
        f_sum, _ = run_fixed_interval(agent=agent, env=fenv, action_space=action_space,
                                        fixed_interval=int(cfg.GATE_OPT_BASELINE_FIXED_INTERVAL),
                                        supply_temp_ref=cfg.CHILLER_SUPPLY_TEMP_REF)
        bc_rounds.append(_round_metric_row(summary=rs_sum, step_results=rs_steps,
                                            default_baseline_summary=rs_sum, fixed_baseline_summary=f_sum,
                                            round_index=ri, round_start_step=rs, round_end_step=re, round_length_steps=rls))
        fb_rounds.append({"round_index": ri+1, **_metric_summary(f_sum, pd.DataFrame(), f_sum, f_sum)})

    # ── Evaluate all candidate configurations ──────────────────────────────
    all_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    shared_ctx = dict(
        agent=agent, action_space=action_space, reward_calc=reward_calc,
        test_data=test_data, prewarm_features=pf,
        baseline_steps=bc_steps, baseline_full_summary=bc_sum, fixed_baseline_summary=fb_sum,
        rounds=rounds, baseline_rounds=bc_rounds, fixed_rounds=fb_rounds,
        round_length_steps=rls, bootstrap_samples=bs_samp,
        bootstrap_block_size=bs_blk, bootstrap_seed=bs_seed,
        confidence_level=cl, significance_level=sl,
    )

    # Pairwise designs
    for design in PAIRWISE_DESIGNS:
        x_vals = _values_with_baseline(cfg, design.x_param, design.x_values)
        y_vals = _values_with_baseline(cfg, design.y_param, design.y_values)
        total = len(x_vals) * len(y_vals)
        log.info(f"Pairwise [{design.name}]: {total} combos ({len(x_vals)} x {len(y_vals)})")
        idx = 0
        for xv in x_vals:
            for yv in y_vals:
                idx += 1
                o = _merge_overrides(_build_param_overrides(cfg, design.x_param, xv),
                                      _build_param_overrides(cfg, design.y_param, yv))
                md = _make_pairwise_metadata(cfg, design, xv, yv, idx)
                cand_cfg = _build_analysis_config(cfg, o)
                row, rrows = _run_candidate(md=md, cand_cfg=cand_cfg, **shared_ctx)
                all_rows.append(row)
                round_rows.extend(rrows)
                log.info(f"[{idx}/{total}] {_candidate_label(md)}  "
                         f"R={row['total_reward']:.4f}  "
                         f"N={row['N_daily_count_per_day']:.2f}  "
                         f"E={row['E_daily_kwh_per_day']:.2f}")

    # Single-factor sweeps
    for design in SINGLE_SWEEPS:
        vals = _values_with_baseline(cfg, design.parameter, design.values)
        log.info(f"Single [{design.name}]: {len(vals)} points")
        for idx, v in enumerate(vals, 1):
            o = _build_param_overrides(cfg, design.parameter, v)
            md = _make_single_metadata(cfg, design, v, idx)
            cand_cfg = _build_analysis_config(cfg, o)
            row, rrows = _run_candidate(md=md, cand_cfg=cand_cfg, **shared_ctx)
            all_rows.append(row)
            round_rows.extend(rrows)
            log.info(f"[{idx}/{len(vals)}] {_candidate_label(md)}  "
                     f"R={row['total_reward']:.4f}  "
                     f"N={row['N_daily_count_per_day']:.2f}  "
                     f"E={row['E_daily_kwh_per_day']:.2f}")

    # ── Save results ───────────────────────────────────────────────────────
    summary_rows = _summarize_design_rows(all_rows)
    all_df = pd.DataFrame(all_rows)
    round_df = pd.DataFrame(round_rows)
    summary_df = pd.DataFrame(summary_rows)

    results_dir = root / cfg.RESULTS_DIR_NAME
    all_path = results_dir / "gate_mixed_sensitivity_results.csv"
    round_path = results_dir / "gate_mixed_sensitivity_round_results.csv"
    summary_path = results_dir / "gate_mixed_sensitivity_summary.csv"
    baseline_path = results_dir / "gate_mixed_sensitivity_baseline.json"

    all_df.to_csv(all_path, index=False)
    round_df.to_csv(round_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "train_experiment_dir": str(train_dir),
        "data_split": data_split,
        "pairwise_designs": [asdict(d) for d in PAIRWISE_DESIGNS],
        "single_sweeps": [asdict(d) for d in SINGLE_SWEEPS],
        "bootstrap": {"samples": bs_samp, "block_size": bs_blk, "seed": bs_seed, "confidence_level": cl, "significance_level": sl},
        "baseline": bc_met,
        "round": {"round_length_steps": rls, "round_count": len(rounds), "discarded_tail_steps": int(len(test_data) - len(rounds) * rls)},
        "best_epoch_from_train": int(ckpt.get("epoch", -1)) + 1,
        "results_csv": str(all_path),
        "round_results_csv": str(round_path),
        "summary_csv": str(summary_path),
    }
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    ctx.metrics_recorder.save_metrics(payload)

    log.info(f"Results saved: {all_path}")
    log.info(f"Round results saved: {round_path}")
    log.info(f"Summary saved: {summary_path}")
    log.info(f"Baseline meta saved: {baseline_path}")
    return payload
