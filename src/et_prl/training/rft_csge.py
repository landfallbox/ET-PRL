"""RFT（Reinforced Fine-Tuning）：RSI 外层环的微调模块。

在策略 π₀ 部署到新工况窗口后，若 τ 触发（工况漂移/性能退化），
用 π₀ 在该窗口的在策略数据做 KL 锚定 PPO 微调，得到 π₁。

用法：
    PYTHONPATH=src python3 -m et_prl.training.rft_csge
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from et_prl.config.base import project_root
from et_prl.environments.csge_env import CsgeEnv
from et_prl.training.ppo_csge_train import (
    ActorCritic, RunningMeanStd, _load_calibration, _load_split,
    assess, REPLAY_SP_COL,
)

logger = logging.getLogger("rft_csge")


@dataclass
class RftConfig:
    # 漂移窗口（s0 前 N 行）
    drift_split: str = "s0"
    drift_rows: int = 4032          # 2 周（12 步/小时 × 24 × 14）
    # 微调超参
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 1e-4
    kl_coef: float = 1.0            # KL 锚定强度（0 = 纯微调）
    rollout_len: int = 4096
    n_epochs: int = 8
    minibatch_size: int = 512
    total_steps: int = 49152        # 12 个 rollout
    # 其他
    seed: int = 42
    device: str = "cpu"
    base_model: str = "outputs/ppo_csge/20261012_231148/model_best.pth"


def load_policy(path: str, device: str):
    ck = torch.load(path, map_location="cpu")
    net = ActorCritic(10, 13, tuple(ck["config"]["hidden"])).to(device)
    net.load_state_dict(ck["model_state"]); net.train(False)
    norm = RunningMeanStd(10)
    norm.mean = np.array(ck["obs_mean"]); norm.var = np.array(ck["obs_var"]); norm.count = 1.0
    return net, norm


def _kl_divergence(net, ref_net, obs_t, mask_t):
    """KL(π ‖ π_ref)，掩码内逐样本均值。"""
    with torch.no_grad():
        ref_logits, _ = ref_net(obs_t)
        ref_logits = ref_logits.masked_fill(~mask_t, -1e9)
    logits, _ = net(obs_t)
    logits = logits.masked_fill(~mask_t, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    logp_ref = F.log_softmax(ref_logits, dim=-1)
    p = logp.exp()
    return (p * (logp - logp_ref)).sum(dim=-1).mean()


def rft_finetune(net, norm, env, cfg: RftConfig) -> dict:
    """KL 锚定 PPO 微调。返回训练指标。"""
    device = torch.device(cfg.device)
    ref_net = ActorCritic(env.observation_size, env.action_size, (256, 128)).to(device)
    ref_net.load_state_dict(net.state_dict()); ref_net.train(False)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    B = cfg.rollout_len

    obs_buf = np.zeros((B, env.observation_size), np.float32)
    act_buf = np.zeros(B, np.int64)
    logp_buf = np.zeros(B, np.float32)
    rew_buf = np.zeros(B, np.float32)
    done_buf = np.zeros(B, np.float32)
    val_buf = np.zeros(B, np.float32)
    mask_buf = np.zeros((B, env.action_size), dtype=bool)

    obs, info0 = env.reset()
    mask = info0["action_mask"]
    global_step = 0
    update_no = 0
    t_start = time.time()
    hist = []

    while global_step < cfg.total_steps:
        last_value = 0.0
        for i in range(B):
            obs_buf[i] = obs
            mask_buf[i] = mask
            obs_t = torch.as_tensor(norm.normalize(obs), dtype=torch.float32, device=device)
            with torch.no_grad():
                a, logp, _, value = net.act(obs_t, mask=torch.as_tensor(mask, dtype=torch.bool, device=device))
            act_buf[i] = int(a.item())
            logp_buf[i] = float(logp.item())
            val_buf[i] = float(value.item())
            next_obs, r, term, trunc, info = env.step(act_buf[i])
            done = bool(term or trunc)
            rew_buf[i] = r
            done_buf[i] = 1.0 if done else 0.0
            if done:
                last_value = 0.0
                obs, info0 = env.reset()
                mask = info0["action_mask"]
            else:
                with torch.no_grad():
                    _, _, _, nv = net.act(torch.as_tensor(norm.normalize(next_obs), dtype=torch.float32, device=device))
                last_value = float(nv.item())
                mask = info["action_mask"]
            global_step += 1

        # GAE
        with torch.no_grad():
            adv = np.zeros(B, np.float32)
            last_gae = 0.0
            for t in reversed(range(B)):
                next_nonterminal = 1.0 - done_buf[t]
                next_val = last_value if t == B - 1 else val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_val * next_nonterminal - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                adv[t] = last_gae
        ret_buf = adv + val_buf

        obs_t = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act_buf, dtype=torch.long, device=device)
        logp_t = torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask_buf, dtype=torch.bool, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret_buf, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_kl = 0.0
        total_frac = 0.0
        n_mb = 0
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                logits, value = net(obs_t[idx])
                logits = logits.masked_fill(~mask_t[idx], -1e9)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[idx])
                ratio = torch.exp(new_logp - logp_t[idx])
                s1 = ratio * adv_t[idx]
                s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t[idx]
                loss_pi = -torch.min(s1, s2).mean()
                loss_v = nn.functional.mse_loss(value, ret_t[idx])
                loss_kl = cfg.kl_coef * _kl_divergence(net, ref_net, obs_t[idx], mask_t[idx])
                loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * dist.entropy().mean() + loss_kl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                total_kl += float(loss_kl.item())
                total_frac += float((ratio - 1.0).abs().mean().item())
                n_mb += 1

        update_no += 1
        kl_mean = total_kl / n_mb
        hist.append({"update": update_no, "step": global_step,
                     "mean_r": float(rew_buf.mean()), "kl": kl_mean,
                     "frac": total_frac / n_mb})
        logger.info("rft update %d | step %d/%d | mean_r %.4f | KL %.4f | frac %.4f | %.0f steps/s",
                    update_no, global_step, cfg.total_steps, float(rew_buf.mean()),
                    kl_mean, total_frac / n_mb, global_step / max(time.time() - t_start, 1e-6))
    return {"hist": hist, "final_kl": hist[-1]["kl"]}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = RftConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    cal = _load_calibration()

    s0 = _load_split(cfg.drift_split)
    drift = s0.iloc[:cfg.drift_rows].reset_index(drop=True)
    t = pd.to_datetime(drift["timeStamp"])
    logger.info("漂移窗口: %s -> %s (%d 步, CL mean=%.0f, Twb mean=%.1f)",
                t.min().date(), t.max().date(), len(drift),
                drift["TotalRealTimeLoad"].mean(), drift["OutdoorWetTemp"].mean())
    s2 = _load_split("s2")
    logger.info("遗忘检查段 S2: %d 步", len(s2))

    base_net, norm = load_policy(cfg.base_model, cfg.device)

    def ev(tag, net, env, replay_sp=None):
        net.train(False)
        m = assess(env, net, norm, device, replay_sp=replay_sp)
        logger.info("%s: E=%.4f viol=%.3f r=%.4f", tag,
                    m["spec_energy_kw_per_kw"], m["comfort_violation_rate"], m["mean_reward"])
        net.train(True)
        return m

    # ---- π₀ 基线 ----
    logger.info("=== π₀ 基线 ===")
    d0 = ev("π0 漂移窗口", base_net, CsgeEnv(drift, cal),
            replay_sp=drift[REPLAY_SP_COL].to_numpy(dtype=float))
    d0_rep = ev("rep 漂移窗口", base_net, CsgeEnv(drift, cal),
                replay_sp=drift[REPLAY_SP_COL].to_numpy(dtype=float)) if False else None
    # 操作员回放基线（net=None）
    m_rep = assess(CsgeEnv(drift, cal), None, None, device,
                   replay_sp=drift[REPLAY_SP_COL].to_numpy(dtype=float))
    logger.info("rep 漂移窗口: E=%.4f viol=%.3f", m_rep["spec_energy_kw_per_kw"], m_rep["comfort_violation_rate"])
    s2_0 = ev("π0 S2(遗忘检查)", base_net, CsgeEnv(s2, cal))

    results = {"pi0_drift": d0, "rep_drift": m_rep, "pi0_s2": s2_0}

    # ---- RFT：KL 锚定 / 无锚定 ----
    for tag, kl in [("kl1.0", 1.0), ("kl0.0", 0.0)]:
        cfg.kl_coef = kl
        net, _ = load_policy(cfg.base_model, cfg.device)
        net.train(True)
        env = CsgeEnv(drift, cal)
        logger.info("=== RFT %s (kl_coef=%.1f) ===", tag, kl)
        out = rft_finetune(net, norm, env, cfg)
        d1 = ev(f"π1_{tag} 漂移窗口", net, CsgeEnv(drift, cal))
        s2_1 = ev(f"π1_{tag} S2(遗忘检查)", net, CsgeEnv(s2, cal))
        results[f"pi1_{tag}_drift"] = d1
        results[f"pi1_{tag}_s2"] = s2_1

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = project_root() / "outputs" / "rft_csge" / f"{stamp}_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": net.state_dict(), "obs_mean": norm.mean,
                    "obs_var": norm.var, "config": cfg.__dict__,
                    "meta": {"tag": tag, "kl_coef": kl, "base": cfg.base_model}},
                   out_dir / "model.pth")
        (out_dir / "hist.json").write_text(json.dumps(out["hist"], indent=2))
        logger.info("模型: %s", out_dir / "model.pth")

    # ---- 汇总 ----
    logger.info("=== 汇总 ===")
    def sav(m, base):
        return (1 - m["spec_energy_kw_per_kw"] / base["spec_energy_kw_per_kw"]) * 100
    base = results["rep_drift"]
    logger.info("漂移窗口 省 vs 操作员: π0=%+.1f%%  π1_kl1.0=%+.1f%%  π1_kl0.0=%+.1f%%",
                sav(results["pi0_drift"], base),
                sav(results["pi1_kl1.0_drift"], base),
                sav(results["pi1_kl0.0_drift"], base))
    logger.info("S2 遗忘检查 E: π0=%.4f  π1_kl1.0=%.4f  π1_kl0.0=%.4f",
                results["pi0_s2"]["spec_energy_kw_per_kw"],
                results["pi1_kl1.0_s2"]["spec_energy_kw_per_kw"],
                results["pi1_kl0.0_s2"]["spec_energy_kw_per_kw"])


if __name__ == "__main__":
    main()
