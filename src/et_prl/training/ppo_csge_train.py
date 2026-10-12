"""PPO（Categorical）训练器：CsgeEnv 校准仿真环境（第二篇 next-paper）。

替代原 DQN 方案：13 档离散供水温度设定动作，actor-critic + GAE + clip。
- 训练段 S1（27 天），部署段 S2（106 天）
- 内置基线：恒定设定值策略（中间档 8.5℃）、操作员回放（数据真实设定值）
- 产物：outputs/ppo_csge/<timestamp>/model_final.pth + model_best.pth + metrics.jsonl

用法：
    PYTHONPATH=src python3 -m et_prl.training.ppo_csge_train --total-steps 5000000
    PYTHONPATH=src python3 -m et_prl.training.ppo_csge_train --quick   # 冒烟测试
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

from et_prl.config.base import project_root
from et_prl.environments.csge_env import ACTION_SPACE, CLEANED_DIR, CsgeEnv
from et_prl.utils.reproducibility import configure_reproducibility

REPLAY_SP_COL = "ChChWTempSupplySetPoint01"


@dataclass
class PpoCsgeConfig:
    # 数据
    train_split: str = "s1"
    test_split: str = "s2"
    # PPO 超参
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    hidden: tuple = (256, 128)
    rollout_len: int = 4096
    n_epochs: int = 4
    minibatch_size: int = 512
    # 预算
    total_steps: int = 5_000_000
    assess_every_steps: int = 500_000
    # 其他
    seed: int = 42
    device: str = "cpu"
    out_root: str = "outputs/ppo_csge"


class RunningMeanStd:
    """观测在线归一化（Welford 批量形式）。"""

    def __init__(self, dim: int) -> None:
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, xs: np.ndarray) -> None:
        xs = np.asarray(xs, dtype=np.float64)
        if xs.ndim == 1:
            xs = xs[None, :]
        batch_mean = xs.mean(axis=0)
        batch_var = xs.var(axis=0)
        batch_count = xs.shape[0]
        self.count += batch_count
        delta = batch_mean - self.mean
        self.var = self.var + (batch_var - self.var) + delta**2 * batch_count / self.count
        self.mean = self.mean + delta * batch_count / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.actor = nn.Linear(last, n_actions)
        self.critic = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.actor(z), self.critic(z).squeeze(-1)

    def act(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(CLEANED_DIR / "split" / f"{name}.csv")


def _load_calibration() -> dict:
    return json.loads((CLEANED_DIR / "calibration.json").read_text())


def assess(
    env: CsgeEnv,
    net: ActorCritic | None,
    norm: RunningMeanStd | None,
    device: torch.device,
    replay_sp: np.ndarray | None = None,
) -> dict:
    """整段评估。net=None 且 replay_sp 给定 -> 操作员回放基线；
    net=None 且 replay_sp=None -> 恒定中间档基线；net 给定 -> 贪心动作。"""
    obs, _ = env.reset()
    tot_r = 0.0
    n = 0
    e_num = 0.0
    e_den = 0.0
    viol = 0
    run_n = 0
    act_hist = np.zeros(len(ACTION_SPACE), dtype=int)
    cfg = env.cfg
    while True:
        if replay_sp is not None:
            sp = float(replay_sp[n % len(replay_sp)])
            if not np.isfinite(sp):
                sp = float(ACTION_SPACE[len(ACTION_SPACE) // 2])
            a = int(np.argmin(np.abs(ACTION_SPACE - sp)))
        elif net is not None:
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(
                    norm.normalize(obs), dtype=torch.float32, device=device))
            a = int(logits.argmax().item())
        else:
            a = len(ACTION_SPACE) // 2
        obs, r, term, trunc, info = env.step(a)
        tot_r += r
        n += 1
        act_hist[a] += 1
        cl = info["cl"]
        if cl > 1.0:
            run_n += 1
            e_num += info["sum_p"]
            e_den += cl
            hi = cfg.comfort_hi(cl)
            if info["t_supply"] < cfg.comfort_lo or info["t_supply"] > hi:
                viol += 1
        if term or trunc:
            break
    return {
        "mean_reward": tot_r / n,
        "spec_energy_kw_per_kw": e_num / e_den if e_den > 0 else float("nan"),
        "comfort_violation_rate": viol / run_n if run_n else float("nan"),
        "n_steps": n,
        "action_dist": (act_hist / n).tolist(),
    }


def _save_model(path: Path, net: ActorCritic, norm: RunningMeanStd,
                cfg: PpoCsgeConfig, meta: dict) -> None:
    torch.save({
        "model_state": net.state_dict(),
        "obs_mean": norm.mean,
        "obs_var": norm.var,
        "config": asdict(cfg),
        "meta": meta,
    }, path)


def train(cfg: PpoCsgeConfig) -> Path:
    configure_reproducibility(cfg.seed)
    device = torch.device(cfg.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root() / cfg.out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ppo_csge")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(out_dir / "train.log", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    metrics_path = out_dir / "metrics.jsonl"

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    logger.info(f"输出目录: {out_dir}")
    logger.info(f"配置: {asdict(cfg)}")

    cal = _load_calibration()
    data = _load_split(cfg.train_split)
    test_data = _load_split(cfg.test_split)
    env = CsgeEnv(data, cal)
    test_env = CsgeEnv(test_data, cal)
    replay_sp = test_data[REPLAY_SP_COL].to_numpy(dtype=float)
    logger.info(f"训练段 {cfg.train_split}: {len(data)} 步; 部署段 {cfg.test_split}: {len(test_data)} 步")

    net = ActorCritic(env.observation_size, env.action_size, cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    norm = RunningMeanStd(env.observation_size)

    B = cfg.rollout_len
    obs_buf = np.zeros((B, env.observation_size), np.float32)
    act_buf = np.zeros(B, np.int64)
    logp_buf = np.zeros(B, np.float32)
    rew_buf = np.zeros(B, np.float32)
    done_buf = np.zeros(B, np.float32)
    val_buf = np.zeros(B, np.float32)

    obs, _ = env.reset()
    global_step = 0
    update_no = 0
    best_score = -float("inf")
    t_start = time.time()

    while global_step < cfg.total_steps:
        # ---------- 采集 ----------
        last_value = 0.0
        for i in range(B):
            obs_buf[i] = obs
            obs_t = torch.as_tensor(norm.normalize(obs), dtype=torch.float32, device=device)
            with torch.no_grad():
                a, logp, _, value = net.act(obs_t)
            act_buf[i] = int(a.item())
            logp_buf[i] = float(logp.item())
            val_buf[i] = float(value.item())
            next_obs, r, term, trunc, _ = env.step(act_buf[i])
            done = bool(term or trunc)
            rew_buf[i] = r
            done_buf[i] = 1.0 if done else 0.0
            if done:
                last_value = 0.0
                obs, _ = env.reset()
            else:
                with torch.no_grad():
                    _, _, _, nv = net.act(torch.as_tensor(
                        norm.normalize(next_obs), dtype=torch.float32, device=device))
                last_value = float(nv.item())
            global_step += 1

        norm.update(obs_buf)

        # ---------- GAE ----------
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
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret_buf, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ---------- 更新 ----------
        total_frac = 0.0
        n_minibatch = 0
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                logits, value = net(obs_t[idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[idx])
                ratio = torch.exp(new_logp - logp_t[idx])
                s1 = ratio * adv_t[idx]
                s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t[idx]
                loss_pi = -torch.min(s1, s2).mean()
                loss_v = nn.functional.mse_loss(value, ret_t[idx])
                loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                total_frac += float((ratio - 1.0).abs().mean().item())
                n_minibatch += 1

        update_no += 1
        mean_reward = float(rew_buf.mean())
        elapsed = time.time() - t_start
        sps = global_step / elapsed
        logger.info(
            f"update {update_no} | step {global_step}/{cfg.total_steps} | "
            f"mean_r {mean_reward:.4f} | frac {total_frac / n_minibatch:.4f} | "
            f"{sps:.0f} steps/s | eta {(cfg.total_steps - global_step) / max(sps, 1e-6) / 60:.1f}min"
        )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "update": update_no, "step": global_step, "mean_reward": mean_reward,
                "frac": total_frac / n_minibatch, "steps_per_sec": sps,
            }) + "\n")

        # ---------- 部署段评估 ----------
        if global_step % cfg.assess_every_steps < B or global_step >= cfg.total_steps:
            net.training = False
            m_ppo = assess(test_env, net, norm, device)
            m_replay = assess(test_env, None, None, device, replay_sp=replay_sp)
            m_const = assess(test_env, None, None, device)
            net.training = True
            logger.info(
                f"[S2 评估] PPO: r={m_ppo['mean_reward']:.4f} "
                f"E={m_ppo['spec_energy_kw_per_kw']:.4f} kW/kW "
                f"viol={m_ppo['comfort_violation_rate']:.3f} | "
                f"操作员回放: r={m_replay['mean_reward']:.4f} "
                f"E={m_replay['spec_energy_kw_per_kw']:.4f} kW/kW "
                f"viol={m_replay['comfort_violation_rate']:.3f} | "
                f"恒定8.5: r={m_const['mean_reward']:.4f} "
                f"E={m_const['spec_energy_kw_per_kw']:.4f} kW/kW"
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"assess": "s2", "step": global_step,
                                    "ppo": m_ppo, "replay": m_replay, "const": m_const},
                                   default=float) + "\n")
            if m_ppo["mean_reward"] > best_score:
                best_score = m_ppo["mean_reward"]
                _save_model(out_dir / "model_best.pth", net, norm, cfg,
                            {"step": global_step, "s2_mean_reward": best_score})
                logger.info(f"  -> 保存 model_best (S2 r={best_score:.4f})")

    _save_model(out_dir / "model_final.pth", net, norm, cfg,
                {"step": global_step, "s2_mean_reward": best_score})
    logger.info(f"训练完成，共 {global_step} 步，用时 {(time.time() - t_start) / 60:.1f} min")
    logger.info(f"模型: {out_dir / 'model_final.pth'}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 训练 CsgeEnv（S1 训练 / S2 部署评估）")
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-split", default="s1")
    parser.add_argument("--test-split", default="s2")
    parser.add_argument("--assess-every", type=int, default=500_000)
    parser.add_argument("--quick", action="store_true", help="冒烟测试（4096 步）")
    args = parser.parse_args()

    cfg = PpoCsgeConfig()
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.train_split = args.train_split
    cfg.test_split = args.test_split
    if args.quick:
        cfg.total_steps = 4096
        cfg.assess_every_steps = 2048
    else:
        cfg.total_steps = args.total_steps
        cfg.assess_every_steps = args.assess_every
    train(cfg)


if __name__ == "__main__":
    main()
