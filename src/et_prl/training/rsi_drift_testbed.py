import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, torch
from et_prl.training.ppo_csge_train import _load_split, _load_calibration, assess, REPLAY_SP_COL
from et_prl.training.rft_csge import RftConfig, rft_finetune, load_policy
from et_prl.environments.drift_env import DriftEnv
from et_prl.environments.csge_env import ACTION_SPACE

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cpu')
cal = _load_calibration('calibration_spring.json')
data = _load_split('s12_deploy')
t = pd.to_datetime(data['timeStamp'])
WIN = 7*24*12; NWIN = 16
data = data.iloc[:NWIN*WIN].reset_index(drop=True)
t = t.iloc[:NWIN*WIN]

DRIFT = dict(b1_0=-0.3111, b1_1=-0.4611, b2_0=0.0165, b2_1=0.0315)
TRIG_REL = 0.01  # 潜漂移触发：策略比操作员差 >1%（相对参考基线）
MARGIN = 0.001

pi0, norm = load_policy('outputs/ppo_csge/20261014_160715/model_best.pth', 'cpu')
cur = pi0
hist_E = []   # 历史动态 E（用于退化触发）

def week_env(k, span=1):
    s, e = k*WIN, min((k+span)*WIN, len(data))
    sub = data.iloc[s:e].reset_index(drop=True)
    return DriftEnv(sub, cal, drift=DRIFT, t0_frac=k/NWIN, t_span=span/NWIN), sub

def oracle_E(sub, t0, tspan):
    """每步取可行集内功率最低的动作（完美跟踪漂移的参考）。"""
    env2 = DriftEnv(sub, cal, drift=DRIFT, t0_frac=t0, t_span=tspan)
    tot_p = tot_cl = 0.0
    for i in range(len(sub)):
        cl = float(sub['TotalRealTimeLoad'].iloc[i]); twb = float(sub['OutdoorWetTemp'].iloc[i])
        mask = env2.feasible_mask(cl)
        best = None
        for a, sp in enumerate(ACTION_SPACE):
            if not mask[a]: continue
            p = env2._simulate(cl, twb, float(sp))['sum_p']
            if best is None or p < best: best = p
        if cl > 1: tot_p += best; tot_cl += cl
    return tot_p/tot_cl if tot_cl else 0.0

rows = []
n_keep = n_rb = 0
for k in range(NWIN):
    env, wk = week_env(k)
    rep_sp = wk[REPLAY_SP_COL].to_numpy(dtype=float)
    m_dyn = assess(env, cur, norm, device)
    m_sta = assess(week_env(k)[0], pi0, norm, device)
    m_rep = assess(week_env(k)[0], None, None, device, replay_sp=rep_sp)
    E_dyn = m_dyn['spec_energy_kw_per_kw']
    E_rep = m_rep['spec_energy_kw_per_kw']
    # 潜漂移触发：策略比操作员(参考基线)差 >TRIG_REL
    triggered = E_dyn > E_rep * (1 + TRIG_REL)
    hist_E.append(E_dyn)
    gate = ''
    if triggered and k < NWIN - 1:
        # RFT：近2周 DriftEnv 训练
        tr_env, tr_sub = week_env(k-2, span=2)
        cfg = RftConfig(kl_coef=0.0, total_steps=400000, n_epochs=32, lr=3e-4, entropy_coef=0.02)
        net, _ = load_policy('outputs/ppo_csge/20261014_160715/model_best.pth', 'cpu')
        net.load_state_dict(cur.state_dict()); net.train(True)
        rft_finetune(net, norm, None, cfg, ref_state=pi0.state_dict(), env_obj=tr_env)
        net.train(False)
        m_new = assess(week_env(k)[0], net, norm, device)
        e_best = min(E_dyn, m_sta['spec_energy_kw_per_kw'])
        keep = m_new['spec_energy_kw_per_kw'] < e_best * (1 - MARGIN)
        gate = 'KEEP' if keep else 'ROLLBACK'
        if keep: cur = net; n_keep += 1
        else: n_rb += 1
    # oracle
    orc = oracle_E(wk, k/NWIN, 1/NWIN)
    ts_star_end = week_env(k)[0].ts_star()
    rows.append(dict(win=k, start=str(t.iloc[k*WIN].date()),
        ts_star=round(ts_star_end,2),
        E_dyn=round(E_dyn,4), E_sta=round(m_sta['spec_energy_kw_per_kw'],4), gap=round(E_dyn-E_rep,4),
        E_rep=round(m_rep['spec_energy_kw_per_kw'],4), E_orc=round(orc,4),
        trigger=int(triggered), gate=gate,
        sav_dyn=round((1-E_dyn/m_rep['spec_energy_kw_per_kw'])*100,2)))
    print('win %2d %s ts*=%.2f | E_dyn=%.4f E_sta=%.4f E_rep=%.4f E_orc=%.4f | %s %s | 省dyn=%+.1f%%' % (
        k, rows[-1]['start'], ts_star_end, E_dyn, m_sta['spec_energy_kw_per_kw'],
        m_rep['spec_energy_kw_per_kw'], orc, 'TRIG' if triggered else '    ', gate,
        rows[-1]['sav_dyn']), flush=True)

df = pd.DataFrame(rows)
df.to_csv('outputs/rsi_drift_16w_big.csv', index=False)
print()
print('=== 汇总 (16周合成漂移测试床) ===')
print('动态 π_k : E mean=%.4f  省 vs 操作员 = %+.2f%%' % (df.E_dyn.mean(), df.sav_dyn.mean()))
print('静态 π₀  : E mean=%.4f  省 vs 操作员 = %+.2f%%' % (
    (1-df.E_sta/df.E_rep).mean()*100, 0))
print('操作员   : E mean=%.4f' % df.E_rep.mean())
print('oracle   : E mean=%.4f  (完美跟踪漂移的下界)' % df.E_orc.mean())
print('RFT 触发: %d/%d 周 | KEEP %d / ROLLBACK %d' % (df.trigger.sum(), NWIN, n_keep, n_rb))
print('dyn 输 sta 的周数: %d/%d' % ((df.E_dyn>df.E_sta).sum(), NWIN))
