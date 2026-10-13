import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, torch
from collections import deque
from et_prl.training.ppo_csge_train import _load_split, _load_calibration, assess, REPLAY_SP_COL
from et_prl.training.rft_csge import RftConfig, rft_finetune, load_policy
from et_prl.environments.csge_env import CsgeEnv

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cpu')
cal = _load_calibration()
s1 = _load_split('s1'); s0 = _load_split('s0')
cl_ref = s1['TotalRealTimeLoad'].values; twb_ref = s1['OutdoorWetTemp'].values
t = pd.to_datetime(s0['timeStamp'])
WIN = 7*24*12; n_win = len(s0)//WIN
TAU_REL = 0.5
MARGIN = 0.001   # 门控相对余量 0.1%

def kl_binned(x, ref, n_bins=10):
    lo, hi = 0, max(x.max(), ref.max())
    edges = np.linspace(lo, hi, n_bins+1)
    p,_ = np.histogram(x, edges); q,_ = np.histogram(ref, edges)
    p=(p+1)/(p.sum()+n_bins); q=(q+1)/(q.sum()+n_bins)
    return float(np.sum(p*np.log(p/q)))

pi0, norm = load_policy('outputs/ppo_csge/20261012_231148/model_best.pth', 'cpu')
cur = pi0
recent = deque(maxlen=2)

rows = []
n_keep = n_rb = 0
for k in range(n_win):
    s, e = k*WIN, (k+1)*WIN
    wk = s0.iloc[s:e].reset_index(drop=True)
    rep_sp = wk[REPLAY_SP_COL].to_numpy(dtype=float)
    m_dyn = assess(CsgeEnv(wk, cal), cur, norm, device)
    m_sta = assess(CsgeEnv(wk, cal), pi0, norm, device)
    m_rep = assess(CsgeEnv(wk, cal), None, None, device, replay_sp=rep_sp)
    if len(recent) >= 1:
        ref = pd.concat(list(recent), ignore_index=True)
        drift = (kl_binned(wk['TotalRealTimeLoad'].values, ref['TotalRealTimeLoad'].values)
                 + kl_binned(wk['OutdoorWetTemp'].values, ref['OutdoorWetTemp'].values))
    else:
        drift = (kl_binned(wk['TotalRealTimeLoad'].values, cl_ref)
                 + kl_binned(wk['OutdoorWetTemp'].values, twb_ref))
    triggered = drift > TAU_REL and k < n_win - 1 and len(recent) >= 1
    gate = ''
    if triggered:
        # 训练：近2周；验证：本周 wk（holdout）
        train_data = pd.concat(list(recent), ignore_index=True)
        cfg = RftConfig(kl_coef=1.0, total_steps=24576, n_epochs=4)
        net, _ = load_policy('outputs/ppo_csge/20261012_231148/model_best.pth', 'cpu')
        net.load_state_dict(cur.state_dict()); net.train(True)
        rft_finetune(net, norm, CsgeEnv(train_data, cal), cfg, ref_state=pi0.state_dict())
        net.train(False)
        m_new = assess(CsgeEnv(wk, cal), net, norm, device)
        # 门控：必须同时优于 当前策略 和 静态 π₀
        e_best = min(m_dyn['spec_energy_kw_per_kw'], m_sta['spec_energy_kw_per_kw'])
        v_best = min(m_dyn['comfort_violation_rate'], m_sta['comfort_violation_rate'])
        keep = (m_new['spec_energy_kw_per_kw'] < e_best * (1 - MARGIN)
                and m_new['comfort_violation_rate'] <= v_best + 1e-9)
        gate = 'KEEP' if keep else 'ROLLBACK'
        if keep:
            cur = net; n_keep += 1
        else:
            n_rb += 1
    recent.append(wk)
    rows.append(dict(win=k, start=str(t.iloc[s].date()), twb=round(float(wk['OutdoorWetTemp'].mean()),1),
        cl=round(float(wk['TotalRealTimeLoad'].mean()),0), drift=round(drift,3),
        trigger=int(triggered), gate=gate,
        E_dyn=round(m_dyn['spec_energy_kw_per_kw'],4),
        E_sta=round(m_sta['spec_energy_kw_per_kw'],4), E_rep=round(m_rep['spec_energy_kw_per_kw'],4),
        sav_dyn=round((1-m_dyn['spec_energy_kw_per_kw']/m_rep['spec_energy_kw_per_kw'])*100,2),
        sav_sta=round((1-m_sta['spec_energy_kw_per_kw']/m_rep['spec_energy_kw_per_kw'])*100,2)))
    print('win %2d %s twb=%4.1f cl=%4.0f drift_rel=%.3f %s %s | E_dyn=%.4f E_sta=%.4f E_rep=%.4f | 省 dyn=%+.1f%% sta=%+.1f%%' % (
        k, rows[-1]['start'], rows[-1]['twb'], rows[-1]['cl'], drift,
        'RFT' if triggered else '   ', gate,
        rows[-1]['E_dyn'], rows[-1]['E_sta'], rows[-1]['E_rep'],
        rows[-1]['sav_dyn'], rows[-1]['sav_sta']), flush=True)

df = pd.DataFrame(rows)
df.to_csv('outputs/rsi_s0_gated.csv', index=False)
print()
print('=== 汇总 (S0 部署段, 相对漂移 + 验证门) ===')
print('动态 π_k : E=%.4f  平均省 vs 操作员 = %+.2f%%' % (df.E_dyn.mean(), df.sav_dyn.mean()))
print('静态 π₀  : E=%.4f  平均省 vs 操作员 = %+.2f%%' % (df.E_sta.mean(), df.sav_sta.mean()))
print('操作员   : E=%.4f' % df.E_rep.mean())
print('RFT 触发: %d/%d 周 | 门控: KEEP %d / ROLLBACK %d' % (df.trigger.sum(), n_win, n_keep, n_rb))
trig = df[df.trigger==1]; non = df[df.trigger==0]
if len(trig): print('触发周   : dyn 比 sta 多省 %+.2f%%' % ((trig.sav_dyn - trig.sav_sta).mean()))
if len(non): print('非触发周 : dyn 比 sta 多省 %+.2f%%' % ((non.sav_dyn - non.sav_sta).mean()))
print('dyn 输 sta 的周数: %d/%d' % ((df.sav_dyn < df.sav_sta).sum(), n_win))
