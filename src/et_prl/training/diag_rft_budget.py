import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, torch
from et_prl.training.ppo_csge_train import _load_split, _load_calibration, assess
from et_prl.training.rft_csge import RftConfig, rft_finetune, load_policy
from et_prl.environments.drift_env import DriftEnv
from et_prl.environments.csge_env import ACTION_SPACE

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cpu')
cal = _load_calibration('calibration_spring.json')
data = _load_split('s12_deploy')
WIN = 7*24*12; NWIN = 16
data = data.iloc[:NWIN*WIN].reset_index(drop=True)
DRIFT = dict(b1_0=-0.3111, b1_1=-0.4611, b2_0=0.0165, b2_1=0.0315)
pi0, norm = load_policy('outputs/ppo_csge/20261014_160715/model_best.pth', 'cpu')

def week_env(k, span=1):
    s, e = k*WIN, min((k+span)*WIN, len(data))
    sub = data.iloc[s:e].reset_index(drop=True)
    return DriftEnv(sub, cal, drift=DRIFT, t0_frac=k/NWIN, t_span=span/NWIN), sub

K = 12
tr_env, tr_sub = week_env(K-2, span=2)
va_env, va_sub = week_env(K)
m0 = assess(va_env, pi0, norm, device)

def avg_action(net, sub, k):
    env = DriftEnv(sub, cal, drift=DRIFT, t0_frac=k/NWIN, t_span=1/NWIN)
    obs, info = env.reset(); acts = []
    for i in range(len(sub)):
        obs_t = torch.as_tensor(norm.normalize(obs), dtype=torch.float32)
        with torch.no_grad():
            a, _, _, _ = net.act(obs_t, mask=torch.as_tensor(info['action_mask'], dtype=torch.bool))
        acts.append(float(ACTION_SPACE[int(a.item())]))
        obs, _, term, trunc, info = env.step(int(a.item()))
        if term or trunc: obs, info = env.reset()
    return np.mean(acts)

print('π₀: 验证周E=%.4f  设定=%.2f  (ts*=%.2f)' % (m0['spec_energy_kw_per_kw'], avg_action(pi0, va_sub, K), va_env.ts_star()))
for steps, epochs, kl, ent in [(196608, 16, 0.0, 0.02), (393216, 32, 0.0, 0.02), (393216, 32, 0.0, 0.05)]:
    net, _ = load_policy('outputs/ppo_csge/20261014_160715/model_best.pth', 'cpu')
    net.train(True)
    cfg = RftConfig(kl_coef=kl, total_steps=steps, n_epochs=epochs, entropy_coef=ent, lr=3e-4)
    rft_finetune(net, norm, None, cfg, ref_state=pi0.state_dict(), env_obj=tr_env)
    net.train(False)
    m_va = assess(DriftEnv(va_sub, cal, drift=DRIFT, t0_frac=K/NWIN, t_span=1/NWIN), net, norm, device)
    a = avg_action(net, va_sub, K)
    print('RFT(%dk,%dep,ent=%.2f): 验证周E=%.4f (π0=%.4f, 省%+.2f%%) | 设定=%.2f' % (
        steps//1000, epochs, ent, m_va['spec_energy_kw_per_kw'], m0['spec_energy_kw_per_kw'],
        (1-m_va['spec_energy_kw_per_kw']/m0['spec_energy_kw_per_kw'])*100, a), flush=True)
