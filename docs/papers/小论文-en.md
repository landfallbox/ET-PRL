# HVAC Predictive Control Method Based on Dynamic Event Triggering

# 1. Introduction

As global climate change intensifies, reducing energy consumption and emissions in the building sector has become an urgent priority. Statistics show that buildings account for over 35 percent of global energy consumption, and heating, ventilation, and air conditioning (HVAC) systems are responsible for about 50 percent of this building energy use and also serve as key resources for demand response where the chiller is the main energy consuming component, and its operating efficiency directly determines the overall system performance to some extent [1]. However, in real engineering scenarios, building thermal load is affected by several changing factors, including outdoor weather, occupant activity, equipment heat release and so on, while the HVAC system is nonlinear, changes over time, and has strong thermal inertia with delayed response, which makes HVAC control challenging [2].

Nowadays, deep reinforcement learning (DRL) has shown strong potential in HVAC control because it can make adaptive decisions without relying on a predefined model. By interacting directly with complex and changing environments, DRL can learn and adapt to nonlinear operating conditions [3]. Moreover, Predictive DRL further advances by using temporal features, which improves the agent's ability to anticipate future changes and helps reduce the control lag caused by the large thermal inertia of buildings [4]. However, in practical deployment, current DRL methods are still limited by the common time triggered control (TTC) scheme, where the agent samples states and updates actions at a fixed clock cycle which ignores the fact that building thermal loads do not evolve at a uniform rate over time [5]. As a result, there is an inherent trade-off between response speed and control cost: frequent updates can react quickly to disturbances, but they also increase computation cost and actuator wear, while infrequent updates reduce wear but often hurt temperature control accuracy [6].

The event triggered control (ETC) mechanism was developed to reduce the extra computation and equipment wear caused by TTC [7]. By updating the control system only when the system state changes significantly, ETC overcomes the constraint of TTC and provides a practical way to balance response speed and control cost [8]. Building on this advantage, prior studies have introduced ETC into DRL, where the control policy is updated when specific predefined events are triggered, which helps balance thermal comfort and energy consumption [9]. However, most existing ETC methods still rely on static rules set by experts, such as fixed temperature deadbands or load fluctuation thresholds [10]. Such static triggering schemes cannot adapt well to real building environments that are highly nonlinear and time varying. They also struggle to handle differences in thermal inertia across buildings and performance drift caused by equipment aging, which can lead to frequent false triggers or delayed control responses [11]. Therefore, a thresholding mechanism that can adapt to the operating state of the system is needed, so that the triggering boundary is determined not by fixed empirical rules but by an adaptive criterion based on the evolution of the deviation signal. Figure 1 compares three update strategies: the fixed interval strategy, the fixed threshold strategy, and the dynamic threshold strategy. In the top row, the room temperature deviation signal is shown together with the corresponding triggering boundaries. The fixed interval strategy updates at a constant period, whereas the fixed threshold strategy triggers an update only when the deviation exceeds a preset threshold. By contrast, the dynamic threshold strategy adjusts the triggering boundary according to the temporal variation of the deviation signal. The bottom row shows the distribution of trigger timings under the three strategies. It indicates that the dynamic threshold strategy increases the update frequency when disturbances intensify and reduces unnecessary triggers during relatively steady periods, thereby providing a better balance between control responsiveness and execution cost.

![](../pics/fig1_update_strategy_comparison.svg)

Driven by the above motivations, this paper proposes a new HVAC control framework, Event-Triggered Predictive Reinforcement Learning with Unsupervised Dynamic Event Gating (ET-PRL), that combines time series prediction with unsupervised dynamic event gating in a deep reinforcement learning setting. ET-PRL learns event-triggering rules online from operational data through a data-driven method and activates a predictive reinforcement learning policy network to generate control actions when critical events are detected, allowing control sensitivity to adapt to different building characteristics and operating conditions. By integrating predictive features, adaptive gating, and the reinforcement learning controller into a unified framework, the method preserves full control capability during major disturbances while reducing redundant updates during stable periods. The main contributions of this paper are as follows:

- **We propose an unsupervised dynamic event gating mechanism based on streaming features**: We build a gating module that combines multi scale anomaly tracking with a two layer adaptive threshold. This removes the need for static manual rules and allows the system to automatically adapt to irregular disturbances under changing operating conditions.
- **We construct a time decoupled, event driven predictive control framework**: The event gating mechanism is smoothly integrated into predictive reinforcement learning. The policy network is activated only during critical disturbances, while a zero order hold (ZOH) maintains the control output during stable periods. Through deep coupling with predictive state features, this framework effectively compensates for information loss caused by sparse on demand execution and further improves the policy's anticipatory decision making capability.
- **Quantified optimization trade off between control accuracy and execution cost**: Using 102 days of real chiller operation data, systematic comparisons, ablation and sensitivity studies show that the proposed method reduces action updates by 53.54%, lowers average daily energy consumption from 7491 to 7332 kWh/day (2.13% energy saving), and retains 94.41% of baseline performance, demonstrating robustness and practical deployability.

# 2. Related Work

Coordinated control of energy efficiency and thermal comfort in building HVAC systems has long been a major engineering challenge. To address rising building energy demand and requirements for stable operation, early studies mainly relied on classic methods such as rule-based control (RBC) and proportional-integral-derivative (PID) control [12]. Choi et al. [13] improved the practicality of HVAC rule-based control through optimization-informed rule extraction. Lee et al. [14] used deep reinforcement learning to tune PID parameters and improve tracking performance. Chojecki et al. [15] further introduced fuzzy control to improve adaptability under specific operating conditions. Tang et al. [16] proposed a physics-aware deep learning-embedded model predictive control (MPC) approach, which builds predictive models of building thermal dynamics and solves constrained optimization problems over a receding horizon to explicitly coordinate energy use, comfort, and equipment constraints. Although these methods are easy to deploy and reliable in industrial scenarios, or can achieve strong control performance when the model is accurate, they still depend on static rules, linear feedback assumptions, or high-fidelity predictive models. As a result, they have limited ability to represent nonlinear coupling, time-varying disturbances, and long thermal delays in buildings, and they often struggle to maintain both energy savings and comfort under complex operation and maintenance conditions [17].

To overcome the modeling and adaptability limits of traditional control, data-driven methods have become an important direction for intelligent HVAC control. Among them, reinforcement learning can learn optimized mappings between operating states and control actions through trial-and-error interaction with the building environment without requiring an accurate physical model, and has therefore been increasingly applied to building energy management, HVAC control, and demand response tasks [18]. Savino et al. [19] demonstrated the potential of deep reinforcement learning for low-level control in multi-zone buildings. Wang et al. [20] and Zhuang et al. [21] used DQN- and DDPG-based frameworks to learn mappings from high-dimensional sensor states to control actions, confirming the optimization benefits of reinforcement learning in complex scenarios. However, most reinforcement learning methods rely mainly on current observations, which can lead to control delay and oscillation in systems with strong thermal inertia and time lag [22]. To mitigate this issue, Li et al. [23] integrated temporal models such as LSTM into the decision process, improving awareness of system evolution and partially improving control stability; He et al. [24] further developed a model-free PRL method for chiller plant optimization by combining LSTM-based load prediction with DQN control, showing that predictive information can improve control foresight while preserving the adaptability of RL. Nevertheless, existing RL and PRL methods still generally follow the TTC paradigm at the execution level, where states are sampled and actions are updated at preset intervals. This makes it difficult to adapt the decision frequency to the intensity of load disturbances and leads to an inherent conflict between redundant computation and actuator wear [25].

As control strategies have evolved from rule-driven to data-driven methods, reducing redundant computation and actuator wear has become another key issue. Liu et al. [26] and Xue et al. [27] showed that event-triggered control (ETC) updates control and communication only when the state exceeds a threshold or a defined event occurs, thereby reducing unnecessary updates in TTC. Fu et al. [9] further proposed ED-DQN, an event-driven deep reinforcement learning control method for multi-zone residential building HVAC systems, which introduces event triggering into DRL so that the agent updates control actions only when predefined events occur, thereby reducing redundant decisions under fixed-step control. Liu et al. [28] pointed out that ETC methods in practical deployment still rely heavily on expert-defined static threshold rules. Under non-stationary factors such as drift in building thermal characteristics, climate variation, and equipment aging, fixed thresholds usually cannot balance sensitivity and stability, which can lead to excessive triggering or delayed responses and thus weaken computational efficiency and comfort performance [29].

Building on these studies, this paper reformulates HVAC event triggering as an online unsupervised anomaly detection problem and proposes ET-PRL, an on-demand framework that combines dynamic gating with predictive reinforcement learning. Compared with traditional static-rule methods and existing ETC schemes, ET-PRL introduces streaming feature tracking and adaptive thresholding to update triggering conditions dynamically, reducing dependence on manual tuning and improving adaptability to long-term drift. Unlike fixed-step reinforcement learning, the framework further decouples decision timing through event-driven execution, thereby reducing redundant inference and equipment wear while using predictive features to compensate for the loss of state information caused by sparse control. As a result, ET-PRL achieves a practical balance between control performance and execution cost under non-stationary building operating conditions.

# 3. Methodology

## 3.1 Overall Framework

Given the thermal inertia and delayed response of building HVAC systems, together with distribution shift caused by seasonal changes in operating conditions, traditional event-triggered control methods that rely on offline calibration and static thresholds usually require frequent recalibration when transferred across scenarios, which increases engineering maintenance cost. To address this issue, this paper proposes ET-PRL, an on-demand control framework that couples streaming anomaly gating with DRL.

As shown in Fig. 2, the core idea of ET-PRL is to reduce unnecessary policy inference during steady periods through event triggering while keeping the control objectives and constraints unchanged, thereby achieving a better trade-off between computational cost and control performance. Unlike traditional schemes that identify events using fixed expert rules or offline trained classifiers, the framework adopts an unsupervised online gating strategy and dynamically determines trigger timing according to the statistical deviation of the system state distribution. The overall framework consists of the following three layers.

1. **Online streaming anomaly gating mechanism**: To address concept drift during long-term operation, an anomaly detection gating module without offline training is constructed. The module tracks multi-scale statistical features of system states online and combines streaming isolation depth with an adaptive threshold strategy to generate event-triggering signals for online identification of control events.
2. **Predictive global reinforcement learning network**: For system-level energy optimization and action selection, a lightweight DQN is used to learn the control policy. This network is activated only when the gating mechanism identifies a critical dynamic event, which reduces redundant inference during stable periods.
3. **Event-driven execution and zero-order hold**: For edge deployment, the system organizes state sensing, event judgment, policy inference, and action execution into a low-latency closed-loop process. During non-triggered intervals, ZOH maintains the previous control command, thereby reducing control switching frequency and limiting adverse effects on actuators and system stability.

> **Figure 2 placeholder: Overall ET-PRL framework**

## 3.2 Dynamic Event Triggering Design Based on Streaming Feature Tracking

This paper reformulates critical event identification as an online anomaly detection problem. The main rationale is that, in control terms, critical events correspond to abrupt operating changes that require immediate policy intervention, while in statistical terms, such moments usually appear as significant deviations of state samples from the normal distribution. Therefore, event judgment can be uniformly modeled as a deviation test rather than relying on manually defined static rules.

Let the gating state be $s_t\in\mathbb{R}^d$, the event label be $e_t\in\{0,1\}$, and the normal reference distribution at time $t$ be denoted by $\mathcal{P}^{\text{norm}}_t$. The event set is defined as

$$
\mathcal{E}_t=\{s:\,d(s,\mathcal{P}^{\text{norm}}_t)>\delta_t\},\qquad
 e_t=\mathbb{I}(s_t\in\mathcal{E}_t)
$$

where $d(\cdot)$ denotes the distribution deviation metric, and $\delta_t$ is the decision threshold. This definition shows that event detection is essentially a binary test of whether the current state deviates from the normal region. The policy is updated when the state enters a high-deviation region; otherwise, the current control pace is maintained.

To convert distribution-level deviation testing into a scalar decision rule that can be updated in a streaming manner, an anomaly score function $A(s_t)$ is used as a proxy for deviation, and the set-based rule is converted into a computable threshold comparison.

$$
 e_t=\mathbb{I}(A(s_t)>\tau_t)
$$

where a larger $A(s_t)$ indicates a stronger deviation from normal behavior, and $\tau_t$ is the trigger threshold. This form has two advantages. First, it can be directly coupled with streaming statistics and updated step by step. Second, it is naturally consistent with the subsequent gating logic and enables end-to-end modeling and implementation from anomaly scoring to trigger decision.

Considering the non-stationarity caused by weather disturbances, equipment aging, and load behavior drift during long-term HVAC operation, the threshold is updated through a streaming adaptive mechanism rather than kept fixed:

$$
    \tau_t=\Phi\big(A(s_{1}),\ldots,A(s_{t-1})\big)
$$

where $\Phi(\cdot)$ denotes the threshold update operator driven by historical anomaly scores. This design is intended to balance sensitivity and stability. It increases event coverage when disturbances intensify and suppresses redundant triggering during steady periods, thereby reducing the risk of false triggers and missed triggers.

(1) Definition of Gating Input Features

To keep the physical meaning and notation consistent between event gating and control state modeling, the input to the event gating module is defined as a compact three-dimensional feature vector:

$$s_t=[Q_{load}^t,\,T_{wb}^t,\,\hat{Q}_{load}^{t+1}]$$

where $Q_{load}^t$ denotes the system cooling load at the current time step, $T_{wb}^t$ denotes the outdoor wet-bulb temperature, and $\hat{Q}_{load}^{t+1}$ denotes the one-step-ahead short-term prediction of the system cooling load.

The main reason for including $T_{wb}^t$ is that this variable reflects both ambient temperature and humidity, and directly constrains cooling tower heat transfer performance and the upper limit of system heat rejection. Therefore, $T_{wb}^t$ serves as a key state variable for characterizing the intensity and direction of exogenous weather disturbances, which improves the ability of the gating mechanism to detect non-stationary environmental changes.

In addition, $\hat{Q}_{load}^{t+1}$ is introduced because building thermal systems exhibit strong thermal inertia and control delay. If gating relies only on current observations, trigger decisions usually lag behind operating changes. By incorporating one-step-ahead load information, the gating module can sense rising or falling load trends earlier and trigger policy updates sooner during critical transitions, thereby reducing the risk of delayed response under sparse triggering.

To preserve representation capability while controlling feature dimension and supporting real-time online gating, no additional variables such as pipe network flow rate or return water temperature are included. The reason is that online anomaly detection in high-dimensional spaces is prone to sample sparsity and distance degradation, which weakens the stability of statistical discrimination and increases computational cost.

(2) Streaming Isolation Depth and Multi-Scale Anomaly Tracking

Traditional offline anomaly detection usually depends on a static sample library and is less stable under seasonal operating drift. To address this issue, a streaming isolation depth mechanism is used to score anomalies of the online gating input feature $s_t$ in real time. Streaming Isolation Depth dynamically builds and updates isolation tree structures over sliding data windows, and uses the average path length of a sample across multiple trees to measure its isolation level. A shorter path means the sample is easier to isolate and therefore more likely to be anomalous. This mechanism can adapt to distribution changes online without retraining a global model.

Each isolation tree isolates data points by recursively selecting random features and split points. For an input point $s_t$, search is performed on $\psi$ isolation trees built from the sliding reference window, and $h_j(s_t)$ denotes the path length on the $j$-th tree, where the path length is defined as the number of edges from the root node to the leaf node. The average isolation depth of the sample is defined as:

$$h_{\mathrm{iso}}(s_t) = \frac{1}{\psi} \sum_{j=1}^{\psi} h_j(s_t)$$

This quantity is further converted into a normalized anomaly score, where $m$ is the number of samples in the reference set and $C(m)$ is the corresponding baseline for average path length.

$$A_{\mathrm{iso}}(s_t) = 2^{-\frac{h_{\mathrm{iso}}(s_t)}{C(m)}}$$

The normalization constant is computed as $C(m) = 2H(m-1) - 2(m-1)/m$, where $H(m-1)=\sum_{i=1}^{m-1}\frac{1}{i}$ is the harmonic series. This formulation maps the anomaly score to the interval $(0,1)$. When a data point is easy to isolate, the path length is short and the score approaches 1. When it is hard to isolate, the path length is long and the score approaches 0.

The algorithm maintains three sliding reference windows of different sizes to form multi-scale dynamic reference sets. The short window $\mathcal{W}_{\mathrm{short}}$ contains the most recent $n_s$ data points and is used to capture high-frequency fluctuations. The medium window $\mathcal{W}_{\mathrm{medium}}$ contains the most recent $n_m$ data points and is used to represent the intraday cycle. The long window $\mathcal{W}_{\mathrm{long}}$ contains the most recent $n_l$ data points and is used to describe seasonal drift, with $n_s < n_m < n_l$. Isolation trees are applied independently to each window, and an anomaly score is produced at each scale at every time step. Let $\kappa\in\{\mathrm{short},\mathrm{medium},\mathrm{long}\}$ and let $n_\kappa$ denote the corresponding window size. The corresponding scale-specific anomaly score is given by:

$$A_{\kappa}(s_t)=2^{-\frac{h_{\mathrm{iso}}^{\kappa}(s_t)}{C(n_{\kappa})}}$$

All scale-specific scores are normalized to the interval $[0,1]$. The overall anomaly score is defined as a weighted fusion of the three scale-specific scores.

$$A(s_t)=w_{\mathrm{s}}A_{\mathrm{short}}(s_t)+w_{\mathrm{m}}A_{\mathrm{medium}}(s_t)+w_{\mathrm{l}}A_{\mathrm{long}}(s_t)$$

where $w_{\mathrm{s}}, w_{\mathrm{m}}, w_{\mathrm{l}} \ge 0$ and $w_{\mathrm{s}}+w_{\mathrm{m}}+w_{\mathrm{l}}=1$, which represent the fusion weights of the corresponding time scales.

Multi-scale fusion is used because building HVAC systems are typically influenced at the same time by short-term disturbances, intraday periodic changes, and long-term slow drift. If only short-term scores are used, the gating mechanism can respond quickly to local changes such as crowd gathering and equipment start-stop events, but it is also more sensitive to high-frequency sensor noise and normal transient fluctuations, which may increase trigger frequency. If only long-term scores are used, the historical reference baseline becomes more robust, but the response to sudden thermal disturbances may be insufficient. If only short-term and medium-term scores are combined, slow processes caused by seasonal transition, equipment aging, or drift in structural thermal resistance are still difficult to capture. Therefore, the weighted fusion of short, medium, and long time scales jointly supports disturbance detection, periodic pattern representation, and long-term baseline tracking, which improves the adaptability and robustness of the gating mechanism in non-stationary scenarios.

(3) Streaming Two-Level Adaptive Threshold Optimization

To convert continuous anomaly scores into robust control triggering decisions, a two-level adaptive dynamic threshold mechanism with local and global thresholds is constructed. If the gating system relies only on a fixed threshold or on an adaptive threshold with a single update rate, it easily faces a trade-off between sensitivity and stability. If the threshold updates too fast, it can be heavily disturbed by recent abnormal data, oscillate sharply, and mask subsequent true anomalies. If it updates too slowly, it may remain exceeded for a long period when the operating regime shifts as a whole, leading to frequent false triggers and unnecessary use of computational resources. To overcome these limitations, a coupled adaptive monitoring structure with a fast local threshold and a slow global threshold is introduced.

Let the sliding historical window of the most recent $W$ anomaly scores be denoted by $\mathcal{A}_t=\{A(s_{t-W+1}),\ldots,A(s_t)\}$. The main goal of threshold estimation is to continually extract a stable and transferable decision baseline from noisy and non-stationary streaming data. Compared with the conventional mean and standard deviation, which are sensitive to anomalous outliers, this module gives priority to a robust quantile threshold and a robust statistical threshold that reflect nonparametric distribution characteristics.

$$\tau_{q}^{(t)}=Q_q(\mathcal{A}_t), \quad\tau_{\mathrm{mad}}^{(t)}=\operatorname{med}(\mathcal{A}_t)+\kappa \cdot c \cdot \operatorname{MAD}(\mathcal{A}_t)$$

$$\operatorname{MAD}(\mathcal{A}_t)=\operatorname{med}\left(\left|a-\operatorname{med}(\mathcal{A}_t)\right|\right),\quad a\in\mathcal{A}_t$$

where $Q_q(\cdot)$ denotes the high quantile of the set, and $\operatorname{med}(\cdot)$ is the median function. The median absolute deviation, $\operatorname{MAD}$, reduces the bias introduced by potential persistent anomalies in the estimate of dispersion and helps preserve a reliable robust baseline. $\kappa$ is the sensitivity coefficient, and $c \approx 1.4826$ is the consistency constant used to ensure asymptotically unbiased estimation. The rationale is that the quantile estimate provides a robust anchor for threshold location, while $\operatorname{MAD}$ provides a robust correction for scale. Their combination constrains threshold distortion caused by both location shift and scale inflation. Based on this, the local candidate threshold is obtained by a weighted fusion of the two terms.

$$\tau_{\mathrm{cand}}^{(t)}=\omega_q\tau_q^{(t)}+(1-\omega_q)\tau_{\mathrm{mad}}^{(t)}$$

where $\omega_q \in [0,1]$ is the weighting coefficient. This coefficient controls the trade-off between quantile-dominant and scale-correction-dominant estimation. A larger $\omega_q$ places more emphasis on robust location, whereas a smaller $\omega_q$ makes the threshold more sensitive to changes in dispersion. Because a one-step regime shift may introduce high-frequency transient fluctuations, the system does not directly use a single candidate value. Instead, a first-order discrete low-pass filter is introduced for local smoothing.

$$\tau_{\mathrm{local}}^{(t)}=(1-\lambda_{\mathrm{local}})\tau_{\mathrm{local}}^{(t-1)}+\lambda_{\mathrm{local}}\tau_{\mathrm{cand}}^{(t)}$$

where $\lambda_{\mathrm{local}} \in (0,1]$ is the local smoothing update rate. It suppresses short-term noise oscillations while allowing the fast-varying threshold to track regime switching quickly. A larger $\lambda_{\mathrm{local}}$ gives more weight to new samples and helps reduce threshold tracking lag, whereas a smaller $\lambda_{\mathrm{local}}$ strengthens suppression of short-term impulse noise. In this way, the local threshold is responsible for fast adaptation to the current operating condition. Based on the local threshold, the global threshold acts as a historical steady-state reference and is updated by exponential smoothing with stronger hysteresis.

$$\tau_{\mathrm{global}}^{(t)}=(1-\lambda_{\mathrm{g}})\tau_{\mathrm{global}}^{(t-1)}+\lambda_{\mathrm{g}}\tau_{\mathrm{local}}^{(t)}$$

where $\lambda_{\mathrm{g}} \ll \lambda_{\mathrm{local}}$, so the global threshold has a much stronger lagged smoothing property. This cascaded dual-time-constant design acts as a buffer when local features change sharply, enabling stable tracking of deeper system evolution caused by long-term equipment aging and seasonal migration, while reducing the risk that the threshold is dragged by short anomaly segments.

Finally, the basic adaptive gating threshold is constructed by coupling the two threshold signals as follows:

$$\tilde{\tau}_t=\alpha_{\mathrm{local}}\tau_{\mathrm{local}}^{(t)}+(1-\alpha_{\mathrm{local}})\tau_{\mathrm{global}}^{(t)}+b_{\mathrm{bias}}$$

where $\alpha_{\mathrm{local}}\in[0,1]$ controls the fusion ratio between the local and global thresholds. A larger $\alpha_{\mathrm{local}}$ emphasizes fast response to local operating changes, whereas a smaller $\alpha_{\mathrm{local}}$ emphasizes consistency with the long-term baseline and trigger stability. $b_{\mathrm{bias}}$ is a bias term. During highly stable periods, such as low-load night hours, the system variance and $\operatorname{MAD}$ may converge to very small values, causing the threshold to shrink excessively. The bias term sets an absolute lower bound for triggering and thus prevents frequent false triggers caused by an overly small threshold.

# References

[1] Arghand, Taha, et al. "Individually controlled localized chilled beam combined with chilled ceiling: Thermal environment." Building and Environment 282 (2025): 113322.

[2] Wu, Zeqing, et al. "AE-TD3 with adaptive expert guidance: towards responsive deep reinforcement learning for building HVAC control systems." Energy and Buildings (2025): 116744.

[3] Xia, Yihan, et al. "Federated accelerated deep reinforcement learning for multi-zone HVAC control in commercial buildings." IEEE Transactions on Smart Grid 16.3 (2025): 2599-2610.

[4] Wu, Zeqing, et al. "AE-TD3 with adaptive expert guidance: towards responsive deep reinforcement learning for building HVAC control systems." Energy and Buildings (2025): 116744.

[5] Xue, Zhouzhou, Zhaoxu Yu, and Shugang Li. "Event-triggered adaptive neural control for uncertain nontriangular nonlinear systems with time-varying delays." International Journal of Control, Automation and Systems 20.12 (2022): 4090-4099.

[6] Coraci, Davide, et al. "An innovative heterogeneous transfer learning framework to enhance the scalability of deep reinforcement learning controllers in buildings with integrated energy systems." Building simulation. Vol. 17. No. 5. Beijing: Tsinghua University Press, 2024.

[7] Gu, Zhou, Ruiyan Cao, and Engang Tian. "Reinforcement learning-based event-triggered optimal control of power systems with control input saturation." IEEE Transactions on Industrial Informatics 21.2 (2024): 1528-1536.

[8] Wang, Ke, Zhuo Tang, and Chaoxu Mu. "Dynamic event-triggered model-free reinforcement learning for cooperative control of multiagent systems." IEEE Transactions on Reliability 74.3 (2024): 3166-3179.

[9] Fu, Qiming, et al. "ED-DQN: An event-driven deep reinforcement learning control method for multi-zone residential buildings." Building and Environment 242 (2023): 110546.

[10] Li, Wenzhuo, Hangxin Li, and Shengwei Wang. "An event-driven multi-agent based distributed optimal control strategy for HVAC systems in IoT-enabled smart buildings." Automation in Construction 132 (2021): 103919.

[11] Wang, Xin, et al. "Observer-based event-triggered optimal control for nonlinear multiagent systems with input delay via reinforcement learning strategy." IEEE Transactions on Emerging Topics in Computational Intelligence 9.3 (2024): 2398-2409.

[12] Chaya, P., et al. "Human-Centric Smart Energy Optimization and Automation System." 2025 3rd International Conference on Intelligent Cyber Physical Systems and Internet of Things (ICoICI). IEEE, 2025.

[13] Choi, Youngsik, et al. "Optimization-informed rule extraction for HVAC system: A case study of dedicated outdoor air system control in a mixed-humid climate zone." Energy and Buildings 295 (2023): 113295.

[14] Lee, Dongkyu, Jinhwa Jeong, and Young Tae Chae. "Application of deep reinforcement learning for proportional–integral–derivative controller tuning on air handling unit system in existing commercial building." Buildings 14.1 (2023): 66.

[15] Chojecki, Adrian, Arkadiusz Ambroziak, and Piotr Borkowski. "Fuzzy controllers instead of classical PIDs in HVAC equipment: Dusting off a well-known technology and Today's implementation for better energy efficiency and user comfort." Energies 16.7 (2023): 2967.

[16] Tang, Lingfeng, et al. "Deeply flexible commercial building HVAC system control: A physics-aware deep learning-embedded MPC approach." Applied Energy 388 (2025): 125631.

[17] Lu, Shengze, et al. "Exploring the comprehensive integration of artificial intelligence in optimizing HVAC system operations: A review and future outlook." Results in Engineering 25 (2025): 103765.

[18] Fu, Qiming, et al. "Applications of reinforcement learning for building energy efficiency control: A review." Journal of Building Engineering 50 (2022): 104165.

[19] Savino, Sabrina, et al. "Deploying deep reinforcement learning for low-level HVAC control in multi-zone buildings: A comparative study with ASHRAE G36 sequences." Energy and Buildings (2025): 116456.

[20] Wang, Man, and Borong Lin. "MF^2: Model-free reinforcement learning for modeling-free building HVAC control with data-driven environment construction in a residential building." Building and Environment 244 (2023): 110816.

[21] Zhuang, Dian, et al. "Data-driven predictive control for smart HVAC system in IoT-integrated buildings with time-series forecasting and reinforcement learning." Applied Energy 338 (2023): 120936.

[22] Manjavacas, Antonio, et al. "An experimental evaluation of deep reinforcement learning algorithms for HVAC control." Artificial Intelligence Review 57.7 (2024): 173.

[23] Li, Kai, Wei Ni, and Falko Dressler. "LSTM-characterized deep reinforcement learning for continuous flight control and resource allocation in UAV-assisted sensor network." IEEE Internet of Things Journal 9.6 (2021): 4179-4189.

[24] Al Sayed, Khalil, et al. "Reinforcement learning for HVAC control in intelligent buildings: A technical and conceptual review." Journal of Building Engineering 95 (2024): 110085.

[25] He, Kun, et al. "Predictive control optimization of chiller plants based on deep reinforcement learning." Journal of Building Engineering 76 (2023): 107158.

[26] Liu, Xinghua, et al. "Event-triggered load frequency control of smart grids under deception attacks." IET Control Theory & Applications 15.10 (2021): 1335-1345.

[27] Xue, Zhouzhou, Zhaoxu Yu, and Shugang Li. "Event-triggered adaptive neural control for uncertain nontriangular nonlinear systems with time-varying delays." International Journal of Control, Automation and Systems 20.12 (2022): 4090-4099.

[28] Liu, Derong, et al. "Adaptive dynamic programming for control: A survey and recent advances." IEEE Transactions on Systems, Man, and Cybernetics: Systems 51.1 (2020): 142-160.

[29] Wang, Yuan, et al. "Dynamic event-triggered control for persistent dwell-time switched nonlinear multiagent systems with random packet loss." IEEE Transactions on Systems, Man, and Cybernetics: Systems 54.4 (2023): 2045-2054.
