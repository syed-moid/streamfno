# StreamFNO — Theoretical Framework

Working document. Destined for `docs/theory.md` in the `streamfno` repository.

Status: informal theorem statements, drafted late July 2026; positioning, T2 reformulation, and literature anchors added early August 2026. Not yet proved.

**One-sentence positioning:**

> *We derive a telemetry-identifiable reflected continuum limit for many-partition streaming queues and establish an estimator-independent upper bound on the lead time over which impending backpressure can be predicted from noisy, sampled broker telemetry.*

**Author's risk ranking (Aug 2026).** T1 is the highest-risk claim: classical fluid limits do not automatically imply an advection–diffusion–reaction PDE, so the continuum coordinate and the empirical-measure (hydrodynamic) limit must be proved, not asserted — modeling the distribution of normalized lag is the most direct route to "drift = netput, diffusion = variance rate." T2 must be phrased as boundary hitting / loss of viability, not PDE blow-up (see §2b). T3 is the strongest conceptual contribution and requires a fully specified loss, information set, model class, and information-theoretic proof — empirical collapse of powerful predictors is supporting evidence, never proof. T4 is publishable only as an end-to-end engineering theorem; a bare Fourier-truncation bound followed by "therefore 16 modes" is incremental.

> **Note on structure.** Sections 1–5 are the author's formulation. Section 6 is a review pass with open technical issues that must be resolved before proofs are attempted. Section 6 is intended to be split out into `docs/open-issues.md` once the repo exists.

---

## 1. Positioning against the fluid-limit literature

The literature supports three distinct approximations, and they are not interchangeable:

1. **Fluid / law-of-large-numbers limits** (Kurtz; Chen–Mandelbaum) — produce deterministic ODEs or reflected flow equations.
2. **Heavy-traffic diffusion limits** (Harrison–Reiman; Reiman; Chen–Mandelbaum) — produce reflected Brownian motion or related stochastic processes.
3. **Hydrodynamic / continuum limits** — require an *additional* limit over the number and spacing of components before a PDE appears.

Chen and Mandelbaum's Theorem 5.1 proves uniform-on-compact convergence of scaled discrete networks to a reflected fluid process

$$Z = X + (I - P^\top)Y.$$

Their Theorem 5.2 shows that after a finite time, non-bottleneck inventories vanish at fluid scale while bottleneck inventories evolve linearly. This is a strong foundation for reflection, bottleneck identification, and finite-time regime changes — **but it does not by itself establish an advection–diffusion–reaction PDE.**

### The key modeling decision

The cleanest route to a defensible coefficient interpretation is to let the PDE coordinate denote **normalized partition lag or queue occupancy**, *not* physical partition number. Then:

$$\text{advection velocity} = \text{mean netput rate} = \lambda - \mu,$$

$$\text{diffusion coefficient} = \tfrac{1}{2}\lim_{\Delta t \downarrow 0}\frac{\operatorname{Var}(\Delta A - \Delta S)}{\Delta t}.$$

The resulting PDE is a **nonlinear Fokker–Planck equation for the empirical distribution of partition lags**. This is mathematically natural under many-partition scaling and far more defensible for Kafka than imposing an artificial Euclidean geometry on partitions.

---

## 2. Theorem 1 — Many-partition reflected Fokker–Planck limit (informal)

Consider a sequence of streaming systems indexed by $N$, where system $N$ contains $N$ partitions. Let $Q_i^N(t)$ denote the backlog or consumer lag of partition $i$, and $B_i^N$ its effective buffer or backpressure threshold. Define normalized occupancy

$$X_i^N(t) = \frac{Q_i^N(t)}{B_i^N} \in [0,1],$$

and the empirical lag-density measure

$$\nu_t^N = \frac{1}{N}\sum_{i=1}^{N}\delta_{X_i^N(t)}.$$

**Assumptions.**

- Arrivals, services, retries, and throttling transitions form a density-dependent Markov system.
- Their conditional rates are uniformly bounded and locally Lipschitz in the empirical state and admissible control $u_t$.
- Partitions are exchangeable within broker or workload classes, or satisfy an appropriate weak-dependence condition.
- Normalized one-step queue increments vanish while their first two conditional moments converge.
- The limiting PDE has a unique weak solution.
- Queue emptiness and finite-buffer saturation are represented by reflecting, sticky, absorbing, or regulated boundary conditions.

**Conclusion.** For each fixed $T$ preceding any loss of well-posedness,

$$\nu^N \Longrightarrow \rho(x,t)\,dx \quad \text{in } D\big([0,T], \mathcal{P}([0,1])\big),$$

in probability, where $\rho$ is the unique solution of

$$\frac{\partial \rho}{\partial t} = -\frac{\partial}{\partial x}\big[b(x, m_t, u_t)\rho\big] + \frac{1}{2}\frac{\partial^2}{\partial x^2}\big[a(x, m_t, u_t)\rho\big] + R(x, \rho, m_t, u_t).$$

**Coefficients.** Aggregate broker-load statistics are collected by

$$m_t = \int_0^1 \phi(x)\rho(x,t)\,dx,$$

the local netput drift is

$$b(x,m,u) = \lim_{\Delta t \downarrow 0}\frac{\mathbb{E}[\Delta A - \Delta S \mid x, m, u]}{\Delta t},$$

and the netput variance rate is

$$a(x,m,u) = \lim_{\Delta t \downarrow 0}\frac{\operatorname{Var}(\Delta A - \Delta S \mid x, m, u)}{\Delta t}.$$

Under conditionally independent arrival and service counting processes, $a = \sigma_A^2 + \sigma_S^2$; otherwise the arrival–service covariance must be retained.

**Reaction term.** $R$ must correspond to an identifiable mechanism — retry amplification, partition activation, quota-induced service degradation, or shared-resource contention. It is not a free fitting term.

**Backpressure observable.** Represented by the upper-boundary flux

$$J_B(t) = \Big[b\rho - \tfrac{1}{2}\partial_x(a\rho)\Big]_{x=1},$$

or equivalently by the associated boundary regulator $K_B(t)$.

Empirical telemetry estimators of $b$, $a$, and the parameters of $R$ are consistent under ergodic sampling and sufficiently fine telemetry resolution.

### 2.1 Refinement — do not conflate the two coordinate choices

If the PDE coordinate instead represents **broker or partition topology**, the advection and deterministic diffusion coefficients come from the first and second *spatial* moments of routing, migration, or rebalancing transitions. Arrival and service variance then appears mainly in the stochastic fluctuation correction — typically a linearized SPDE:

$$\sqrt{N}\big(\nu^N - \rho\big) \Longrightarrow Z.$$

**This distinction must be explicit in the paper.** Otherwise reviewers will object that primitive counting-process variance has been incorrectly converted into spatial diffusion.

---

## 2b. Theorem 2 — Inevitability of backpressure as a viability theorem (reformulated)

**Decision (Aug 2026): T2 is a viability / controlled-reachability statement, not a blow-up statement.** Kafka queues have finite limits and throttling; literal finite-time blow-up is usually a modeling artifact unless a retry mechanism produces a genuinely superlinear reaction. The natural formulation:

Let $\mathcal{U}$ be the class of admissible controls (consumer scaling, partition reassignment, quota throttling), each with its actuation-delay constraint. Define the **safe set**

$$\mathcal{S}_\epsilon = \big\{\rho \in \mathcal{P}([0,1]) : J_B(t) \le \epsilon \big\},$$

and say backpressure is *inevitable from state $\rho_0$ within horizon $T$* if for **every** $u \in \mathcal{U}$ the controlled flow started at $\rho_0$ exits $\mathcal{S}_\epsilon$ before $T$.

**T2 (informal).** Characterize the *viability kernel* $\mathrm{Viab}_T(\mathcal{S}_\epsilon)$ — the set of states from which some admissible control keeps the system safe through $T$ — and give a computable inner/outer approximation of its complement (the *unavoidable set*) in terms of telemetry-identifiable quantities: current lag density, netput drift $b$, variance rate $a$, actuation delays, and control authority bounds.

Notes:

- This connects to Aubin's viability theory and to Hamilton–Jacobi reachability. HJ reachability machinery is finite-dimensional, so a practical route is a moment closure or spectral truncation of the PDE (which T4 justifies), followed by level-set computation on the reduced state.
- Operationally this is the most valuable theorem in the paper: it tells an operator *when prediction has ceased to matter because the incident is already committed*, and it defines the safe operating envelope for capacity planning.
- It also cleanly resolves the well-posedness tension in §6.3: with the regulator/boundary-flux framing fixed, $R$ stays tame and "blow-up" never needs to be invoked.

---

## 3. Theorem 3 — Information-limited backpressure prediction horizon (informal)

Let telemetry be observed at sampling interval $\Delta$:

$$Y_k = \mathcal{H}\rho(k\Delta) + \eta_k, \qquad \eta_k \sim \mathcal{N}(0, R).$$

Define the backpressure event at lead time $h$:

$$E_h = \mathbf{1}\Big\{\sup_{s \in [t, t+h]} K_B(s) - K_B(t) > 0\Big\}.$$

A predictor is any measurable function of the telemetry history $\mathcal{Y}_t = \sigma(Y_k : k\Delta \le t)$. Define the intrinsic prediction horizon at tolerated error $\delta < 1/2$:

$$H_\delta^* = \sup\Big\{h : \inf_{\widehat{E}_h(\mathcal{Y}_t)} \Pr(\widehat{E}_h \ne E_h) \le \delta\Big\}.$$

### 3.1 Lyapunov / unstable-direction version

Suppose the nonlinear PDE flow has a finite-time unstable direction with expansion rate at least $\lambda_+ > 0$. Assume two admissible states separated initially by approximately

$$d_0(h) = m\,e^{-\lambda_+ h}$$

produce opposite backpressure outcomes at lead time $h$, where $m$ is a computable event-boundary margin. Let

$$\mathcal{G}_{\Delta,R} = \sum_{k\Delta \le t}\big\|R^{-1/2}\mathcal{H}\Phi(t, k\Delta)e\big\|^2$$

be the telemetry information (observability) Gramian in the unstable direction $e$. A two-point information bound gives

$$\inf_{\widehat{E}_h}\Pr(\widehat{E}_h \ne E_h) \ge \frac{1}{2}\left(1 - \frac{m}{2}e^{-\lambda_+ h}\sqrt{\mathcal{G}_{\Delta,R}}\right),$$

so a necessary condition for error at most $\delta$ is

$$h \le \frac{1}{\lambda_+}\log\left[\frac{m\sqrt{\mathcal{G}_{\Delta,R}}}{2(1-2\delta)}\right],$$

yielding the computable upper bound

$$H_\delta^* \le \frac{1}{\lambda_+}\log\left[\frac{m\sqrt{\mathcal{G}_{\Delta,R}}}{2(1-2\delta)}\right]_+.$$

Higher observation noise decreases $\mathcal{G}_{\Delta,R}$; slower sampling supplies fewer terms in the Gramian; a larger positive Lyapunov exponent shortens the horizon.

### 3.2 Spectral-gap version

For an ergodic stochastic model, suppose the hidden Markov or PDE semigroup satisfies a strong data-processing inequality with rate $\gamma > 0$:

$$I(E_h; \mathcal{Y}_t) \le e^{-2\gamma h} I(X_t; \mathcal{Y}_t).$$

Then

$$\inf_{\widehat{E}_h}\Pr(\widehat{E}_h \ne E_h) \ge \frac{1}{2}\left[1 - e^{-\gamma h}\sqrt{\frac{I(X_t;\mathcal{Y}_t)}{2}}\right],$$

and therefore

$$H_\delta^* \le \frac{1}{\gamma}\log\left[\frac{\sqrt{I(X_t;\mathcal{Y}_t)/2}}{1-2\delta}\right]_+.$$

### 3.3 Two things that must not be confused

**$\gamma$ is the spectral gap of the state evolution semigroup** — not the Kafka graph Laplacian, not the telemetry Fourier spectrum, not the FNO weight matrix. These are different quantities.

**Empirical training cannot establish an impossibility result.** It can only test whether practical predictors approach the theoretical ceiling. Any oracle used in validation must have the *same observation σ-algebra* as the predictors covered by the theorem. An oracle given the true hidden state is not subject to the bound.

---

## 4. Theorem 4 — End-to-end error decomposition

The publishable contribution is not the Fourier tail inequality. It is an end-to-end rule for selecting $K$ that accounts for aliasing, telemetry resolution, FNO approximation, rollout amplification, and a compute constraint. The decomposition:

$$\text{total forecast error} \le \underbrace{\text{finite-partition error}}_{\text{T1 rate}} + \underbrace{\text{PDE-model error}}_{\text{misspecification}} + \underbrace{\text{Fourier truncation error}}_{\text{spectral decay}} + \underbrace{\text{learned-operator error}}_{\text{FNO approximation}}$$

If measured telemetry satisfies $|\widehat{\rho}_k| \le C|k|^{-s}$, the classical Fourier tail gives (in one dimension)

$$\|\rho - P_K\rho\|_{L^2} \lesssim K^{1/2-s},$$

with the exponent adjusted for dimension and Sobolev regularity.

---

## 5. Novelty memo — intrinsic limits of backpressure forecasting

Backpressure forecasting is usually treated as an empirical time-series problem: collect broker metrics, select a prediction window, compare forecasting models. This leaves two fundamental questions unanswered. First, why should the telemetry field of a large streaming platform obey a low-dimensional evolution law? Second, what is the longest lead time at which *any* predictor — not merely a particular neural network — can retain useful information about an impending backpressure event?

This work answers both by connecting stochastic queueing limits, continuum modeling, information-theoretic predictability, and neural operators. The central object is the empirical distribution of normalized lag across a large number of partitions. Each partition is modeled as a finite-capacity queue whose arrival, service, retry, and throttling intensities depend on observable workload variables, control actions, and aggregate broker utilization. Under many-partition and small-increment scaling, the empirical lag distribution converges to a nonlinear reflected Fokker–Planck equation. In this representation, advection is the measured arrival-minus-service drift, diffusion is the measured netput variance rate, and the reaction term represents explicitly modeled feedback such as retry amplification or capacity degradation.

**This step goes beyond the classical fluid-limit literature.** Kurtz establishes convergence of density-dependent Markov chains to finite-dimensional deterministic dynamics and develops diffusion corrections. Chen and Mandelbaum establish sample-path fluid limits and bottleneck decompositions for discrete flow networks. Harrison, Reiman, and Chen–Mandelbaum establish reflected diffusion approximations in heavy traffic. **None of these results alone yields a PDE indexed by partition lag.** The novel theoretical bridge is the empirical-measure limit over many interacting queues, including finite-buffer regulation and telemetry-identifiable coefficients. Related continuum-limit work shows that large Markov networks can converge to PDEs, but does not address backpressure, finite queue boundaries, streaming telemetry, or intrinsic prediction horizons.

Chen–Mandelbaum supplies two particularly useful ingredients. Its oblique reflection representation gives the correct mathematical treatment of queue non-negativity and work conservation. Its bottleneck theorem shows that non-bottleneck fluid queues drain while bottleneck queues persist or grow after a finite transient — which suggests the continuum model should retain a regulator or boundary-flux variable rather than describing backpressure as unconstrained PDE blow-up. The paper also warns that bottleneck identity can depend on service discipline, and that multi-commodity extensions are difficult. Kafka topics, quotas, consumer groups, and broker assignments are effectively multi-class, so the manuscript must either model several workload classes or state a defensible aggregation theorem.

**The second contribution is the intrinsic prediction-horizon theorem.** Rather than defining predictability as the point where an FNO's test accuracy falls below a chosen threshold, the work defines the optimal Bayes or minimax error over all causal predictors using the same telemetry. For unstable nonlinear dynamics, two initial states can be exponentially close in observation space yet reach opposite sides of the backpressure boundary after lead time $h$. Le Cam or Bayesian-risk inequalities convert telemetry noise and sampling resolution into an estimator-independent error lower bound. For stochastic systems with a spectral gap, strong data-processing inequalities quantify the rate at which current telemetry loses information about a future event. The resulting horizon is computable from an instability or mixing rate, the observation operator, noise covariance, sampling interval, and distance to the backpressure boundary.

This is materially different from the familiar heuristic $H \approx \lambda_{\max}^{-1}\log(\epsilon/\epsilon_0)$. The proposed result specifies the prediction target, the predictor information set, the loss function, the admissible model class, and the statistical lower-bound argument. It can therefore support the strong claim that no predictor with the stipulated telemetry achieves a specified accuracy beyond the computed horizon.

**The FNO is positioned as a fast approximation of the continuum solution operator** — not as the source of the physical model, and not as the source of the predictability result. Existing work already establishes FNO universality and several PDE-specific approximation bounds, so novelty must come from the end-to-end decomposition in §4.

**Validation should mirror the theory.** A simulator should first demonstrate convergence of finite-$N$ empirical lag distributions toward the PDE. Telemetry-derived drift and variance coefficients should be compared against event-level ground truth. Backpressure should be evaluated as boundary hitting or boundary flux, not only as pointwise lag error. The intrinsic horizon should be compared against FNO, transformer, state-space, queueing-model, and Bayes-filter baselines. Kafka experiments should vary partition count, broker count, workload burstiness, service heterogeneity, quotas, and rebalance behavior. **The strongest empirical figure will show forecast error curves crossing the theorem-derived impossibility region at approximately the predicted lead time.**

---

## 6. Review pass — open technical issues

Ranked by how likely each is to be raised by a referee at a probability-literate venue. None is fatal; all need resolving before proofs.

### 6.1 T1 — the double scaling limit is not stated ⚠️ highest priority

The assumption "normalized one-step queue increments vanish while their first two conditional moments converge" is doing enormous work and hides a second limit.

**The issue.** $N \to \infty$ (many partitions) and the diffusive rescaling *within* each partition are two different limits, and the order and relative rate matter. A pure mean-field limit with $N \to \infty$ alone yields a **first-order transport (Vlasov/Liouville) equation with no diffusion term** — individual noise averages out across partitions. To retain a genuine second-order Fokker–Planck at the empirical-measure level, each partition must keep its own diffusive fluctuation in the limit.

Concretely: if $Q_i^N$ jumps by $\pm 1$ at rate $O(1)$ and $B_i^N = O(1)$, then $X_i = Q_i/B_i$ has $O(1)$ jumps and is a *jump process*, not a diffusion. Getting a diffusion requires $B_i^N \to \infty$ with jump rates scaling like $(B_i^N)^2$ — i.e. **a heavy-traffic scaling per partition, jointly with the many-partition scaling.**

**Required fix.** State the joint scaling regime explicitly — $N$, $B^N$, and $\lambda^N$ — together with their relative rates, and either prove or cite the diffusive rescaling that makes normalized occupancy converge to a diffusion. Without it, T1 delivers a transport equation and the $a(x,m,u)$ term never appears.

**Alternative worth considering.** Keep the jumps and target a *nonlocal* jump-diffusion Fokker–Planck with an explicit jump kernel. Arguably more faithful to Kafka's discrete batch semantics, at the cost of weakening the clean "diffusion = netput variance" story. Decide deliberately rather than by default.

### 6.2 T1 — partitions are not globally exchangeable

The interaction is written through a scalar $m_t$, which is a global mean-field coupling. But partitions on the *same broker* share a disk, page cache, and network interface — the dominant interaction is **broker-local, not cluster-global**.

**Fix.** Make this a multi-class (or graphon) mean-field limit where the class is the broker, and $m_t$ becomes a vector of per-broker aggregates $m_t^{(j)}$. The document already gestures at this ("within broker or workload classes") but the theorem statement doesn't carry it. This is also exactly the multi-commodity difficulty Chen–Mandelbaum warn about, so addressing it head-on strengthens the positioning.

### 6.3 T1 — well-posedness versus the physics of retry amplification — ✅ RESOLVED

"The limiting PDE has a unique weak solution" is listed as a hypothesis, but there was a tension: $R$ models retry amplification, which is superlinear — precisely the mechanism that causes blow-up.

**Resolution (Aug 2026): the boundary-regulator framing is adopted, and T2 is reformulated as a viability theorem (§2b).** Backpressure is the boundary flux $J_B$ / regulator $K_B$; $R$ is kept tame enough (bounded-but-steep) for global well-posedness; inevitability of backpressure is expressed as exit from a viability kernel under every admissible control, not as loss of well-posedness. The phrase "$T$ preceding any loss of well-posedness" in T1 should be removed accordingly.

### 6.4 T3 — the linearization is applied exactly where it fails ⚠️ main vulnerability

The Gramian uses $\Phi(t, k\Delta)$, the state-transition operator of the *linearized* flow. But the premise of the theorem is exponential divergence at rate $\lambda_+$ over horizon $h$. A referee will observe that linearization is valid only while trajectories stay close — which is precisely not the regime the theorem is about.

**Options, in order of preference:**

1. **Lead with the SDPI version (§3.2).** It needs no linearization and is the more robust argument. Demote the Lyapunov version to an interpretable special case with explicitly stated local validity.
2. **Semi-empirical two-point bound.** Exhibit two *actual* admissible PDE initial conditions with opposite outcomes and compute their observation KL numerically. Theoretically weaker, but honest and directly checkable — and it produces a figure.
3. Keep the linearized bound but state and numerically validate its regime of validity.

Also state the Le Cam chain explicitly as a lemma: $\inf \Pr(\text{err}) \ge \tfrac12(1 - \mathrm{TV})$, with $\mathrm{TV} \le \sqrt{\mathrm{KL}/2}$ (Pinsker) and $\mathrm{KL} \approx \tfrac12 d_0^2 \mathcal{G}$ under the Gaussian-observation/LAN assumption. That chain reproduces the stated bound, but the LAN assumption is currently implicit.

### 6.5 T3 — the margin $m$ is load-bearing and undefined

$m$ is the distance from the current state to the separatrix between "backpressure by $h$" and "no backpressure by $h$." Computing it requires characterizing the basin boundary of a nonlinear PDE flow — not trivial. If $m$ is large the bound is loose; if the state sits near the boundary, $m \to 0$ and the bound says nothing is predictable (true, but useless).

**The headline figure depends entirely on $m$ being computable.** Define it precisely, and characterize its typical magnitude numerically in the Kafka setting early. This is the top risk in T3.

### 6.6 T3 — the two versions apply in different regimes, not to the same system

A system with a positive Lyapunov exponent in the relevant direction is *not* mixing toward a stationary measure in that direction. Unstable-direction and spectral-gap/ergodic assumptions are in tension. Present §3.1 and §3.2 as covering **different operating regimes** (near-critical/unstable versus stable/ergodic), not as two proofs of one theorem. Say which regime Kafka is in, and when.

Also: $E_h$ is a functional of the *limiting PDE* through $K_B$, while predictors observe *finite-$N$* telemetry. Specify which system's event is being predicted, or carry the finite-$N$ approximation term — §4's decomposition already has the right slot for it.

For the SDPI, the chain is $\mathcal{Y}_t \to X_t \to X_{t+h} \to E_h$, so data-processing plus semigroup contraction gives the result. The exponential form $e^{-2\gamma h}$ needs a citation (Polyanskiy–Wu; Raginsky) and holds under conditions worth stating.

### 6.7 T4 — boundary layers may break spectral decay ⚠️ measure this early

$\rho$ is a density on $[0,1]$ with a **regulated/reflecting boundary at $x=1$**. Regulated boundaries characteristically produce kinks or boundary layers in the density, and boundary layers destroy spectral accuracy: $s$ becomes small, and Fourier truncation converges slowly.

This matters far beyond T4 — **if $s$ is small, the premise that an FFT-based operator is the right tool is undermined**, and a Chebyshev or cosine basis (Neumann-appropriate, given reflection) may dominate. Note also that with reflecting boundaries the natural basis is cosine, not complex exponential.

**Recommendation:** measure the empirical spectral decay $s$ of the lag density from the simulator in Phase 2 and from real Kafka in Phase 3, and treat it as a go/no-go for the FNO framing. This is cheap, early, and could save months.

### 6.8 Minor

- Notation: $\Longrightarrow$ conventionally denotes weak convergence in distribution, but mean-field limits to a deterministic limit give convergence *in probability*. The text says both. Pick one and be consistent; also specify the topology on $\mathcal{P}([0,1])$ (weak or Wasserstein-$p$).
- The Fourier tail exponent checks out: $\sum_{|k|>K}|\widehat\rho_k|^2 \approx \int_K^\infty k^{-2s}dk = K^{1-2s}/(2s-1)$, so the $L^2$ norm is $\sim K^{1/2-s}$. ✓
- Define $\phi$ in $m_t = \int \phi \rho$ concretely for at least one worked case, so the reader sees what aggregate is meant.

---

## 7. Immediate technical to-do

1. Write the joint scaling regime for T1 ($N$, $B^N$, $\lambda^N$ and relative rates). Nothing else in T1 can be proved until this exists.
2. Decide: regulator framing or blow-up framing. Write one sentence committing to it.
3. Promote the SDPI version of T3 to primary.
4. Define the margin $m$ precisely and estimate it numerically for one concrete Kafka-like configuration.
5. Measure the empirical spectral decay $s$ as soon as the simulator produces lag densities.
6. Find a collaborator in applied probability. Items 1–3 are standard *to a specialist* and treacherous otherwise.

---

## 8. Literature anchors (reviewed, July–Aug 2026)

### Fluid and diffusion limits of queueing networks

- Kurtz, T. G. (1976). *Limit Theorems and Diffusion Approximations for Density Dependent Markov Chains.* Mathematical Programming Study, 5, 67–78. [link](https://link.springer.com/chapter/10.1007/BFb0120765)
- Chen, H., & Mandelbaum, A. (1991). *Discrete Flow Networks: Bottleneck Analysis and Fluid Approximations.* Mathematics of Operations Research, 16(2), 408–446. [link](https://pubsonline.informs.org/doi/abs/10.1287/moor.16.2.408) — Thm 5.1 (reflected fluid limit), Thm 5.2 (bottleneck decomposition).
- Chen, H., & Mandelbaum, A. (1991). *Stochastic Discrete Flow Networks: Diffusion Approximations and Bottlenecks.* Annals of Probability, 19(4), 1463–1519.
- Harrison, J. M., & Reiman, M. I. (1981). *Reflected Brownian Motion on an Orthant.* Annals of Probability, 9(2), 302–308.
- Reiman, M. I. (1984). *Open Queueing Networks in Heavy Traffic.* Mathematics of Operations Research, 9(3), 441–458. [link](https://pubsonline.informs.org/doi/abs/10.1287/moor.9.3.441)
- Aoki Hillas, L., Caldentey, R., & Gupta, V. (2024). *Heavy Traffic Analysis of Multi-Class Bipartite Queueing Systems Under FCFS.* Queueing Systems. [link](https://link.springer.com/article/10.1007/s11134-024-09903-4) — relevant to the multi-class/broker-class structure in §6.2.

### Continuum / hydrodynamic limits of large networks

- Zhang, Y., Chong, E. K. P., Hannig, J., & Estep, D. (2013). *Approximating Extremely Large Networks via Continuum Limits.* IEEE Access, 1. — **the closest prior for "network → PDE"; the related-work section must position against it explicitly** (differs by: no finite-buffer boundaries, no backpressure observable, no telemetry identifiability, no prediction-horizon theory).

### T1 proof machinery — to obtain (gap in current list)

- Sznitman, A.-S. (1991). *Topics in Propagation of Chaos.* — the standard route to empirical-measure limits of interacting particle systems; Sznitman (1984) also treats **reflected** McKean–Vlasov diffusions, directly relevant to the $[0,1]$ boundary.
- Graham, C., & Méléard, S. — mean-field limits for interacting jump processes (relevant if the jump-diffusion alternative of §6.1 is chosen).
- Oelschläger, K. (1984). *A Martingale Approach to the Law of Large Numbers for Weakly Interacting Stochastic Processes.*
- Kolokoltsov, V. — nonlinear Markov processes and kinetic equations.
- Dai, J. G. (1995). *On Positive Harris Recurrence of Multiclass Queueing Networks* — fluid-limit stability arguments, useful for T2's safe-set characterization.

### T2 machinery — to obtain

- Aubin, J.-P. — *Viability Theory.*
- Mitchell, I., Bayen, A., & Tomlin, C. (2005). *A Time-Dependent Hamilton–Jacobi Formulation of Reachable Sets* — computable unavoidable sets for the moment-closed/truncated system.

### Predictability and information-theoretic lower bounds (T3)

- Boffetta, G., Cencini, M., Falcioni, M., & Vulpiani, A. (2002). *Predictability: A Way to Characterize Complexity.* Physics Reports, 356(6), 367–474. [arXiv](https://arxiv.org/abs/nlin/0101029) — the heuristic $H \approx \lambda^{-1}\log(\epsilon/\epsilon_0)$ that T3 formalizes; cite and contrast.
- Han, Y., Jana, S., & Wu, Y. (2023). *Optimal Prediction of Markov Chains With and Without Spectral Gap.* IEEE Transactions on Information Theory. — **directly usable for the spectral-gap version (§3.2)**; note their prediction risk is over the next-step distribution, so the extension to event prediction at lead $h$ is part of T3's contribution.
- Esposito, A. R., Vandenbroucque, A., & Gastpar, M. (2024). *Lower Bounds on the Bayesian Risk via Information Measures.* JMLR, 25. — generalizes the Le Cam/Fano route; may give tighter constants than Pinsker.
- Polyanskiy, Y., & Wu, Y. — *Information Theory: From Coding to Learning* — strong data-processing inequalities (needed to justify $e^{-2\gamma h}$ in §3.2).
- Le Cam; Tsybakov — two-point and minimax lower-bound techniques (§6.4's explicit lemma chain).

### Neural operators (T4)

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). *Fourier Neural Operator for Parametric PDEs.* ICLR. [arXiv](https://arxiv.org/abs/2010.08895)
- Kovachki, N., Lanthaler, S., & Mishra, S. (2021). *On Universal Approximation and Error Bounds for Fourier Neural Operators.* JMLR, 22(290). — supplies the learned-operator term in §4.
- Lanthaler, S., Stuart, A. M., & Trautner, M. (2024). *Discretization Error of Fourier Neural Operators.* [arXiv:2405.02221](https://arxiv.org/abs/2405.02221) — supplies the aliasing/discretization term in §4; together these two make the end-to-end decomposition assemblable from cited parts plus the T1 rate.
