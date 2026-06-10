# Trust Region Policy Optimization (TRPO)

Date: 2015-02-19  
Course: agentic-rl-course  
Topic: policy-gradient  
Paper: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel / ICML 2015  
Link: https://arxiv.org/abs/1502.05477

## Core problem

Policy gradient methods are sensitive to step size. Too small → slow convergence. Too large → the policy collapses, sample efficiency drops, and training destabilizes. Standard gradient descent with a fixed learning rate does not respect the **geometry** of the policy parameter space.

## Historical context

By 2015, the Policy Gradient Theorem (Sutton et al., 2000) was well-established, and deep RL had just seen breakthroughs with DQN (Mnih et al., 2013/2015). But DQN was value-based. Policy gradient methods — REINFORCE, actor-critic — could handle stochastic policies and continuous actions, yet were notoriously brittle. Natural Policy Gradient (Kakade, 2002) proposed using the Fisher information matrix to account for policy geometry, but required expensive second-order computations. The field needed a practical algorithm that combined the stability of natural gradients with the scalability of deep neural networks.

## Main idea

Enforce a **trust region** on each update: the new policy must stay close to the old policy in KL divergence. This guarantees monotonic improvement (in theory) and stable training (in practice). The approach is a minorization-maximization (MM) algorithm: construct a surrogate objective that lower-bounds the true return, then maximize it under a KL constraint.

## Technical details

**Key identity** (Kakade & Langford, 2002):

$$
\eta(\tilde{\pi}) = \eta(\pi) + \sum_s \rho_{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a|s) A_\pi(s,a)
$$

where $\rho_{\tilde{\pi}}$ is the discounted visitation frequency under $\tilde{\pi}$. The local approximation $L_\pi(\tilde{\pi})$ replaces $\rho_{\tilde{\pi}}$ with $\rho_\pi$ — it matches $\eta$ to first order.

**Monotonic improvement bound (Theorem 1):**

$$
\eta(\pi_{\text{new}}) \geq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - \frac{4\epsilon\gamma}{(1-\gamma)^2} \alpha^2
$$

where $\alpha = D_{TV}^{\max}(\pi_{\text{old}}, \pi_{\text{new}})$. Using $D_{TV}^2 \leq D_{KL}$:

$$
\eta(\tilde{\pi}) \geq L_\pi(\tilde{\pi}) - C \cdot D_{KL}^{\max}(\pi, \tilde{\pi}), \quad C = \frac{4\epsilon\gamma}{(1-\gamma)^2}
$$

**Practical algorithm** — replace the penalty with a **hard constraint** (trust region):

$$
\max_\theta \; L_{\theta_{\text{old}}}(\theta) \quad \text{s.t.} \quad \bar{D}_{KL}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) \leq \delta
$$

The constraint is on the **average** KL (not max), estimated from samples. Optimized via conjugate gradient for the natural gradient direction + line search.

**Two sampling variants:**
- **Single-path**: standard trajectory rollouts (model-free).
- **Vine**: generates rollouts from a set of states, branching multiple actions per state (lower variance, needs state reset).

## Research takeaway

Constraining policy updates with KL divergence — rather than trusting a fixed learning rate — is the key to stable deep RL training. The theoretical monotonic improvement guarantee is approximate in practice, but the empirical stability it provides is transformative.

## Influence

TRPO was the bridge between theoretical natural gradient methods and practical deep RL. Its direct descendants include:
- **PPO** (Schulman et al., 2017) — replaces the second-order KL constraint with a first-order clipped surrogate, making it far simpler and equally effective.
- **ACKTR** (Wu et al., 2017) — uses Kronecker-factored approximation to scale natural gradients.
- **MPO** (Abdolmaleki et al., 2018) — applies trust regions in the context of maximum a posteriori policy optimization.
- **RLHF** — the KL penalty in the PPO-ptx objective used by InstructGPT and subsequent LLM alignment methods is a direct instantiation of TRPO's trust region idea: keep the fine-tuned model close to the reference model.

TRPO also set a standard for rigorous evaluation in RL — its experimental methodology (multiple random seeds, standardized environments, careful hyperparameter reporting) became the norm.

## Modern perspective

TRPO's main practical limitation is complexity: conjugate gradient + Fisher-vector products + line search is heavy compared to PPO's simple clipping. This is why PPO, not TRPO, became the standard for both deep RL and LLM alignment.

The average-KL constraint also deviates from the theory (which requires pointwise $D_{KL}^{\max}$), meaning the monotonic guarantee is approximate. In practice, the constraint threshold $\delta$ still needs tuning — though much less than a learning rate.

For LLM-scale models, second-order methods remain impractical, and PPO's clipped surrogate dominates. However, TRPO's core insight — that constraining policy divergence prevents collapse — is now embedded in almost every modern policy optimization algorithm, including methods that don't explicitly compute KL. Its legacy is more conceptual than algorithmic: **update the policy, but don't let it stray too far.**
