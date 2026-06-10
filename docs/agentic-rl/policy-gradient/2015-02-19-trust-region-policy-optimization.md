# Trust Region Policy Optimization (TRPO)

Date: 2015-02-19  
Course: agentic-rl-course  
Topic: policy-gradient  
Paper: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel / ICML 2015  
Link: https://arxiv.org/abs/1502.05477

## Why this paper today?

TRPO is the direct descendant of the Policy Gradient Theorem that solved the step-size problem. It introduced **constrained policy updates** — an idea that PPO later simplified and that RLHF inherits. Understanding TRPO explains why RLHF uses KL penalties, not gradient clipping.

## Core problem

Policy gradient methods are sensitive to step size. Too small → slow convergence. Too large → the policy collapses, sample efficiency drops, and the training destabilizes. Standard gradient descent with a fixed learning rate does not respect the **geometry** of the policy parameter space.

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

## Key results

- Demonstrated monotonic improvement on robotic locomotion (swimming, hopping, walking) and Atari games with neural network policies.
- Robust across tasks with minimal hyperparameter tuning — the KL constraint $\delta$ is far easier to set than a learning rate.
- The vine variant outperformed single-path on harder tasks due to better advantage estimates.

## Limitations

- The KL constraint is on **average** KL, not the pointwise bound from theory — the monotonic guarantee is approximate in practice.
- Conjugate gradient + line search adds implementation complexity.
- Second-order optimization (Fisher-vector products) is computationally expensive for very large models.
- Does not scale directly to LLM-sized policies (billions of parameters) — this is why PPO was adopted for RLHF instead.

## Connection to this course

TRPO's constrained update philosophy lives on in every modern RL algorithm used for agent training. PPO (next in our reading list) is the simplified first-order approximation. In RLHF, the KL penalty in the PPO objective (keeping the fine-tuned LM close to the reference model) is a direct descendant of TRPO's trust region — preventing the policy from collapsing after a bad reward signal.

## Notes for future reading

- PPO (Schulman et al., 2017) — replaces the constraint with a clipped surrogate; far simpler, equally effective.
- Natural Policy Gradient (Kakade, 2002) — the algorithmic precursor using Fisher information as the metric.
- GAE (Schulman et al., 2016) — how advantage estimation interacts with trust region methods.
- RLHF reward model training — why separate reward models and KL penalties are needed when applying these algorithms to language models.
