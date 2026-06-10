# Policy Gradient Methods for Reinforcement Learning with Function Approximation

Date: 1999-12-03 (NeurIPS 1999)  
Course: agentic-rl-course  
Topic: policy-gradient  
Paper: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour / NeurIPS 1999  
Link: https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation

## Core problem

Value-function-based RL (Q-learning, Sarsa, dynamic programming) with function approximation has no convergence guarantees — small changes in estimated values can cause discontinuous policy switches, making the optimization path unstable. The field needed a principled way to optimize policies directly.

## Historical context

Throughout the 1990s, RL research was dominated by the value-function approach: approximate a value function, then derive a greedy policy from it. But Bertsekas & Tsitsiklis (1996), Baird (1995), and Gordon (1995/1996) showed that Q-learning and Sarsa can fail to converge even for simple MDPs with simple function approximators. Williams' REINFORCE (1992) offered a Monte Carlo policy gradient with unbiased estimates but suffered from high variance and had no convergence theory for function approximation. The gap: a principled gradient-based method with convergence guarantees.

## Main idea

Represent the policy **explicitly** as a differentiable function $\pi(s, a, \theta)$ and update it by following the gradient of expected reward $\rho$ with respect to $\theta$. The key insight is that the gradient can be computed without modeling how policy changes affect the state distribution — only the action probabilities and their $Q$-values matter.

## Technical details

**Policy Gradient Theorem:**

$$
\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta} Q^\pi(s,a)
$$

where $d^\pi(s)$ is the stationary distribution under $\pi$. Notice: **no $\partial d^\pi / \partial \theta$ term** — the effect of policy changes on state visitation does not need to be modeled.

**Compatible Function Approximation (Theorem 2):**
If $f_w(s,a)$ approximates $Q^\pi(s,a)$ and satisfies:

$$
\frac{\partial f_w(s,a)}{\partial w} = \frac{1}{\pi(s,a)} \frac{\partial \pi(s,a)}{\partial \theta}
$$

then using $f_w$ in place of $Q^\pi$ still gives the true gradient. This bridges REINFORCE (high variance, unbiased) and actor-critic (lower variance, but biased) — a properly compatible critic eliminates the bias.

## Research takeaway

The gradient of expected reward can be estimated without differentiating through environment dynamics — only the policy's own parameters need gradients. Combined with compatible function approximation, this makes policy gradient methods both principled and practical for complex policies.

## Influence

This paper was the first to prove convergence of policy iteration with arbitrary differentiable function approximators. It established the theoretical foundation for:
- **Actor-critic methods** (Konda & Tsitsiklis, 2000) — using learned value functions as critics.
- **Natural Policy Gradient** (Kakade, 2002) — adding geometric awareness to the gradient direction.
- **TRPO** (Schulman et al., 2015) and **PPO** (Schulman et al., 2017) — stable policy updates with trust regions.
- **RLHF** (InstructGPT, 2022) — fine-tuning language models by treating the LM as a policy and human preferences as reward.

Every policy gradient algorithm in use today — from robotics to LLM alignment — traces its lineage back to this theorem.

## Modern perspective

The compatible function approximation condition (Theorem 2) is restrictive: it requires the critic features to align with the policy parameterization in a specific way. In practice, modern actor-critic methods (A2C, PPO, SAC) simply use a separate value network without enforcing compatibility — the empirical bias is small enough to ignore, and the simplicity gain is large.

The core theorem itself has held up flawlessly. It is now taught as standard material in graduate RL courses (e.g., Sutton & Barto, Chapter 13) and is implemented in every major RL library (Stable-Baselines3, Ray RLlib, CleanRL). The main practical limitation is variance — Monte Carlo gradient estimates can be noisy, which is why subsequent work (GAE, TD($\lambda$), n-step returns) focused on variance reduction rather than changing the theorem.
