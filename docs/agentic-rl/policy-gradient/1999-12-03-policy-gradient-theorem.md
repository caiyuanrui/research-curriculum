# Policy Gradient Methods for Reinforcement Learning with Function Approximation

Date: 1999-12-03 (NeurIPS 1999)  
Course: agentic-rl-course  
Topic: policy-gradient  
Paper: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour / NeurIPS 1999  
Link: https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation

## Why this paper today?

This is the foundational paper for all policy gradient methods used in modern RLHF (InstructGPT, RLHF for LLMs). Understanding the Policy Gradient Theorem is a prerequisite for everything else in this course.

## Core problem

Value-function-based RL (Q-learning, Sarsa, DP) with function approximation has no convergence guarantees — small changes in estimated values can cause discontinuous policy switches, making the optimization path unstable. The field needed a principled way to optimize policies directly.

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

## Key results

- First convergence proof for policy iteration with **arbitrary differentiable function approximation** to a locally optimal policy.
- The gradient does not need to differentiate through state dynamics — a breakthrough that made policy gradient methods practical.
- Compatible function approximation theoretically justifies using learned value functions (critics) without introducing bias.

## Limitations

- The compatible condition is restrictive — in practice, people just use a separate value network anyway, and the empirical bias is acceptable.
- Convergence is to a **local** optimum only.
- Analysis assumes the policy is **differentiable** everywhere; discrete action distributions (softmax) satisfy this, but the gradient variance can still be high.

## Connection to this course

This paper is the theoretical bedrock. Every subsequent method in this course — TRPO, PPO, GAE, and ultimately RLHF — derives from this theorem. When InstructGPT uses PPO to fine-tune language models, it is directly applying the principle: "parameterize the policy, collect experience, estimate the gradient of expected reward, update."

## Notes for future reading

- REINFORCE (Williams, 1992) — the original Monte Carlo policy gradient; unbiased but high variance.
- Actor-Critic methods (Konda & Tsitsiklis, 2000) — using a learned value function to reduce variance.
- Natural Policy Gradient (Kakade, 2002) — next paper in this reading list; addresses the "how big a step" question.
- The discount factor $\gamma$ and horizon tradeoff in advantage estimation.
