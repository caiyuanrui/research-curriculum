# Policy Gradient Foundations

**Question:** What RL foundations are necessary for agentic training?

## Overview

Policy gradient methods optimize a policy *directly* by following the gradient of expected reward, bypassing the value-function approach that dominated early RL. This topic covers the theoretical foundations from the Policy Gradient Theorem through to PPO — the algorithm used in modern RLHF.

## Reading List

| # | Paper | Year | Status |
|---|-------|------|--------|
| 1 | [Policy Gradient Methods for RL with Function Approximation](1999-12-03-policy-gradient-theorem.md) — Sutton et al. | 2000 | ✅ Done |
| 2 | A Natural Policy Gradient — Kakade | 2002 | ⬜ Unread |
| 3 | [Trust Region Policy Optimization (TRPO)](2015-02-19-trust-region-policy-optimization.md) — Schulman et al. | 2015 | ✅ Done |
| 4 | Generalized Advantage Estimation (GAE) — Schulman et al. | 2016 | ⬜ Unread |
| 5 | Proximal Policy Optimization (PPO) — Schulman et al. | 2017 | ⬜ Unread |
| 6 | Deterministic Policy Gradient (DPG) — Silver et al. | 2014 | ⬜ Unread |

## Summary

The path from Sutton's Policy Gradient Theorem to PPO traces a single narrative: **how to take a good gradient step in policy space**. Sutton proved the gradient exists and can be estimated. Kakade added geometric awareness (natural gradient). TRPO turned that into a practical algorithm with monotonic improvement. GAE gave us reliable advantage estimates. PPO made it all simple enough to scale to LLMs.
