---
title: "Introducing GPT-5.5 — A New Class of Intelligence for Real Work"
authors: "OpenAI"
year: 2026
venue: "OpenAI Technical Report & System Card"
date: 2026-04-23
course: "research-lab-course"
tags: [openai, llm, agentic-coding, safety, system-card, gpt-5.5]
phase: "weekly-lab-note"
order: 1
---

# Introducing GPT-5.5 — A New Class of Intelligence for Real Work

**Authors:** OpenAI
**Year:** 2026 | **Venue:** OpenAI Technical Report & System Card
**Link:** [OpenAI Blog](https://openai.com/index/introducing-gpt-5-5/) | [System Card](https://deploymentsafety.openai.com/gpt-5-5)

---

## Historical Context

By early 2026, the frontier of large language models had reached a plateau where raw scaling alone no longer yielded commensurate performance gains. GPT-5 (August 2025) and GPT-5.4 had established strong baselines, but the community was increasingly focused on **agentic usage patterns** — models that could operate autonomously over long horizons, use tools, write code, and self-correct. The challenge was no longer just about making models bigger or faster at single-turn tasks; it was about making them **reliable, persistent, and safe** in multi-step, tool-augmented workflows. OpenAI's GPT-5.5 directly addresses this gap: it is the first model explicitly co-designed for agentic coding, knowledge work, and scientific research at scale.

## Core Idea

GPT-5.5 is built on the same underlying architecture as GPT-5.4 but achieves substantially higher intelligence through improvements in **training methodology, reasoning training, and inference-time compute allocation**. The key innovations include:

**1. Reinforcement Learning from Internal Chain-of-Thought:** The model is trained to produce and follow extended internal reasoning traces, improving its ability to decompose complex tasks, check its own work, and persist through long agentic rollouts.

**2. Token Efficiency:** GPT-5.5 consistently uses **fewer tokens** than GPT-5.4 to achieve the same or better results on coding and knowledge tasks — a hallmark of deeper understanding rather than brute-force computation.

**3. Co-Design with NVIDIA GB200/GB300 NVL72 Systems:** The model, training stack, and serving infrastructure were co-designed for maximum performance on NVIDIA's latest hardware, yielding the same per-token latency as GPT-5.4 despite higher intelligence.

**4. Strongest Safeguards to Date:** The release includes a comprehensive system card with evaluations across cybersecurity, biology, hallucination, deception, and jailbreak robustness — setting a new bar for pre-deployment safety testing.

**Performance Highlights:**

| Domain | Benchmark | GPT-5.5 | GPT-5.4 | Δ |
|--------|-----------|---------|---------|---|
| Agentic Coding | Terminal-Bench 2.0 | **82.7%** | 75.1% | +7.6% |
| Agentic Coding | Expert-SWE (Internal) | **73.1%** | 68.5% | +4.6% |
| Knowledge Work | GDPval (wins/ties) | **84.9%** | 83.0% | +1.9% |
| Computer Use | OSWorld-Verified | **78.7%** | 75.0% | +3.7% |
| Scientific Research | GeneBench | **25.0%** | 19.0% | +6.0% |
| Scientific Research | FrontierMath Tier 4 | **35.4%** | 27.1% | +8.3% |
| Scientific Research | BixBench (Bioinformatics) | **80.5%** | 74.0% | +6.5% |

The model has a **1M token context window** (API) and **400K tokens in Codex**, enabling it to work with entire codebases and long documents in a single pass.

## Influence

GPT-5.5 represents a significant step in several ongoing trends:

- **Agentic Coding as a Primary Use Case:** 85%+ of OpenAI's own workforce uses Codex weekly. The model's ability to "stay on task for significantly longer without stopping early" (Cursor CEO Michael Truell) signals that AI coding assistants are transitioning from single-turn code completions to multi-hour autonomous engineering sessions.

- **Real-World Validation at Scale:** The internal use cases (71,637 pages of K-1 tax forms reviewed, automated Slack agents, weekly business reports saving 5-10 hours/week) demonstrate that frontier models are crossing into **enterprise production workflows**, not just prototyping.

- **Scientific Research Contributions:** The discovery of a new proof about off-diagonal Ramsey numbers (verified in Lean) marks one of the first concrete mathematical contributions from an LLM — not just solving known problems but advancing mathematical knowledge.

- **Safety Transparency Standards:** The GPT-5.5 System Card is among the most comprehensive pre-deployment safety documents published, covering cybersecurity capabilities, biological risk, jailbreak robustness, prompt injection, hallucination rates, and misalignment in agentic coding — setting a benchmark for the industry.

## Modern Perspective

Looking at GPT-5.5 from the present vantage point, several observations stand out:

**Still Relevant:**
- The **token efficiency vs. raw capability** trade-off remains central to frontier model design. Models that achieve more with less compute will define the next generation.
- The **safety-as-transparency** approach (system cards with detailed per-category evaluations) has become industry standard.
- **Agentic persistence** — the ability to execute long-horizon tasks without human intervention — is now a primary evaluation axis for frontier models.

**Evolving:**
- The distinction between "thinking" and "fast" modes (GPT-5.5 vs GPT-5.5 Thinking vs GPT-5.5 Pro) is increasingly fluid — future models may unify these into a single continuous capability.
- Computer use (OSWorld, GUI agent benchmarks) is still in early stages for complex multi-app workflows, though GPT-5.5's 78.7% on OSWorld-Verified marks significant progress.

**Limitations Observed:**
- The model still shows a slight regression in **self-harm** and **emotional reliance** categories compared to GPT-5.4-thinking, indicating that safety alignment for sensitive scenarios remains an open challenge.
- While hallucination rates decreased at the claim level (23% improvement), the model makes more claims per response, making response-level factual accuracy only marginally better.

## Research Takeaway

Three lessons stand out for researchers:

1. **Reasoning training is the new scaling law.** The improvements in GPT-5.5 come primarily from better training methodology (RL from internal chain-of-thought) rather than larger model size or more data. Researchers should focus on **training techniques that induce structured reasoning** rather than brute-force scaling.

2. **Token efficiency is a signal of genuine understanding.** When a model achieves better results with fewer tokens (as GPT-5.5 does on coding benchmarks), it indicates deeper conceptual clarity. This should be a primary optimization target for future models.

3. **Production deployment is the ultimate testbed.** The most valuable evaluations are not benchmarks but real-world workflows — tax form analysis, codebase maintenance, scientific data analysis. Researchers should design evaluation suites that capture **multi-step, long-horizon task completion** rather than single-turn accuracy.

## Optional Deep Dive

The **GPT-5.5 System Card's evaluation methodology** deserves close study. It introduces several novel evaluation protocols:

- **Dynamic Multi-Turn Evaluations:** Adversarial user simulations that run extended conversations (not just single prompts) to catch policy violations that emerge over time — a crucial technique for agentic models that operate over long rollouts.

- **Destructive Action Avoidance:** A new metric measuring how well the model can revert *only its own changes* while preserving user modifications — critically important for code-editing and document-writing agents. GPT-5.5 improved perfect reversion from 18% to 52%.

- **Multiturn Jailbreak Robustness:** Replacing the prior StrongReject benchmark with a more challenging interactive attack simulation that tests sophisticated multi-step persuasion strategies.

- **Internal Misalignment in Agentic Coding:** A novel evaluation that resamples model behavior from fixed trajectory prefixes to detect subtle misalignment that only emerges in agentic contexts (e.g., the model overwriting user code instead of appending).

These methodological advances are arguably as important as the model's performance gains — they define how future agentic AI systems should be evaluated before deployment.
