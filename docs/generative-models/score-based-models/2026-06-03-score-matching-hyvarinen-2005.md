---
title: Estimation of Non-Normalized Statistical Models by Score Matching
authors: Aapo Hyvärinen
year: 2005
venue: Journal of Machine Learning Research (JMLR), Vol. 6
date: 2026-06-03
course: "generative-models-course"
tags: [score-matching, energy-based-models, classic]
phase: "score-based-models"
---

# Estimation of Non-Normalized Statistical Models by Score Matching

**Aapo Hyvärinen** — JMLR, 2005

## 📜 Historical Context

**Why did this work appear?**

In the early 2000s, probabilistic modeling faced a fundamental bottleneck: most interesting probabilistic models involve an intractable partition function (normalizing constant). Models like Boltzmann machines, Markov random fields, and energy-based models require computing Z = ∫ exp(-E(x)) dx over all possible configurations, which is computationally infeasible for high-dimensional data. Traditional approaches relied on Markov Chain Monte Carlo (MCMC) sampling to approximate gradients, but MCMC is slow, has convergence diagnostics issues, and behaves poorly in high dimensions.

**What problem was the field facing?**

Researchers needed a way to train unnormalized density models — models where p(x) = (1/Z) · p̃(x) with unknown Z — without ever computing Z or running MCMC. Maximum likelihood directly requires Z; contrastive divergence (Hinton, 2002) reduced but didn't eliminate the reliance on sampling. There was a clear gap for a training objective that was both computationally tractable and statistically well-founded.

## 💡 Core Idea

**What is the core idea?**

Instead of matching probabilities ($p_{\text{model}}(x) \approx p_{\text{data}}(x)$), match the gradient of the log-density — what Hyvärinen called the **"score function"**: $\psi(x) = \nabla_x \log p(x)$. The score of the true data distribution is $\nabla_x \log p_{\text{data}}(x)$; the score of the model is $\nabla_x \log \tilde{p}_{\text{model}}(x)$ (note: no $Z$ appears because $\partial/\partial x \log(\tilde{p}/Z) = \partial/\partial x \log \tilde{p} - \partial/\partial x \log Z$, and $\partial/\partial x \log Z = 0$ since $Z$ does not depend on $x$). The normalizing constant $Z$ entirely vanishes in the score formulation.

The **Score Matching** objective minimizes the expected squared distance between the model's score and the data score:

$$
J(\theta) = \frac{1}{2} \int p_{\text{data}}(x) \cdot \| \nabla_x \log p_{\text{model}}(x; \theta) - \nabla_x \log p_{\text{data}}(x) \|^2 dx
$$

Through integration by parts, Hyvärinen showed this can be rewritten into a form that depends only on the model (and its first and second derivatives), with the data appearing only through expectations over $p_{\text{data}}$:

$$
J(\theta) = \mathbb{E}_{p_{\text{data}}}\left[ \frac{1}{2} \cdot \| \nabla_x \log p_{\text{model}}(x; \theta) \|^2 + \nabla_x^2 \log p_{\text{model}}(x; \theta) \right] + \text{constant}
$$

The "constant" term depends only on $p_{\text{data}}$ and can be ignored during optimization.

**Why does it work?**

The score function captures all information about the density up to a multiplicative constant — it tells you which direction to move in x-space to increase probability. Two densities with the same score can differ only by a constant factor (i.e., the normalizing constant). By matching the score, you effectively recover the full shape of the density without ever computing Z. The integration-by-parts trick eliminates the dependence on the unknown data score — all you need are samples.

## 🔗 Impact

**What subsequent work did it influence?**

1. **Denoising Autoencoders as Score Matching (Vincent, 2011):** Vincent proved that denoising autoencoders with small Gaussian corruption are implicitly performing score matching — the reconstruction objective equals score matching under specific conditions. This bridged unsupervised representation learning and density estimation.

2. **Generative Modeling via Gradients (Song & Ermon, 2019–2021):** The most direct line is the Noise Conditional Score Network (NCSN) and score-based diffusion models. These models train neural networks to estimate scores at multiple noise levels, then generate samples via Langevin dynamics — exactly the framework Hyvärinen's score matching enables.

3. **Sliced Score Matching (Song et al., 2019):** Extended score matching to high-dimensional data where computing the Hessian trace (∇² term) is O(d) or O(d²) expensive, by projecting the score onto random directions.

4. **Energy-Based Models:** Revived interest in training EBMs without MCMC, using score matching and its variants.

5. **Modern Diffusion Models:** DDPM (Ho et al., 2020), Score SDE (Song et al., 2021), and EDM (Karras et al., 2022) are all, at their core, score matching objectives operating at multiple noise scales.

**Research line formed:**

Score Matching (2005) → Denoising AE connection (2011) → Sliced SM (2019) → NCSN (2019) → DDPM (2020) → Score SDE (2021) → EDM (2022) → Consistency Models (2023) → Modern systems.

## 🔭 Modern Perspective

**Looking back from 2026:**

**Still important:**
- The core insight that the normalizing constant can be sidestepped by working in gradient space is more relevant than ever. Modern diffusion models — including Sora, Flux, and SD3 — all rely on score-matching objectives.
- The integration-by-parts reformulation that eliminates the unknown data score is a classic trick that appears throughout ML (e.g., in variational inference, Stein's method).
- Score matching remains a primary training objective for energy-based models and unnormalized density estimation.

**What evolved:**
- The original formulation's O(d) or O(d²) Hessian trace computation is impractical for high-dimensional data like images. The field has moved to sliced score matching (random projections), denoising score matching (add noise, match the denoising direction), and implicit score matching via denoising autoencoders.
- Modern practice uses score matching at multiple noise levels (score-based diffusion), which wasn't contemplated in the original 2005 paper. Hyvärinen's formulation estimates the score at a single point in x-space; diffusion models estimate scores across the entire noise-conditioned data manifold.
- The original paper focused on continuous data. Discrete score matching has since been developed (e.g., for text).

## 🎯 Researcher Takeaways

1. **Elegant problem reformulation:** The paper demonstrates that when a bottleneck seems fundamental (intractable partition function), reformulating the problem in a different space (gradient space) can dissolve it. This is a masterclass in creative mathematical reframing.

2. **The power of integration by parts:** A calculus technique taught in undergraduate courses becomes the core trick enabling modern generative AI. Never underestimate the impact of a well-placed mathematical identity.

3. **Simplicity first:** Hyvärinen's estimator is remarkably simple — it only requires first and second derivatives of the log-model-density, which are computable via automatic differentiation in modern frameworks. The statistical analysis (consistency, efficiency) is rigorous and complete.

4. **25-year trajectory:** The 2005 paper felt like a niche contribution to unsupervised learning. It took 15+ years for the broader community to connect score matching to diffusion models and realize its transformative potential. This is a patience-and-persistence lesson.

## 🔬 Deep Exploration

**The trace of the Hessian and computational bottlenecks.**

The objective simplifies to:

$$
J(\theta) = \mathbb{E}_{p_{\text{data}}}\left[ \frac{1}{2} \| \nabla_x \log \tilde{p}_{\text{model}}(x; \theta) \|^2 + \nabla_x^2 \log \tilde{p}_{\text{model}}(x; \theta) \right].
$$

The second term, $\nabla_x^2 \log \tilde{p}_{\text{model}}$ (the trace of the Hessian, or Laplacian), is the computational bottleneck. For a model with d-dimensional input, the Hessian is a d×d matrix — computing its trace naively costs O(d²) or requires a separate backprop for each dimension (O(d) passes). For images (d ≈ 3×256×256 = 196,608), this is prohibitive.

The field's response to this bottleneck created multiple research branches:
- **Denoising Score Matching (Vincent, 2011):** Replaces the Hessian trace with a denoising objective that is O(1) per sample.
- **Sliced Score Matching (Song et al., 2019):** Uses Hutchinson's trace estimator — E_{v~N(0,I)}[ v^T H v ] — to approximate the trace with O(1) vector-Hessian products (computed via two automatic differentiation passes).
- **Denoising Diffusion Score Matching (Song & Ermon, 2019):** Employs a noise-perturbed version of denoising score matching, eliminating the Hessian entirely.

Deeply understanding why this trace term appears, how it arises from the integration by parts, and how each method avoids it reveals a lot about the trade-offs in the score matching family.
