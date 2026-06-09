---
title: "Learning Transferable Visual Models From Natural Language Supervision"
authors: "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever"
year: 2021
venue: "International Conference on Machine Learning (ICML 2021)"
date: 2021-02-26
course: "multimodal"
tags: [vision-language, contrastive-learning, zero-shot, CLIP, multimodal-pretraining]
phase: "Vision-Language Pretraining"
order: 1
---

# Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
**Year:** 2021 | **Venue:** ICML 2021
**Link:** [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

---

## Historical Context

Before CLIP, computer vision was dominated by supervised learning on fixed-label datasets like ImageNet (1,000 categories). Every new visual concept required collecting labeled examples and retraining a classifier — a brittle, unscalable paradigm. Natural language processing had already shown the power of large-scale pretraining (GPT, BERT), but vision was stuck in closed-vocabulary classification. The fundamental question was: *Can we learn visual representations directly from natural language supervision, at internet scale, without any manual annotations?*

## Core Idea

CLIP trains a **dual-encoder architecture** — an image encoder (ResNet or ViT) and a text encoder (Transformer) — with a **contrastive loss** on 400 million image-text pairs crawled from the internet. The training objective is simple: for a batch of $N$ image-text pairs, maximize the cosine similarity of the $N$ correct pairs while minimizing it for the $N^2 - N$ incorrect ones, using a symmetric cross-entropy loss with temperature scaling.

$$
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left(\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)} + \log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_j, T_i)/\tau)}\right)
$$

where $\text{sim}(I, T) = \frac{\mathbf{v}_I \cdot \mathbf{v}_T}{\|\mathbf{v}_I\|\|\mathbf{v}_T\|}$ and $\tau$ is a learned temperature parameter.

Why it works:
1. **Contrastive efficiency** — The contrastive objective is far more sample-efficient than predicting exact text (which is high-dimensional and sparse). Learning to discriminate which caption matches which image captures rich semantic alignment without needing generative capability.
2. **Natural language as a flexible interface** — After pretraining, any visual concept can be described in natural language and used for zero-shot classification via prompt engineering (e.g., "a photo of a {class}"). This removes the need for dataset-specific classifiers.
3. **Massive data scale** — 400M pairs from the internet provide orders of magnitude more supervision than any manually annotated dataset, covering a vast range of visual concepts beyond fixed taxonomies.

Key design choices:
- **Contrastive rather than generative** — They tried image-captioning-style objectives but found contrastive learning to be 10-100x more efficient.
- **Prompt engineering** — Simple templates like "a photo of a {label}" significantly outperform raw class names, and ensembling across multiple prompts (e.g., "a photo of a big {label}") helps further.
- **Zero-shot via natural language** — For a given dataset, they convert its label set into natural language descriptions and pick the highest-scoring one.

## Influence

CLIP fundamentally reshaped computer vision and multimodal AI:

- **Zero-shot vision became practical** — Prior work had limited zero-shot capabilities; CLIP demonstrated broad zero-shot transfer across 30+ datasets, matching ResNet-50 on ImageNet without any ImageNet training.
- **Inspired a new paradigm** — Every major vision-language model since 2021 builds on CLIP's contrastive pretraining approach: ALIGN (Google), SigLIP (Google), OpenCLIP (community), and the vision encoders in LLaVA, GPT-4V, and Gemini.
- **Foundation for generative AI** — CLIP embeddings power DALL·E 2, Stable Diffusion (via CLIP text encoders), and image-text retrieval systems. CLIP's text encoder became a standard component in text-to-image generation.
- **ImageNet accuracy leap** — By 2023, improved CLIP variants surpassed 85% zero-shot on ImageNet, outperforming the original model by 10+ percentage points.
- **Community scale** — OpenCLIP reproduced training at even larger scales; LAION-400M and LAION-5B datasets were built for CLIP-style training.

## Modern Perspective

CLIP's core ideas remain foundational:

**Still relevant:**
- Contrastive vision-language pretraining is the standard recipe for multimodal encoders. SigLIP improved the objective (sigmoid loss instead of softmax), but the dual-encoder architecture is unchanged.
- Zero-shot evaluation via prompt engineering is still the standard protocol for vision-language models.
- The scaling insight — more data + bigger models = better transfer — has been validated repeatedly.

**Evolution:**
- Better losses: SigLIP replaces softmax normalization with sigmoid loss, removing the need for global batch normalization and enabling training with smaller batch sizes.
- Better architectures: ViT-based encoders (notably in SigLIP and EVA-CLIP) surpass the original ResNet-based CLIP.
- Better data: Filtering and deduplication strategies (e.g., in DataComp) improve efficiency.
- Multimodal LLMs: Rather than using CLIP only for zero-shot classification, LLaVA and similar models fuse CLIP vision features into LLM token space for open-ended visual reasoning.

**Limitations revealed:**
- CLIP struggles with fine-grained visual attributes (color, count, position).
- It exhibits social biases from web data.
- Zero-shot classification requires careful prompt engineering — performance varies significantly with prompt choice.
- The contrastive objective captures alignment but not generative understanding.

## Research Takeaway

CLIP's most important lesson is about **scaling supervision through natural language**: rather than designing better architectures, CLIP succeeded by leveraging an existing abundant data source (image-text pairs) with a simple, well-matched objective. The key research skill here is identifying the right *source of signal* — the 400M pairs were not carefully curated, but the contrastive objective was precisely calibrated to extract maximum value from noisy, uncurated data. This principle — choose the data source and loss function together, not independently — applies directly to many domains beyond vision-language.

## Optional Deep Dive

The contrastive objective in CLIP is worth studying in detail. The batch construction of $N \times N$ similarity pairs creates $N$ positives and $N^2 - N$ negatives. The multi-class N-pair loss (also called InfoNCE or contrastive loss with temperature) has deep connections to mutual information estimation. The temperature parameter $\tau$ plays a critical role: if $\tau$ is too large, the loss becomes uniform across negatives and training lacks discrimination; if too small, the model focuses on just the hardest negative and becomes unstable. CLIP learned $\tau$ as a log-parameterized scalar, automatically finding the right scale for the contrastive task. This temperature scaling insight was inherited by nearly all subsequent contrastive vision-language models.
