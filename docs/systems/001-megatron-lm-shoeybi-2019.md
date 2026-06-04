---
title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
authors: "Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro"
year: 2019
venue: "SC 2020 (Supercomputing)"
date: 2019-09-17
course: llm-systems
tags: [model-parallelism, distributed-training, transformers, megatron]
phase: "megatron"
order: 1
---

# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

**Authors:** Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro
**Year:** 2019 | **Venue:** SC 2020 (Supercomputing)
**Link:** [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)

---

## Historical Context

By 2019, language models had grown dramatically in size — OpenAI's GPT-2 (1.5B parameters) had shown the power of scaling, but training such models was becoming increasingly difficult due to GPU memory constraints. A single NVIDIA V100 GPU has only 32GB of memory, far too small to hold even the parameters of a multi-billion parameter model, let alone the activations and optimizer states required during training.

Existing distributed training approaches relied on **data parallelism** (each GPU holds a full copy of the model, processes different data shards, and synchronizes gradients). This approach fails when the model itself exceeds a single GPU's memory — there simply isn't room for the complete model on any one device. What was needed was a way to **split the model itself across devices** so that no single GPU had to hold the entire thing.

Prior work on model parallelism (placing different layers on different devices, i.e., pipeline parallelism) existed but suffered from inherent inefficiencies — devices spent much of their time idle waiting for upstream computations, and the communication patterns were complex.

## Core Idea

The key insight of Megatron-LM is remarkably simple: **split the individual transformer layers across GPUs by partitioning the weight matrices column-wise or row-wise**, requiring communication only at layer boundaries.

### How it works

For a standard transformer layer with a Multi-Head Attention (MHA) block followed by a Feed-Forward Network (FFN):

**1. Attention block (column-wise partition):**

Each attention head's Query ($Q$), Key ($K$), and Value ($V$) projection matrices are computed independently by different GPUs. In standard MHA, we have:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$. Under Megatron's scheme, each GPU holds a column slice of each weight matrix:

$$W_Q = [W_{Q,1}, W_{Q,2}, ..., W_{Q,g}]$$

so GPU $i$ computes $Q_i = XW_{Q,i}$, $K_i = XW_{K,i}$, $V_i = XW_{V,i}$, and then performs attention on its subset of heads: $\text{Attention}_i(Q_i, K_i, V_i)$. After each GPU computes its attention output, an **all-reduce** operation combines the results.

**2. FFN block (row-wise partition):**

The FFN has two weight matrices: $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$ (expansion) and $W_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$ (contraction). The FFN computation is:

$$\text{FFN}(X) = \text{GELU}(XW_1)W_2$$

Under Megatron's scheme, $W_1$ is split column-wise across GPUs, so GPU $i$ holds $W_{1,i} \in \mathbb{R}^{d_{\text{model}} \times (4d_{\text{model}}/g)}$ and computes $Z_i = \text{GELU}(XW_{1,i})$. Then $W_2$ is split row-wise, with GPU $i$ holding $W_{2,i} \in \mathbb{R}^{(4d_{\text{model}}/g) \times d_{\text{model}}}$ and computing $Y_i = Z_i W_{2,i}$. An all-reduce combines all $Y_i$.

### Communication pattern

The beauty of this design is that **communication is needed only at the boundaries between transformer layers**. Within each layer, all operations are independent on each GPU until the final all-reduce. This means:

- Total communication per layer: **2 all-reduce operations** (one after self-attention, one after FFN)
- No communication during the attention computation itself
- No complex scheduling or load-balancing logic required

### Orthogonality to pipeline parallelism

Megatron's intra-layer parallelism is **orthogonal to pipeline parallelism** (inter-layer splitting). They can be combined: use Megatron to split each large layer across GPUs within a node (where intra-node bandwidth is high via NVLink), and use pipeline parallelism to split layers across nodes (where bandwidth is lower).

## Influence

Megatron-LM fundamentally changed the landscape of large-scale model training:

1. **Made billion-parameter training accessible** — The approach was implemented entirely in native PyTorch with just a few `torch.distributed` communication primitives. No custom CUDA kernels, no compiler changes. Any team using PyTorch could adopt it.

2. **Direct precursor to NVIDIA's NeMo framework** — The Megatron codebase evolved into the Megatron-LM core that powers NVIDIA's NeMo Megatron framework, used to train models like Megatron-Turing NLG (530B).

3. **Sparked the "Model Parallelism Renaissance"** — After Megatron, the community rapidly developed more advanced schemes: DeepSpeed's ZeRO (eliminates redundant data), GSPMD/XLA (automatic sharding), FSDP (fully sharded data parallel), and tensor parallelism in frameworks like Colossal-AI.

4. **Practical scaling recipe** — Demonstrated 76% scaling efficiency on 512 GPUs, proving the approach was practical at production scale.

5. **SOTA results** — Set new records on WikiText103 (perplexity 10.8 vs previous 15.8), LAMBADA (66.5% vs 63.2%), and RACE (90.9% vs 89.4%), showing that scaling alone drove real benchmark improvements.

## Modern Perspective

Looking back from 2026, Megatron-LM's contributions remain foundational:

**Still relevant:**
- The intra-layer tensor parallelism scheme is **still the standard** in virtually every large-scale training framework (NVIDIA NeMo, Megatron-DeepSpeed, Hugging Face Transformers, Colossal-AI)
- The communication-efficient design (all-reduce at layer boundaries only) remains optimal for dense transformers
- The insight that model parallelism is indispensable when models exceed single-GPU memory is now taken for granted

**Evolved:**
- Modern systems combine tensor parallelism (Megatron-style) with ZeRO-style parameter sharding and pipeline parallelism in a 3D-parallel configuration
- FlashAttention has changed the attention implementation, but Megatron's parallel decomposition of attention heads is unchanged
- The original PyTorch-only approach has been supplemented by more specialized communication libraries (NCCL, RCCL)

**Limitations acknowledged:**
- Megatron's approach requires the model to be a transformer — it exploits the specific structure of attention heads and FFN dimensions
- For non-transformer architectures (e.g., Mamba, RWKV), different parallelism strategies are needed
- The approach requires careful load balancing — uneven head assignments could create stragglers

## Research Takeaway

Megatron-LM is a masterclass in **identifying the minimum viable abstraction** for a hard problem. The authors didn't try to build a general-purpose auto-parallelizing compiler. They observed that transformers have a natural parallelism structure (attention heads, FFN hidden dimensions) and exploited that directly. The result: a few hundred lines of PyTorch code that enabled training 8-billion parameter models.

The lesson: **the best systems contributions often come from deeply understanding the specific model architecture rather than building general solutions.**

## Optional Deep Dive

**The all-reduce communication cost model.** In Megatron-LM, each all-reduce on $d$ elements across $g$ GPUs has a cost of:

$$\text{Time} = 2 \cdot \alpha \cdot \log(g) + 2 \cdot \beta \cdot d$$

where $\alpha$ is the latency per message and $\beta$ is the bandwidth per element. The factor of 2 comes from the ring all-reduce algorithm (scatter-reduce + all-gather). For transformer layers with hidden dimension $d_{\text{model}}$ and sequence length $s$, the attention all-reduce transfers $bd_{\text{model}}s$ elements per layer (batch size $b$, sequence length $s$), and the FFN all-reduce transfers $4bd_{\text{model}}s$ elements. This means the **communication-to-computation ratio** stays nearly constant as the model scales, making the scheme highly scalable.

A subtle but critical detail: the authors discovered that for BERT training, **LayerNorm placement matters enormously**. In their early experiments, placing LayerNorm inside the residual path caused training instability at large scales. Moving LayerNorm to the "pre-norm" position (before each sub-layer, as used in GPT-2) fixed this — an insight that later became standard practice in nearly all transformer architectures.
