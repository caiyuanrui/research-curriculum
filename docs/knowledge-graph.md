# Knowledge Graph

概念关系图——跨课程的知识关联。

## Concepts

### diffusion
- **Score Function** ([Hyvärinen 2005](diffusion/001-score-matching-hyvarinen-2005.md)) — Gradient of log-density; key concept for training unnormalized models
- **Score Matching** ([Hyvärinen 2005](diffusion/001-score-matching-hyvarinen-2005.md)) — Training objective that matches model score to data score
- **Denoising Autoencoder** ([Vincent 2011](diffusion/002-vincent-2011.md)) — Neural network that learns to reconstruct clean input from corrupted input
- **Denoising Score Matching** ([Vincent 2011](diffusion/002-vincent-2011.md)) — Equivalence between DAE reconstruction and score estimation
- **Tweedie's Formula** ([Vincent 2011](diffusion/002-vincent-2011.md)) — Statistical identity connecting posterior expectation and score

### llm-systems
- **Model Parallelism** ([Shoeybi 2019](systems/001-megatron-lm-shoeybi-2019.md)) — Splitting model parameters across GPUs within a single layer
- **Intra-layer Tensor Parallelism** ([Shoeybi 2019](systems/001-megatron-lm-shoeybi-2019.md)) — Column/row-wise partition of weight matrices
- **All-Reduce Communication** ([Shoeybi 2019](systems/001-megatron-lm-shoeybi-2019.md)) — Communication pattern at layer boundaries
- **Scaling Efficiency** ([Shoeybi 2019](systems/001-megatron-lm-shoeybi-2019.md)) — 76% efficiency on 512 GPUs for 8.3B parameter models

### labs
- **Agentic Coding** ([GPT-5.5 Tech Report](labs/001-gpt-5-5-technical-report.md)) — Frontier model achieving 82.7% on Terminal-Bench 2.0 for autonomous software engineering
- **Token Efficiency** ([GPT-5.5 Tech Report](labs/001-gpt-5-5-technical-report.md)) — Correlates reduced token usage with deeper task understanding
- **Destructive Action Avoidance** ([GPT-5.5 Tech Report](labs/001-gpt-5-5-technical-report.md)) — Safety metric for agentic rollouts: 52% perfect reversion rate
- **FrontierMath** ([GPT-5.5 Tech Report](labs/001-gpt-5-5-technical-report.md)) — 35.4% on Tier 4 math problems; discovered new Ramsey number proof
- **Multiturn Jailbreak Evaluation** ([GPT-5.5 Tech Report](labs/001-gpt-5-5-technical-report.md)) — Novel adversary simulation protocol replacing StrongReject

### multimodal
- **Contrastive Vision-Language Pretraining** ([CLIP 2021](multimodal/001-clip-radford-2021.md)) — Train dual encoders (image + text) with contrastive loss on 400M image-text pairs for zero-shot transfer
- **Zero-shot Image Classification** ([CLIP 2021](multimodal/001-clip-radford-2021.md)) — Use natural language prompts as classifiers without any dataset-specific training
- **Dual-Encoder Architecture** ([CLIP 2021](multimodal/001-clip-radford-2021.md)) — Separate image encoder (ResNet/ViT) and text encoder (Transformer) aligned via cosine similarity
- **Prompt Engineering for Vision** ([CLIP 2021](multimodal/001-clip-radford-2021.md)) — Template-based prompting ("a photo of a {class}") significantly improves zero-shot accuracy
- **InfoNCE Contrastive Loss** ([CLIP 2021](multimodal/001-clip-radford-2021.md)) — Symmetric cross-entropy over N×N similarity matrix with learned temperature τ

## Cross-Course Connections

- **Model Parallelism → Large-scale Training** — The scaling techniques pioneered in Megatron-LM (tensor parallelism) are prerequisites for training the massive diffusion models covered in the diffusion course, enabling multi-billion parameter vision models
- **All-Reduce → Distributed Systems** — The communication-efficient patterns established here underpin modern LLM serving systems (vLLM, SGLang) and their distributed inference architectures
- **Agentic Coding → LLM Systems** — GPT-5.5's 82.7% on Terminal-Bench 2.0 requires underlying infrastructure (vLLM, FlashAttention, PagedAttention) studied in the LLM Systems course
- **Bottleneck Elimination Pattern** ([cross-domain week-23](weekly-review/cross-domain-week-23.md)) — Score matching, denoising score matching, and Megatron-LM all identify a core computational bottleneck and solve it through mathematical/engineering reframing rather than brute force
- **Denoising as Universal Learning Principle** ([cross-domain week-23](weekly-review/cross-domain-week-23.md)) — Vincent's DAE-score connection, GPT-5.5's RL from internal CoT, and modern diffusion training all share the "add noise → learn to denoise → learn distribution/reasoning" pattern
- **Multi-scale Gradient Information** ([cross-domain week-23](weekly-review/cross-domain-week-23.md)) — Score matching operates on data-space gradients (∇ₓ log p), Megatron on parameter-space gradients (∇_θ ℒ), GPT-5.5 on reasoning-space "gradients" (RL reward signals)

### agent
- **ReAct Prompting** ([Yao 2023](agent/001-react-yao-2022.md)) — Interleaved reasoning traces and task-specific actions for LLM agents
- **Reasoning-Action Synergy** ([Yao 2023](agent/001-react-yao-2022.md)) — Bidirectional benefit: reasoning guides action planning, actions ground reasoning in external observation
- **Thought-Action-Observation Loop** ([Yao 2023](agent/001-react-yao-2022.md)) — Canonical format (Thought → Action → Observation → Thought) underlying modern agent frameworks
- **ReAct + CoT Hybrid** ([Yao 2023](agent/001-react-yao-2022.md)) — Combining parametric knowledge with external grounding outperforms either alone
- **In-context Agent Design** ([Yao 2023](agent/001-react-yao-2022.md)) — Zero-additional-training agents via few-shot prompting with 1-2 examples

```mermaid
graph LR
    SM[Score Matching] --> DAE[Denoising Autoencoders]
    DAE --> SG[Score-Based Gen Models]
    SM --> SSM[Sliced Score Matching]
    TP[Tensor Parallelism] --> PS[Parallel Systems]
    PS --> LI[LLM Inference]
    TP --> 3D[3D Parallelism]
    3D --> Z[ZeRO/FSDP]
    AC[Agentic Coding] --> LI
    AC --> SS[Safety Systems]
```
